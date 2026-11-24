"""`brain`: create a business, index it, ask as a person, run the eval suites.

`--as` is a demo identity, as in agentgate: this unit has no login. Live answers need the
operator's provider keys file (JSON, provider name to key, mode 0600), which only the
agentgate proxy process reads.
"""

import json
import logging
import signal
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Sequence
from contextlib import ExitStack
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import anyio
import typer
import yaml

from brain.answer.compose import Answer
from brain.answer.models import ModelError
from brain.evals.ablation import answer_rows, retrieval_rows, summarize
from brain.evals.agreement import (
    chunk_sources,
    format_agreement,
    labelling_sheet,
    load_labels,
    read_report,
    score,
)
from brain.evals.golden import GoldenSet, load_golden
from brain.evals.permissions import expected_visibility, format_report, load_suite, run_suite
from brain.evals.report import (
    answer_fingerprint,
    build_report,
    check_report,
    check_retrieval,
    format_summary,
    load_thresholds,
    newest_answer_report,
    readme_block,
    require_clean_tree,
    write_report,
)
from brain.evals.report import write_readme as write_evals_readme
from brain.gate.policy import GateError, InvariantError, open_business_world, prepare_world
from brain.identity.people import load_directory, load_people
from brain.ingest.embedding import BgeM3Embedder, Embedder, HashEmbedder
from brain.ingest.index import index_models, rebuild_index, require_week_zero
from brain.ingest.items import Item, load_items
from brain.ingest.sync import Applier
from brain.ingest.versions import load_not_versions, version_problems
from brain.ingest.worker import WorkerBusy, run_worker
from brain.interfaces.mcp.server import build_server
from brain.interfaces.mcp.tools import DemoCaller, Tools
from brain.redteam.mcp import McpSuite, reachable
from brain.redteam.report import baseline_problems, format_table, write_readme
from brain.redteam.report import markdown as markdown_block
from brain.redteam.report import write_baseline as write_baseline_file
from brain.redteam.runner import Runner
from brain.redteam.scenarios import load_decoys, load_scenarios
from brain.runtime import ModelMode, open_business, open_run, serve_runs
from brain.search.query import PRODUCTION, Hit, Ranking
from brain.search.rerank import BgeReranker, IdentityReranker, Reranker
from brain.sim.generate import storyline_events
from brain.sim.report import format_simulation, sim_json
from brain.sim.run import SimError, simulate
from brain.tenancy.config import BusinessPaths, ConfigError, business_paths, load_business
from brain.tenancy.database import connect, create_database

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
business_app = typer.Typer(no_args_is_help=True, help="Create businesses.")
app.add_typer(business_app, name="business")

BusinessesDir = Annotated[
    Path, typer.Option("--businesses-dir", envvar="BRAIN_BUSINESSES_DIR", help="Configs.")
]
DataDir = Annotated[
    Path, typer.Option("--data-dir", envvar="BRAIN_DATA_DIR", help="Runtime state.")
]
KeysFile = Annotated[
    Path | None,
    typer.Option(
        "--keys-file",
        envvar="BRAIN_PROVIDER_KEYS_FILE",
        help="Provider keys for the agentgate proxy (live model only). The brain never reads it.",
    ),
]
# One live question: one model call, a small ceiling on what it may spend.
ASK_MAX_CALLS = 2
ASK_MAX_USD = Decimal("0.50")
DEMO_IDENTITY_NOTE = "note: --as is a demo identity; this unit has no login"


class Models(StrEnum):
    BGE = "bge"
    FAKE = "fake"


class Via(StrEnum):
    DIRECT = "direct"
    MCP = "mcp"


class Suite(StrEnum):
    PERMISSIONS = "permissions"
    GOLDEN = "golden"
    ABLATION = "ablation"
    RETRIEVAL = "retrieval"
    AGREEMENT = "agreement"


def _fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


@business_app.command("create")
def business_create(
    business_id: str,
    businesses_dir: BusinessesDir = Path("businesses"),
    data_dir: DataDir = Path("data"),
) -> None:
    """Create the business database (if missing), migrate it and set up its agentgate world."""
    paths = business_paths(business_id, businesses_dir, data_dir)
    try:
        business = load_business(paths)
        directory = load_directory(business, paths.people_file)
        created = create_database(business.id)
        with open_business_world(prepare_world(business, directory, paths)):
            pass
    except ConfigError as exc:
        _fail(str(exc))
    typer.echo(
        f"{business.id}: database {'created' if created else 'up to date'},"
        f" {len(directory)} people, agentgate world in {paths.data}"
    )


@app.command("index")
def index(
    business_id: str,
    models: Annotated[
        Models, typer.Option("--models", help="bge: BGE-M3 + reranker; fake: test doubles.")
    ] = Models.BGE,
    businesses_dir: BusinessesDir = Path("businesses"),
) -> None:
    """Reset the business to week 0: rebuild the index from its item fixtures and the people
    from people.yaml, and empty the change feed."""
    paths = business_paths(business_id, businesses_dir, Path("data"))
    try:
        items = load_items(paths.items_dir)
        people = load_people(paths.people_file)
        check_versions(items, paths)
    except ConfigError as exc:
        _fail(str(exc))
    embedder: Embedder = BgeM3Embedder() if models is Models.BGE else HashEmbedder()
    reranker: Reranker = BgeReranker() if models is Models.BGE else IdentityReranker()
    with connect(business_id) as conn:
        summary = rebuild_index(conn, items, people, embedder, reranker)
    typer.echo(
        f"{business_id}: {summary.items} items, {summary.chunks} chunks,"
        f" {summary.postings} postings ({summary.embedder}, {summary.reranker})"
    )


def check_versions(items: Sequence[Item], paths: BusinessPaths) -> None:
    """Refuse items with a likely version pair that nothing links (ingest.versions)."""
    not_versions = load_not_versions(paths.not_versions_file, {i.id for i in items})
    problems = version_problems(items, not_versions)
    if problems:
        raise ConfigError("unlinked versions:\n  " + "\n  ".join(problems))


@app.command("ask")
def ask(
    business_id: str,
    question: str,
    as_email: Annotated[str, typer.Option("--as", help="Email of the person asking.")],
    model: Annotated[ModelMode, typer.Option("--model", help="live or mock.")] = ModelMode.LIVE,
    keys_file: KeysFile = None,
    businesses_dir: BusinessesDir = Path("businesses"),
    data_dir: DataDir = Path("data"),
) -> None:
    """Answer one question as one person, with citations."""
    typer.echo(DEMO_IDENTITY_NOTE, err=True)
    try:
        with open_business(business_id, businesses_dir, data_dir) as opened:
            person = opened.person(as_email)
            with open_run(
                opened, person, model, keys_file, max_usd=ASK_MAX_USD, max_calls=ASK_MAX_CALLS
            ) as brain:
                asked = brain.ask(person, question)
            typer.echo(_render(asked.answer, asked.hits))
            typer.echo(
                f"\nrun {asked.run_id}; trace: uv run panoptes session {asked.run_id}"
                f" --db {opened.world_paths.audit}",
                err=True,
            )
    except (ConfigError, GateError, InvariantError, ModelError, LookupError) as exc:
        _fail(str(exc))


@app.command("mcp")
def mcp(
    business_id: str,
    as_email: Annotated[
        str, typer.Option("--as", help="Email of the person every tool call answers as.")
    ],
    model: Annotated[ModelMode, typer.Option("--model", help="live or mock.")] = ModelMode.LIVE,
    keys_file: KeysFile = None,
    businesses_dir: BusinessesDir = Path("businesses"),
    data_dir: DataDir = Path("data"),
) -> None:
    """Serve the brain as an MCP server over stdio, as one person (a demo identity).

    Tools: search, read_item, deals and ask. Each call is one agentgate run on a long-running
    model proxy, ended when the call returns. stdout carries the MCP protocol; notes and logs
    go to stderr."""
    typer.echo(DEMO_IDENTITY_NOTE, err=True)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        with open_business(business_id, businesses_dir, data_dir) as opened:
            person = opened.person(as_email)
            with serve_runs(
                opened, model, keys_file, max_usd=ASK_MAX_USD, max_calls=ASK_MAX_CALLS
            ) as server:
                caller = DemoCaller(person.id)
                build_server(Tools(opened, server), lambda: caller, demo_identity=True).run("stdio")
    except (ConfigError, GateError, LookupError) as exc:
        _fail(str(exc))


def _render(answer: Answer, hits: Sequence[Hit]) -> str:
    if answer.not_found:
        return answer.text()
    by_chunk = {h.chunk_id: h for h in hits}
    numbers: dict[str, int] = {}
    lines = []
    for sentence in answer.sentences:
        marks = []
        for chunk_id in sentence.chunk_ids:
            numbers.setdefault(chunk_id, len(numbers) + 1)
            marks.append(f"[{numbers[chunk_id]}]")
        lines.append(f"{sentence.text} {''.join(marks)}")
    lines.extend(d.note(answer.language) for d in answer.disagreements)
    lines.append("")
    for chunk_id, n in numbers.items():
        hit = by_chunk[chunk_id]
        lines.append(
            f"[{n}] {hit.item_title} ({hit.source}, {hit.item_location}, {hit.location},"
            f" {hit.updated.isoformat()}) {chunk_id}"
        )
    return "\n".join(lines)


@app.command("eval")
def evaluate(
    business_id: str,
    suite: Annotated[Suite, typer.Option("--suite")] = Suite.PERMISSIONS,
    model: Annotated[
        ModelMode | None,
        typer.Option("--model", help="Default: mock for permissions, live for golden/ablation."),
    ] = None,
    keys_file: KeysFile = None,
    repeats: Annotated[
        int | None, typer.Option("--repeats", min=1, help="Default: 1 golden, 3 ablation.")
    ] = None,
    report: Annotated[
        str | None,
        typer.Option("--report", help="Write reports/<date>-<label>.{json,md} (clean tree)."),
    ] = None,
    gate: Annotated[
        bool, typer.Option("--gate", help="retrieval: fail below the thresholds.")
    ] = False,
    readme: Annotated[
        bool, typer.Option("--readme", help="With --report: write the README evals table.")
    ] = False,
    stats: Annotated[bool, typer.Option("--stats", help="golden: print the set's shape.")] = False,
    sample: Annotated[
        bool, typer.Option("--sample", help="agreement: print the blind labelling sheet.")
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            "--check-report",
            help="Fail unless the newest live report matches the answer fingerprint and"
            " meets the answer thresholds (no model call).",
        ),
    ] = False,
    businesses_dir: BusinessesDir = Path("businesses"),
    reports_dir: Annotated[Path, typer.Option("--reports-dir")] = Path("reports"),
) -> None:
    """Run an eval suite. Runs in a fresh temporary agentgate world and audit log.

    permissions: the person x canary suite. golden: the golden set with production ranking.
    ablation: the golden set over keyword, vector, hybrid and hybrid_rerank. retrieval:
    search only, no model (the CI gate). agreement: grader vs hand labels.
    """
    paths = business_paths(business_id, businesses_dir, Path("data"))
    try:
        if check:
            _check_report(paths, reports_dir)
            return
        if readme and not report:
            raise ConfigError("--readme needs --report: the README table names its report")
        commit = require_clean_tree() if report else None
        items = load_items(paths.items_dir)
        business = load_business(paths)
        directory = load_directory(business, paths.people_file)
        people = {p.id for p in directory}
        oracle = load_suite(paths.evals_dir / "permissions.yaml", items, people)
        golden = load_golden(
            paths.evals_dir / "golden.yaml",
            {i.id for i in items},
            expected_visibility(oracle, items, people),
        )
        if stats:
            typer.echo(golden_stats(golden))
            return
        if suite is Suite.AGREEMENT:
            _agreement(business_id, businesses_dir, paths, golden, sample=sample)
            return
        live_default = suite in (Suite.GOLDEN, Suite.ABLATION)
        mode = model or (ModelMode.LIVE if live_default else ModelMode.MOCK)
        with (
            tempfile.TemporaryDirectory(prefix="brain-eval-") as scratch,
            open_business(business_id, businesses_dir, Path(scratch)) as opened,
        ):
            require_week_zero(opened.conn)
            if suite is Suite.PERMISSIONS:
                typer.echo(f"permissions suite, model {mode.value}, {len(people)} people")
                result = run_suite(opened, items, oracle, mode, keys_file=keys_file)
                typer.echo(format_report(result))
                if not result.passed:
                    raise typer.Exit(code=1)
                return
            rankings = [PRODUCTION] if suite is Suite.GOLDEN else list(Ranking)
            n = len(golden.questions)
            typer.echo(f"{suite.value}: {n} questions, rankings {', '.join(rankings)}")
            retrieval = retrieval_rows(opened, golden, rankings)
            answers = []
            reps = 0
            if suite is not Suite.RETRIEVAL:
                reps = repeats or (1 if suite is Suite.GOLDEN else 3)
                typer.echo(f"model {mode.value}, {reps} repeat(s)")
                answers = answer_rows(
                    opened, golden, rankings, reps, mode, keys_file, progress=typer.echo
                )
            summaries = summarize(golden, retrieval, answers, rankings)
            typer.echo(format_summary(summaries))
            if report is not None and commit is not None:
                embedder, reranker = index_models(opened.conn)
                built = build_report(
                    label=report,
                    commit=commit,
                    command=_command_line(),
                    business=opened.business,
                    golden_file=paths.evals_dir / "golden.yaml",
                    model_mode=mode.value if answers else "none",
                    repeats=reps,
                    question_count=n,
                    summaries=summaries,
                    retrieval=retrieval,
                    answers=answers,
                    embedder=embedder,
                    reranker=reranker,
                )
                json_path, md_path = write_report(reports_dir, built)
                typer.echo(f"wrote {json_path} and {md_path}")
                if readme:
                    write_evals_readme(Path("README.md"), readme_block(built, json_path))
                    typer.echo("wrote the README evals table")
            if gate:
                problems = check_retrieval(summaries, load_thresholds(paths.thresholds_file))
                for problem in problems:
                    typer.echo(f"GATE: {problem}")
                typer.echo("retrieval gate: " + ("FAIL" if problems else "PASS"))
                if problems:
                    raise typer.Exit(code=1)
    except (ConfigError, GateError, InvariantError, ModelError, LookupError) as exc:
        _fail(str(exc))


def _check_report(paths: BusinessPaths, reports_dir: Path) -> None:
    business = load_business(paths)
    newest = newest_answer_report(reports_dir)
    problems = check_report(
        newest,
        load_thresholds(paths.thresholds_file),
        answer_fingerprint(business, paths.evals_dir / "golden.yaml"),
    )
    meta = newest["meta"]
    typer.echo(f"newest live report: {meta['label']} ({meta['date']}, commit {meta['commit']})")
    for problem in problems:
        typer.echo(f"REPORT: {problem}")
    typer.echo("report check: " + ("FAIL" if problems else "PASS"))
    if problems:
        raise typer.Exit(code=1)


def _agreement(
    business_id: str, businesses_dir: Path, paths: BusinessPaths, golden: GoldenSet, *, sample: bool
) -> None:
    labels_path = paths.evals_dir / "agreement.yaml"
    if sample:
        source = typer.prompt("report to sample (reports/<file>.json)")
        loaded = read_report(Path(source))
        with (
            tempfile.TemporaryDirectory(prefix="brain-eval-") as scratch,
            open_business(business_id, businesses_dir, Path(scratch)) as opened,
        ):
            sources = chunk_sources(opened.conn)
        sheet = labelling_sheet(loaded, golden, sources)
        typer.echo(yaml.safe_dump(sheet, allow_unicode=True, sort_keys=False, width=100))
        return
    labels = load_labels(labels_path)
    typer.echo(format_agreement(score(labels, read_report(Path(labels.source)))))


def _command_line() -> str:
    """The command as typed, with the keys file path left out of the record."""
    args, hidden = [], False
    for arg in sys.argv[1:]:
        if hidden:
            args.append("<keys-file>")
            hidden = False
        elif arg == "--keys-file":
            args.append(arg)
            hidden = True
        elif arg.startswith("--keys-file="):
            args.append("--keys-file=<keys-file>")
        else:
            args.append(arg)
    return " ".join(["brain", *args])


def golden_stats(golden: GoldenSet) -> str:
    qs = golden.questions
    lines = Counter(q.line for q in qs)
    languages = Counter(q.language for q in qs)
    return "\n".join(
        [
            f"questions: {len(qs)}",
            "per line: " + ", ".join(f"{k} {v}" for k, v in sorted(lines.items())),
            "languages: " + ", ".join(f"{k} {v}" for k, v in sorted(languages.items())),
            f"cross-language: {sum(q.cross_language for q in qs)}",
            f"no answer: {sum(q.no_answer for q in qs)}",
            f"two or more sources: {sum(q.two_source for q in qs)}",
            "every asker may see every expected item: yes (checked against the oracle)",
        ]
    )


@app.command("redteam")
def redteam(
    business_id: str,
    model: Annotated[
        ModelMode, typer.Option("--model", help="echo, obey (worst cases), mock or live.")
    ] = ModelMode.ECHO,
    keys_file: KeysFile = None,
    markdown: Annotated[
        bool, typer.Option("--markdown", help="Write the results table into README.md.")
    ] = False,
    write_baseline: Annotated[
        bool, typer.Option("--write-baseline", help="Rewrite the baseline from this run.")
    ] = False,
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Run only these scenario ids.")
    ] = None,
    via: Annotated[
        Via, typer.Option("--via", help="direct (in-process) or mcp (as an MCP client).")
    ] = Via.DIRECT,
    businesses_dir: BusinessesDir = Path("businesses"),
) -> None:
    """Run the leak suite: every scenario plus the person x canary grid. Exit 1 on any
    failure, or on any scenario missing from (or not yet in) the baseline. With --via mcp,
    the scenarios one MCP caller can pose, and the MCP grid, through the MCP server."""
    if only and (markdown or write_baseline):
        _fail("--markdown and --write-baseline need the whole suite, not --only")
    if via is Via.MCP and (markdown or write_baseline):
        _fail("--markdown and --write-baseline are for the in-process suite")
    paths = business_paths(business_id, businesses_dir, Path("data"))
    redteam_dir = paths.config / "redteam"
    baseline = redteam_dir / "baseline.json"
    expected_ids: set[str] = set()
    try:
        items = load_items(paths.items_dir)
        with (
            tempfile.TemporaryDirectory(prefix="brain-redteam-") as scratch,
            open_business(business_id, businesses_dir, Path(scratch)) as opened,
            ExitStack() as stack,
        ):
            require_week_zero(opened.conn)
            people = {p.id for p in opened.directory}
            oracle = load_suite(paths.evals_dir / "permissions.yaml", items, people)
            scenarios = load_scenarios(redteam_dir, people)
            if only:
                scenarios = [s for s in scenarios if s.id in set(only)]
            story = storyline_events(
                opened.business, items, load_people(paths.people_file), paths.sim_dir
            )
            runner = Runner(
                opened, items, oracle, load_decoys(redteam_dir), model, keys_file, stack, story
            )
            # The README stamp names the real model and the index the run used.
            label = f"`{model.value}`"
            if model is ModelMode.LIVE:
                label = f"`{opened.business.model.name}` (live, through the agentgate proxy)"
            index = " + ".join(f"`{name}`" for name in index_models(opened.conn))
            if via is Via.MCP:
                scenarios = [s for s in scenarios if reachable(s)]
                typer.echo(
                    f"leak suite over MCP, model {model.value}, {len(scenarios)} scenarios + grid"
                )
                server = stack.enter_context(serve_runs(opened, model, keys_file))
                suite = McpSuite(opened, items, oracle, load_decoys(redteam_dir), model, server)
                results = anyio.run(lambda: suite.run(scenarios, grid=not only))
                expected_ids = {s.id for s in scenarios}
            else:
                typer.echo(f"leak suite, model {model.value}, {len(scenarios)} scenarios + grid")
                results = [runner.run(scenario) for scenario in scenarios]
                if not only:
                    results += runner.grid()
    except (ConfigError, GateError, InvariantError, ModelError, LookupError) as exc:
        _fail(str(exc))
    typer.echo(format_table(results))
    if via is Via.MCP:
        ran = {r.id for r in results}
        problems = [f"missing from the run: {sid}" for sid in sorted(expected_ids - ran)]
    else:
        problems = [] if only else baseline_problems(baseline, results)
    if write_baseline:
        write_baseline_file(baseline, results)
        problems = []
        typer.echo(f"baseline written: {len(results)} scenarios")
    for problem in problems:
        typer.echo(f"BASELINE {problem}")
    if markdown:
        command = f"uv run brain redteam {business_id} --model {model.value}"
        write_readme(Path("README.md"), markdown_block(results, label, index, command))
        typer.echo("README.md results table updated")
    if problems or not all(r.passed for r in results):
        raise typer.Exit(code=1)


@app.command("sync")
def sync(
    business_id: str,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", min=0.0, help="Seconds between feed polls.")
    ] = 1.0,
    businesses_dir: BusinessesDir = Path("businesses"),
    data_dir: DataDir = Path("data"),
) -> None:
    """Run the sync worker: apply the change feed in order until stopped (Ctrl-C, SIGTERM).
    Prints `ready` once it holds the business's worker lock."""
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    try:
        with (
            open_business(business_id, businesses_dir, data_dir) as opened,
            connect(business_id, autocommit=True) as conn,
        ):
            applier = Applier(opened.business, opened.world, opened.searcher.embedder)
            applied = run_worker(
                conn,
                applier,
                poll_interval,
                should_stop=stop.is_set,
                sleep=stop.wait,
                on_error=lambda exc: typer.echo(f"sync: {type(exc).__name__}: {exc}", err=True),
                on_ready=lambda: typer.echo("ready", nl=True),
            )
    except (ConfigError, GateError, LookupError, WorkerBusy) as exc:
        _fail(str(exc))
    typer.echo(f"sync stopped after {applied} events", err=True)


@app.command("simulate")
def simulate_command(
    business_id: str,
    weeks: Annotated[int, typer.Option("--weeks", min=0)] = 8,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    models: Annotated[
        Models, typer.Option("--models", help="bge: BGE-M3 + reranker; fake: test doubles.")
    ] = Models.BGE,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", min=0.0, help="The worker's seconds per poll.")
    ] = 1.0,
    forget: Annotated[
        bool, typer.Option("--forget", help="Measure time to forget for every revocation.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    businesses_dir: BusinessesDir = Path("businesses"),
    data_dir: DataDir = Path("data"),
) -> None:
    """Reset the business to week 0, then replay N simulated weeks through the change feed
    and a separate sync worker. Leaves the database at week N; `brain index` returns it to
    week 0. Exit 1 if a revocation is not forgotten within the bound."""
    embedder: Embedder = BgeM3Embedder() if models is Models.BGE else HashEmbedder()
    reranker: Reranker = BgeReranker() if models is Models.BGE else IdentityReranker()
    try:
        result = simulate(
            business_id,
            businesses_dir,
            data_dir,
            weeks=weeks,
            seed=seed,
            embedder=embedder,
            reranker=reranker,
            poll_interval=poll_interval,
            forget=forget,
            log=(lambda line: None) if as_json else typer.echo,
        )
    except (ConfigError, GateError, LookupError, SimError) as exc:
        _fail(str(exc))
    if as_json:
        typer.echo(json.dumps(sim_json(result), indent=2, ensure_ascii=False))
    else:
        typer.echo(format_simulation(result))
    if result.unforgotten:
        raise typer.Exit(code=1)


def main() -> None:
    app()
