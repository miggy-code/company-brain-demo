"""`brain`: create a business, index it, ask as a person, run the eval suites.

`--as` is a demo identity, as in agentgate: this unit has no login. Live answers need the
operator's provider keys file (JSON, provider name to key, mode 0600), which only the
agentgate proxy process reads.
"""

import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from brain.answer.compose import Answer
from brain.answer.models import ModelError
from brain.evals.golden import format_golden, load_golden, run_golden
from brain.evals.permissions import format_report, load_suite, run_suite
from brain.gate.policy import GateError, InvariantError, open_business_world, prepare_world
from brain.identity.people import load_directory
from brain.ingest.embedding import BgeM3Embedder, Embedder, HashEmbedder
from brain.ingest.index import rebuild_index
from brain.ingest.items import load_items
from brain.redteam.report import baseline_problems, format_table, write_readme
from brain.redteam.report import markdown as markdown_block
from brain.redteam.report import write_baseline as write_baseline_file
from brain.redteam.runner import Runner
from brain.redteam.scenarios import load_decoys, load_scenarios
from brain.runtime import ModelMode, open_business, open_run
from brain.search.query import Hit
from brain.search.rerank import BgeReranker, IdentityReranker, Reranker
from brain.tenancy.config import ConfigError, business_paths, load_business
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


class Suite(StrEnum):
    PERMISSIONS = "permissions"
    GOLDEN = "golden"


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
    """Rebuild the business index from its item fixtures."""
    paths = business_paths(business_id, businesses_dir, Path("data"))
    try:
        items = load_items(paths.items_dir)
    except ConfigError as exc:
        _fail(str(exc))
    embedder: Embedder = BgeM3Embedder() if models is Models.BGE else HashEmbedder()
    reranker: Reranker = BgeReranker() if models is Models.BGE else IdentityReranker()
    with connect(business_id) as conn:
        summary = rebuild_index(conn, items, embedder, reranker)
    typer.echo(
        f"{business_id}: {summary.items} items, {summary.chunks} chunks,"
        f" {summary.postings} postings ({summary.embedder}, {summary.reranker})"
    )


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
        typer.Option("--model", help="Default: mock for permissions, live for golden."),
    ] = None,
    keys_file: KeysFile = None,
    businesses_dir: BusinessesDir = Path("businesses"),
) -> None:
    """Run an eval suite. Runs in a fresh temporary agentgate world and audit log."""
    mode = model or (ModelMode.MOCK if suite is Suite.PERMISSIONS else ModelMode.LIVE)
    paths = business_paths(business_id, businesses_dir, Path("data"))
    try:
        items = load_items(paths.items_dir)
        with (
            tempfile.TemporaryDirectory(prefix="brain-eval-") as scratch,
            open_business(business_id, businesses_dir, Path(scratch)) as opened,
        ):
            people = {p.id for p in opened.directory}
            typer.echo(f"{suite.value} suite, model {mode.value}, {len(people)} people")
            if suite is Suite.PERMISSIONS:
                loaded = load_suite(paths.evals_dir / "permissions.yaml", items, people)
                result = run_suite(opened, items, loaded, mode)
                typer.echo(format_report(result))
                if not result.passed:
                    raise typer.Exit(code=1)
            else:
                golden = load_golden(paths.evals_dir / "golden.yaml", {i.id for i in items}, people)
                typer.echo(format_golden(run_golden(opened, golden, mode, keys_file)))
    except (ConfigError, GateError, InvariantError, ModelError, LookupError) as exc:
        _fail(str(exc))


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
    businesses_dir: BusinessesDir = Path("businesses"),
) -> None:
    """Run the leak suite: every scenario plus the person x canary grid. Exit 1 on any
    failure, or on any scenario missing from (or not yet in) the baseline."""
    paths = business_paths(business_id, businesses_dir, Path("data"))
    redteam_dir = paths.config / "redteam"
    baseline = redteam_dir / "baseline.json"
    try:
        items = load_items(paths.items_dir)
        with (
            tempfile.TemporaryDirectory(prefix="brain-redteam-") as scratch,
            open_business(business_id, businesses_dir, Path(scratch)) as opened,
            ExitStack() as stack,
        ):
            people = {p.id for p in opened.directory}
            oracle = load_suite(paths.evals_dir / "permissions.yaml", items, people)
            scenarios = load_scenarios(redteam_dir, people)
            if only:
                scenarios = [s for s in scenarios if s.id in set(only)]
            runner = Runner(
                opened, items, oracle, load_decoys(redteam_dir), model, keys_file, stack
            )
            typer.echo(f"leak suite, model {model.value}, {len(scenarios)} scenarios + grid")
            results = [runner.run(scenario) for scenario in scenarios]
            if not only:
                results += runner.grid()
    except (ConfigError, GateError, InvariantError, ModelError, LookupError) as exc:
        _fail(str(exc))
    typer.echo(format_table(results))
    problems = [] if only else baseline_problems(baseline, results)
    if write_baseline:
        write_baseline_file(baseline, results)
        problems = []
        typer.echo(f"baseline written: {len(results)} scenarios")
    for problem in problems:
        typer.echo(f"BASELINE {problem}")
    if markdown:
        command = f"uv run brain redteam {business_id} --model {model.value}"
        write_readme(Path("README.md"), markdown_block(results, model.value, command))
        typer.echo("README.md results table updated")
    if problems or not all(r.passed for r in results):
        raise typer.Exit(code=1)


def main() -> None:
    app()
