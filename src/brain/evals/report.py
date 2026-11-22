"""Eval reports, thresholds and the answer fingerprint.

A report (`reports/<date>-<label>.json` and `.md`) is written only from a clean commit and
records the command, models, revisions, sampling, n, repeats and the answer fingerprint.

The answer fingerprint hashes everything that shapes an answer, its grade or its score
without being retrieval: the system prompt and schema, the grader's rubrics, the source of
the answer, pipeline, grader, language and aggregation modules, the business's model block
and the golden set itself. CI cannot call a live model, so
`check_report` stands in: the newest live report must carry the current fingerprint and
meet the answer thresholds. A prompt or model change therefore lands with a fresh report.
Retrieval is gated in CI directly (`check_retrieval`), with the real models.
"""

import hashlib
import json
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import brain.answer.compose as compose
import brain.answer.models as models
import brain.answer.pipeline as pipeline
import brain.evals.ablation as ablation
import brain.evals.grader as grader
import brain.evals.language as language
from brain.evals.ablation import AnswerRow, RankingSummary, RetrievalRow
from brain.ingest.embedding import BgeM3Embedder
from brain.search.query import Ranking
from brain.search.rerank import BgeReranker
from brain.tenancy.config import Business, ConfigError, read_yaml, validate

GATED = Ranking.HYBRID_RERANK
README_START = "<!-- evals:start -->"
README_END = "<!-- evals:end -->"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievalThresholds(_Strict):
    recall_at_8: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)


class AnswerThresholds(_Strict):
    citation_precision: float = Field(ge=0, le=1)
    correctness: float = Field(ge=0, le=1)
    correct_not_found: float = Field(ge=0, le=1)
    language_match: float = Field(ge=0, le=1)


class Thresholds(_Strict):
    # The report the thresholds were set from (reports/<name>.json).
    source: str
    retrieval: RetrievalThresholds
    answer: AnswerThresholds


def load_thresholds(path: Path) -> Thresholds:
    if not path.is_file():
        raise ConfigError(f"{path}: no thresholds yet; set them from the first recorded run")
    return validate(Thresholds, read_yaml(path), path)


def answer_fingerprint(business: Business, golden_file: Path) -> str:
    digest = hashlib.sha256()
    parts = [
        compose.SYSTEM_PROMPT,
        json.dumps(compose.SCHEMA, sort_keys=True),
        grader.CORRECTNESS_SYSTEM,
        grader.CITATIONS_SYSTEM,
        *(_source(module) for module in (compose, models, pipeline, grader, language, ablation)),
        business.model.model_dump_json(),
        golden_file.read_text(encoding="utf-8"),
    ]
    for part in parts:
        digest.update(hashlib.sha256(part.encode()).digest())
    return "sha256:" + digest.hexdigest()


# Gates


def check_retrieval(summaries: Sequence[RankingSummary], thresholds: Thresholds) -> list[str]:
    return _below(summaries, thresholds.retrieval.model_dump())


def check_report(report: dict[str, Any], thresholds: Thresholds, fingerprint: str) -> list[str]:
    problems = []
    meta = report["meta"]
    if meta.get("model_mode") != "live":
        problems.append(f"the newest report ({meta['label']}) is not a live run")
    if meta.get("answer_fingerprint") != fingerprint:
        problems.append(
            f"answer fingerprint changed since {meta['label']} ({meta['commit']}): prompts, "
            "grader or model config changed without a new recorded run"
        )
    gated = report["rankings"].get(GATED.value, {})
    for name, floor in thresholds.answer.model_dump().items():
        found = gated.get(name)
        if found is None:
            problems.append(f"{GATED.value} {name}: missing from the report")
        elif found["mean"] < floor:
            problems.append(f"{GATED.value} {name} {found['mean']:.3f} < threshold {floor:.3f}")
    return problems


def newest_answer_report(reports_dir: Path) -> dict[str, Any]:
    found = []
    for path in sorted(reports_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["meta"].get("answers") and report["meta"].get("model_mode") == "live":
            found.append(report)
    if not found:
        raise ConfigError(f"{reports_dir}: no live report with answer metrics")
    return max(found, key=lambda r: (r["meta"]["date"], r["meta"]["written"]))


def _below(summaries: Sequence[RankingSummary], floors: dict[str, float]) -> list[str]:
    gated = next((s for s in summaries if s.ranking is GATED), None)
    if gated is None:
        return [f"{GATED.value} was not run"]
    problems = []
    for name, floor in floors.items():
        found = gated.metrics.get(name)
        if found is None:
            problems.append(f"{GATED.value} {name}: not measured")
        elif found.mean < floor:
            problems.append(f"{GATED.value} {name} {found.mean:.3f} < threshold {floor:.3f}")
    return problems


# Writing


def require_clean_tree() -> str:
    """The short commit, refusing a dirty tree: a report names exactly what it measured."""
    status = _git("status", "--porcelain")
    if status is None:
        raise ConfigError("a report needs a git checkout")
    if status.strip():
        raise ConfigError("a report needs a clean tree; commit first")
    return (_git("rev-parse", "--short=12", "HEAD") or "unknown").strip()


def build_report(
    *,
    label: str,
    commit: str,
    command: str,
    business: Business,
    golden_file: Path,
    model_mode: str,
    repeats: int,
    question_count: int,
    summaries: Sequence[RankingSummary],
    retrieval: Sequence[RetrievalRow],
    answers: Sequence[AnswerRow],
    embedder: str,
    reranker: str,
) -> dict[str, Any]:
    return {
        "meta": {
            "label": label,
            "date": date.today().isoformat(),
            "written": _now(),
            "commit": commit,
            "command": command,
            "business": business.id,
            "model_mode": model_mode,
            "answer_model": business.model.model_dump(),
            "grader_model": business.model.name,
            "index": {"embedder": embedder, "reranker": reranker},
            "pinned_revisions": {
                BgeM3Embedder.model_id: BgeM3Embedder.revision,
                BgeReranker.model_id: BgeReranker.revision,
            },
            "questions": question_count,
            "repeats": repeats,
            "answers": len(answers),
            "answer_fingerprint": answer_fingerprint(business, golden_file),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "notes": [
                "Synthetic data: Casa Prado and everything in it is fictional.",
                "The grader is the answer model itself (self-preference risk).",
                "Intervals: 95% percentile bootstrap over questions, fixed seed.",
            ],
        },
        "rankings": {
            s.ranking.value: {
                **{k: (None if v is None else asdict(v)) for k, v in s.metrics.items()},
                "counts": s.counts,
            }
            for s in summaries
        },
        "rows": {
            "retrieval": [_row(r) for r in retrieval],
            "answers": [_row(a) for a in answers],
        },
    }


def write_report(reports_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report['meta']['date']}-{report['meta']['label']}"
    json_path, md_path = reports_dir / f"{stem}.json", reports_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")
    md_path.write_text(markdown(report) + "\n", "utf-8")
    return json_path, md_path


def markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    model = meta["answer_model"]
    lines = [
        f"# Retrieval evals: {meta['label']}",
        "",
        f"From `{meta['command']}` at commit `{meta['commit']}` on {meta['date']}.",
        f"Answer model and grader `{model['name']}` (temperature {model['temperature']},"
        f" reasoning effort {model['reasoning_effort']}), mode `{meta['model_mode']}`."
        f" Index `{meta['index']['embedder']}` + `{meta['index']['reranker']}`.",
        f"{meta['questions']} questions, {meta['repeats']} repeats per question and ranking,"
        f" {meta['answers']} answers. Synthetic data (Casa Prado is fictional).",
        "",
        summary_table(report),
        "",
        "Means with 95% bootstrap intervals over questions; n is the number of questions"
        " scored. The grader is the answer model itself, so grader-based numbers carry a"
        " self-preference risk.",
    ]
    return "\n".join(lines)


def readme_block(report: dict[str, Any], report_path: Path) -> str:
    meta = report["meta"]
    model = meta["answer_model"]
    stamp = (
        f"From [`{report_path.as_posix()}`]({report_path.as_posix()}): `{meta['command']}` at"
        f" commit `{meta['commit']}` on {meta['date']}, answer model and grader"
        f" `{model['name']}`, {meta['questions']} questions x {meta['repeats']} repeats."
        " Means with 95% bootstrap intervals over questions. Synthetic data (Casa Prado is"
        " fictional)."
    )
    return "\n".join([README_START, summary_table(report), "", stamp, README_END])


def write_readme(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1:
        raise ConfigError(f"{readme}: missing the {README_START} / {README_END} markers")
    readme.write_text(text[:start] + block + text[end + len(README_END) :], encoding="utf-8")


def summary_table(report: dict[str, Any]) -> str:
    names = [
        ("recall_at_8", "Recall@8"),
        ("mrr", "MRR"),
        ("citation_precision", "Citation precision"),
        ("correctness", "Correct"),
        ("correct_not_found", "Correct not-found"),
        ("false_not_found", "False not-found"),
        ("language_match", "Language match"),
    ]
    rankings = report["rankings"]
    present = [(k, title) for k, title in names if any(rankings[r].get(k) for r in rankings)]
    header = ["Ranking", *(title for _, title in present), "USD/answer", "p50 / p95 ms"]
    rows = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for ranking, metrics in rankings.items():
        cells = [f"`{ranking}`"]
        for key, _ in present:
            value = metrics.get(key)
            cells.append(
                "-"
                if value is None
                else f"{value['mean']:.2f} [{value['low']:.2f}, {value['high']:.2f}] n={value['n']}"
            )
        counts = metrics["counts"]
        cells.append(str(counts.get("answer_usd_per_answer", "-")))
        if "ask_p50_ms" in counts:
            cells.append(f"{counts['ask_p50_ms']} / {counts['ask_p95_ms']}")
        else:
            cells.append(
                f"{counts.get('search_p50_ms', '-')} / {counts.get('search_p95_ms', '-')} (search)"
            )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def format_summary(summaries: Sequence[RankingSummary]) -> str:
    lines = []
    for s in summaries:
        lines.append(f"{s.ranking.value}")
        for name, value in s.metrics.items():
            lines.append(f"  {name:<20} {'-' if value is None else value.text()}")
        for name, count in s.counts.items():
            lines.append(f"  {name:<20} {count}")
    return "\n".join(lines)


def _row(row: Any) -> dict[str, Any]:
    out = asdict(row)
    for key, value in out.items():
        if isinstance(value, Decimal):
            out[key] = str(value)
        elif isinstance(value, Ranking):
            out[key] = value.value
    return out


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _source(module: ModuleType) -> str:
    return Path(str(module.__file__)).read_text(encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
