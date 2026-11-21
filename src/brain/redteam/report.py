"""The leak suite's results table, its README block, and the baseline check.

The baseline (businesses/<id>/redteam/baseline.json) lists every scenario id the suite must
run. A run fails on any failed scenario, on any baseline id it did not run (a deleted or
renamed scenario), and on any id it ran that the baseline lacks (a new scenario must be
added to the baseline in the same change).
"""

import json
import subprocess
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from brain.redteam.runner import ScenarioResult
from brain.redteam.scenarios import Category

README_START = "<!-- redteam:start -->"
README_END = "<!-- redteam:end -->"


def table(results: Sequence[ScenarioResult]) -> list[tuple[str, int, int]]:
    rows = []
    for category in Category:
        mine = [r for r in results if r.category is category]
        if mine:
            rows.append((category.value, len(mine), sum(r.passed for r in mine)))
    return rows


def format_table(results: Sequence[ScenarioResult]) -> str:
    lines = [f"{'category':<20}{'scenarios':>10}{'passed':>8}", "-" * 38]
    for category, total, passed in table(results):
        lines.append(f"{category:<20}{total:>10}{passed:>8}")
    total = len(results)
    passed = sum(r.passed for r in results)
    lines.append(f"{'total':<20}{total:>10}{passed:>8}")
    for result in results:
        lines.extend(f"  FAIL {result.id}: {failure}" for failure in result.failures)
    return "\n".join(lines)


def markdown(results: Sequence[ScenarioResult], model: str, command: str) -> str:
    rows = ["| Category | Scenarios | Passed |", "| --- | ---: | ---: |"]
    for category, total, passed in table(results):
        rows.append(f"| {category} | {total} | {passed}/{total} |")
    total = len(results)
    passed = sum(r.passed for r in results)
    rows.append(f"| **total** | **{total}** | **{passed}/{total}** |")
    stamp = (
        f"From `{command}` at commit `{_commit()}` on {date.today().isoformat()}, model"
        f" `{model}`. Synthetic data (Casa Prado is fictional)."
    )
    return "\n".join([README_START, *rows, "", stamp, README_END])


def write_readme(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1:
        raise ValueError(f"{readme}: missing the {README_START} / {README_END} markers")
    readme.write_text(text[:start] + block + text[end + len(README_END) :], encoding="utf-8")


def baseline_problems(path: Path, results: Sequence[ScenarioResult]) -> list[str]:
    listed = set(json.loads(path.read_text(encoding="utf-8"))["scenarios"])
    ran = {r.id for r in results}
    problems = [f"missing from the run: {sid}" for sid in sorted(listed - ran)]
    problems += [f"not in the baseline: {sid}" for sid in sorted(ran - listed)]
    return problems


def write_baseline(path: Path, results: Sequence[ScenarioResult]) -> None:
    ids = sorted(r.id for r in results)
    path.write_text(json.dumps({"scenarios": ids}, indent=2) + "\n", encoding="utf-8")


def _commit() -> str:
    try:
        found = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode != 0
    return found.stdout.strip() + ("-dirty" if dirty else "")
