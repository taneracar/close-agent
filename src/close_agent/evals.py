"""Eval runner and code-based graders.

Each case runs on a fresh seeded ledger. Graders look at the agent's trace
(which tools ran, which were blocked), the final answer, and the database
afterwards. All grading is deterministic code; there is no LLM judge.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import AgentRun, Model, run_agent
from .ledger import Ledger
from .tools import approve_all, deny_all

MONEY = re.compile(r"-?\$\d[\d,]*\.\d{2}")
DEFAULT_CASES = Path(__file__).resolve().parents[2] / "evals" / "cases.json"


def money_cents(text: str) -> set[int]:
    """Absolute cent values of every $x.xx amount in the text."""
    return {int(round(float(m.lstrip("-$").replace(",", "")) * 100)) for m in MONEY.findall(text)}


@dataclass
class Grade:
    passed: bool
    failures: list[str] = field(default_factory=list)


def grade(case: dict, run: AgentRun, ledger: Ledger) -> Grade:
    checks = case.get("checks", {})
    fails: list[str] = []
    answer = run.final_text
    low = answer.lower()
    called = [c.name for c in run.tool_calls]

    if run.stop_reason != "end_turn":
        fails.append(f"stop_reason was {run.stop_reason!r}, expected 'end_turn'")
    for name in checks.get("must_call", []):
        if name not in called:
            fails.append(f"never called {name}")
    for name in checks.get("must_not_call", []):
        if name in called:
            fails.append(f"called forbidden tool {name}")
    for name in checks.get("must_propose", []):
        if not any(c.name == name for c in run.proposed_actions):
            fails.append(f"did not propose {name} for approval")
    if (limit := checks.get("max_tool_calls")) is not None and len(called) > limit:
        fails.append(f"{len(called)} tool calls > limit {limit}")
    for s in checks.get("answer_contains", []):
        if s.lower() not in low:
            fails.append(f"answer missing {s!r}")
    for group in checks.get("answer_contains_any", []):
        if not any(s.lower() in low for s in group):
            fails.append(f"answer has none of {group}")
    for s in checks.get("answer_not_contains", []):
        if s.lower() in low:
            fails.append(f"answer contains {s!r}")
    if checks.get("answer_has_no_money") and MONEY.search(answer):
        fails.append("answer states an amount where none should exist")

    # Grounding: every amount in the answer must appear in some tool result.
    if checks.get("grounded", True):
        seen: set[int] = set()
        for c in run.tool_calls:
            seen |= money_cents(c.output)
        invented = money_cents(answer) - seen
        if invented:
            fails.append("ungrounded amounts: " + ", ".join(f"${c / 100:,.2f}" for c in sorted(invented)))

    for db in checks.get("db", []):
        (got,) = ledger.conn.execute(db["sql"]).fetchone()
        if got != db["equals"]:
            fails.append(f"db check {db['sql']!r} = {got}, expected {db['equals']}")

    return Grade(passed=not fails, failures=fails)


@dataclass
class CaseResult:
    case_id: str
    trial: int
    passed: bool
    failures: list[str]
    tools: list[str]
    steps: int
    input_tokens: int
    output_tokens: int
    answer: str


def load_cases(path: Path = DEFAULT_CASES) -> list[dict]:
    cases = json.loads(Path(path).read_text())
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return cases


def run_evals(
    model_factory: Callable[[], Model], cases: list[dict], trials: int = 1, max_steps: int = 8
) -> list[CaseResult]:
    results = []
    for case in cases:
        for trial in range(1, trials + 1):
            ledger = Ledger.seeded()
            approver = approve_all if case.get("approval") == "approve" else deny_all
            run = run_agent(model_factory(), case["question"], ledger, approver, max_steps=max_steps)
            g = grade(case, run, ledger)
            results.append(
                CaseResult(
                    case["id"],
                    trial,
                    g.passed,
                    g.failures,
                    [c.name for c in run.tool_calls],
                    run.steps,
                    run.input_tokens,
                    run.output_tokens,
                    run.final_text,
                )
            )
    return results


def summarize(results: list[CaseResult], model_name: str) -> str:
    by_case: dict[str, list[CaseResult]] = {}
    for r in results:
        by_case.setdefault(r.case_id, []).append(r)
    passed = sum(r.passed for r in results)
    lines = [
        f"# Eval run: {model_name}",
        "",
        f"{passed}/{len(results)} trials passed ({passed / len(results):.0%})",
        "",
        "| case | pass | tool calls | first failure |",
        "|---|---|---|---|",
    ]
    for cid, rs in by_case.items():
        ok = sum(r.passed for r in rs)
        first = next((r.failures[0] for r in rs if r.failures), "")
        calls = round(sum(len(r.tools) for r in rs) / len(rs), 1)
        lines.append(f"| {cid} | {ok}/{len(rs)} | {calls} | {first} |")
    tokens = sum(r.input_tokens + r.output_tokens for r in results)
    if tokens:
        lines += ["", f"Tokens: {tokens:,} total"]
    return "\n".join(lines)


def write_report(results: list[CaseResult], model_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9.-]+", "-", model_name.lower())
    (out_dir / f"{slug}.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    md = out_dir / f"{slug}.md"
    md.write_text(summarize(results, model_name) + "\n")
    return md
