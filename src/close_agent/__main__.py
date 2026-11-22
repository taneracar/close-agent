"""CLI.

python -m close_agent eval                       # offline, scripted FAKE model
python -m close_agent eval --model claude-opus-5-5 --trials 3   # real model, needs credentials
python -m close_agent ask "Why doesn't Operating Cash tie out for 2026-10?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import run_agent
from .evals import DEFAULT_CASES, load_cases, run_evals, summarize, write_report
from .ledger import Ledger
from .models import ClaudeModel, ScriptedModel


def make_model(name: str, effort: str):
    if name == "scripted":
        return ScriptedModel
    return lambda: ClaudeModel(model=name, effort=effort)


def ask_human(name: str, args: dict) -> bool:
    print(f"\n[approval needed] {name} {json.dumps(args)}")
    return input("approve? [y/N] ").strip().lower() == "y"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="close_agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="run the eval set")
    e.add_argument("--model", default="scripted", help="'scripted' (offline fake) or a Claude model id")
    e.add_argument("--effort", default="medium")
    e.add_argument("--trials", type=int, default=1)
    e.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    e.add_argument("--only", nargs="*", help="case ids to run")
    e.add_argument("--out", type=Path, default=Path("eval-results"))
    e.add_argument("--min-pass", type=float, default=1.0, help="exit 1 below this pass rate")

    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--model", default="scripted")
    a.add_argument("--effort", default="medium")
    a.add_argument("--trace", action="store_true")

    args = p.parse_args(argv)
    factory = make_model(args.model, args.effort)

    if args.cmd == "eval":
        cases = load_cases(args.cases)
        if args.only:
            cases = [c for c in cases if c["id"] in set(args.only)]
        results = run_evals(factory, cases, trials=args.trials)
        label = args.model + (" (FAKE, not an LLM)" if args.model == "scripted" else "")
        print(summarize(results, label))
        print(f"\nreport: {write_report(results, args.model, args.out)}")
        rate = sum(r.passed for r in results) / len(results)
        return 0 if rate >= args.min_pass else 1

    run = run_agent(factory(), args.question, Ledger.seeded(), approver=ask_human)
    if args.trace:
        for c in run.tool_calls:
            flag = "DENIED" if c.denied else ("ERROR" if c.is_error else "ok")
            print(f"  -> {c.name} {json.dumps(c.input)} [{flag}]")
    print(run.final_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
