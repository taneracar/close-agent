"""Model adapters.

- ClaudeModel: the real thing, via the official `anthropic` SDK. Needs credentials.
- ScriptedModel: a FAKE. A deterministic, keyword-driven policy that is NOT a
  language model. It exists so the harness, tools and graders can run offline
  in CI. Its eval scores say the plumbing works, nothing about model quality.
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

from .agent import ModelTurn

DEFAULT_MODEL = "claude-opus-5-5"


class ClaudeModel:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", client: Any = None, max_tokens: int = 16000):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.name = model
        self.effort = effort
        self.max_tokens = max_tokens

    def respond(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn:
        resp = self.client.beta.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            output_config={"effort": self.effort},
            # Server-side fallback: if a safety classifier refuses, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        blocks: list[dict] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return ModelTurn(
            blocks=blocks,
            stop_reason=resp.stop_reason,
            raw_content=resp.content,  # sent back unchanged so thinking blocks survive
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        )


# --------------------------------------------------------------------------- fakes

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}


def parse_period(text: str) -> str | None:
    if m := re.search(r"\b(\d{4})-(\d{2})\b", text):
        return f"{m[1]}-{m[2]}"
    if m := re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", text.lower()):
        return f"{m[2]}-{MONTHS[m[1]]:02d}"
    return None


def history(messages: list[dict]) -> list[dict]:
    """Pair every tool_use with its tool_result: [{name, input, output, is_error}]."""
    results = {}
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for r in msg["content"]:
                results[r["tool_use_id"]] = r
    calls = []
    for msg in messages:
        if msg["role"] == "assistant" and isinstance(msg["content"], list):
            for b in msg["content"]:
                if b["type"] == "tool_use" and b["id"] in results:
                    r = results[b["id"]]
                    calls.append(
                        {
                            "name": b["name"],
                            "input": b["input"],
                            "output": r["content"],
                            "is_error": r.get("is_error", False),
                        }
                    )
    return calls


class ScriptedModel:
    """FAKE model: keyword rules -> tool calls -> templated answer. Not an LLM."""

    name = "scripted-fake"

    def __init__(self):
        self._ids = itertools.count(1)

    # helpers -------------------------------------------------------------
    def _use(self, name: str, **inp: Any) -> dict:
        return {"type": "tool_use", "id": f"toolu_fake_{next(self._ids)}", "name": name, "input": inp}

    def _calls(self, *uses: dict) -> ModelTurn:
        return ModelTurn(blocks=list(uses), stop_reason="tool_use")

    def _say(self, text: str) -> ModelTurn:
        return ModelTurn(blocks=[{"type": "text", "text": text}], stop_reason="end_turn")

    def respond(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn:
        question = messages[0]["content"]
        q = question.lower()
        hist = history(messages)
        done = {h["name"]: h for h in hist}

        if "list_accounts" not in done:
            return self._calls(self._use("list_accounts"))
        accounts = json.loads(done["list_accounts"]["output"])
        period = parse_period(question)
        if period is None:
            return self._say("Which period do you mean? Please give it as YYYY-MM, for example 2026-10.")

        # Multi-account question: reconcile every bank-fed account in parallel.
        if "which accounts" in q:
            recs = [h for h in hist if h["name"] == "reconcile_account"]
            if not recs:
                return self._calls(
                    *[
                        self._use("reconcile_account", account_id=a["id"], period=period)
                        for a in accounts
                        if a["bank_feed"]
                    ]
                )
            out = [json.loads(h["output"]) for h in recs if not h["is_error"]]
            bad = [r for r in out if r["status"] != "reconciled"]
            if not bad:
                return self._say(f"All bank accounts reconcile for {period}.")
            lines = [
                f"- {r['account_id']}: GL {r['gl_balance']} vs bank {r['bank_balance']}, "
                f"difference {r['difference_gl_minus_bank']}"
                for r in bad
            ]
            return self._say(f"Unreconciled for {period}:\n" + "\n".join(lines))

        acct = self._find_account(q, accounts)
        if acct is None:
            return self._say("I couldn't find that account in the chart of accounts, so I have no figures for it.")
        aid = acct["id"]

        if "task" in q:
            if "create_close_task" not in done:
                who = re.search(r"\bfor ([A-Z][a-z]+)\b", question)
                return self._calls(
                    self._use(
                        "create_close_task",
                        title=f"Review {acct['name']} for {period}",
                        account_id=aid,
                        assignee=who[1] if who else "Controller",
                    )
                )
            return self._after_write(done["create_close_task"], "task")

        if any(w in q for w in ("adjust", "book", "post ")):
            if "list_unmatched" not in done:
                return self._calls(self._use("list_unmatched", account_id=aid, period=period))
            if "post_adjusting_entry" not in done:
                um = json.loads(done["list_unmatched"]["output"])
                lines = [b for b in um["bank_only"] if b["date"].startswith(period)]
                if not lines:
                    return self._say("There are no bank-only items to book.")
                return self._calls(
                    *[
                        self._use(
                            "post_adjusting_entry",
                            account_id=aid,
                            entry_date=b["date"],
                            amount_cents=_cents(b["amount"]),
                            memo="Book bank-only item",
                        )
                        for b in lines
                    ]
                )
            return self._after_write(done["post_adjusting_entry"], "adjusting entry")

        if any(w in q for w in ("why", "explain", "unmatched")):
            if "reconcile_account" not in done:
                return self._calls(
                    self._use("reconcile_account", account_id=aid, period=period),
                    self._use("list_unmatched", account_id=aid, period=period),
                )
            rec, um = done["reconcile_account"], done["list_unmatched"]
            if rec["is_error"]:
                return self._say(f"I can't reconcile {acct['name']}: {rec['output']}")
            r, u = json.loads(rec["output"]), json.loads(um["output"])
            items = [f"GL only: {x['amount']} {x['memo']} ({x['date']})" for x in u["gl_only"]] + [
                f"Bank only: {x['amount']} ({x['date']})" for x in u["bank_only"]
            ]
            return self._say(
                f"{acct['name']} is off by {r['difference_gl_minus_bank']} "
                f"(GL {r['gl_balance']}, bank {r['bank_balance']}). Items:\n- " + "\n- ".join(items)
            )

        if any(w in q for w in ("reconcile", "tie out")):
            if "reconcile_account" not in done:
                return self._calls(self._use("reconcile_account", account_id=aid, period=period))
            rec = done["reconcile_account"]
            if rec["is_error"]:
                return self._say(f"I can't reconcile {acct['name']}: {rec['output']}")
            r = json.loads(rec["output"])
            return self._say(
                f"{acct['name']} for {period}: GL {r['gl_balance']}, bank {r['bank_balance']}, "
                f"difference {r['difference_gl_minus_bank']} ({r['status']})."
            )

        if "get_gl_balance" not in done:
            return self._calls(self._use("get_gl_balance", account_id=aid, period=period))
        r = json.loads(done["get_gl_balance"]["output"])
        return self._say(f"The GL balance of {acct['name']} at the end of {period} is {r['gl_balance']}.")

    def _find_account(self, q: str, accounts: list[dict]) -> dict | None:
        for a in sorted(accounts, key=lambda a: -len(a["name"])):
            if a["name"].lower() in q or re.search(rf"\b{a['id']}\b", q):
                return a
        return None

    def _after_write(self, call: dict, what: str) -> ModelTurn:
        if call["is_error"] and "NOT EXECUTED" in call["output"]:
            return self._say(
                f"I prepared the {what} but it needs human approval, so nothing was changed. "
                f"Proposed: {json.dumps(call['input'])}"
            )
        if call["is_error"]:
            return self._say(f"The {what} failed: {call['output']}")
        return self._say(f"Done: {what} created ({call['output']}).")


def _cents(money: str) -> int:
    neg = money.startswith("-")
    value = int(round(float(money.lstrip("-$").replace(",", "")) * 100))
    return -value if neg else value
