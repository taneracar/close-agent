"""Tool registry: pydantic input models -> JSON schemas the Messages API accepts.

Every tool is either a read (runs immediately) or a write (runs only if the
approval policy says yes). Inputs are validated here, before any tool code
runs, so a malformed call becomes an `is_error` tool result the model can
recover from instead of an exception.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .ledger import Ledger, LedgerError, fmt_money

PERIOD = Field(description="Accounting period as YYYY-MM, for example 2026-10.")
ACCOUNT = Field(description="Account id from list_accounts, for example 1000.")


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoInput(_Input):
    pass


class AccountPeriod(_Input):
    account_id: str = ACCOUNT
    period: str = PERIOD


class CreateTask(_Input):
    title: str = Field(min_length=3, max_length=200)
    account_id: str = ACCOUNT
    assignee: str = Field(min_length=1, max_length=80)


class PostAdjustment(_Input):
    account_id: str = ACCOUNT
    entry_date: str = Field(description="YYYY-MM-DD")
    amount_cents: int = Field(description="Signed amount in cents; negative reduces the balance.")
    memo: str = Field(min_length=3, max_length=200)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[_Input]
    run: Callable[[Ledger, Any], Any]
    writes: bool = False

    def spec(self) -> dict:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        return {"name": self.name, "description": self.description, "input_schema": schema}


def _list_accounts(ledger: Ledger, _: NoInput) -> list[dict]:
    return [{"id": a.id, "name": a.name, "kind": a.kind, "bank_feed": a.bank_linked} for a in ledger.accounts()]


def _gl_balance(ledger: Ledger, args: AccountPeriod) -> dict:
    cents = ledger.gl_balance(args.account_id, args.period)
    return {"account_id": args.account_id, "period": args.period, "gl_balance": fmt_money(cents)}


def _reconcile(ledger: Ledger, args: AccountPeriod) -> dict:
    gl = ledger.gl_balance(args.account_id, args.period)
    bank = ledger.bank_balance(args.account_id, args.period)
    diff = gl - bank
    return {
        "account_id": args.account_id,
        "period": args.period,
        "gl_balance": fmt_money(gl),
        "bank_balance": fmt_money(bank),
        "difference_gl_minus_bank": fmt_money(diff),
        "status": "reconciled" if diff == 0 else "unreconciled",
    }


def _unmatched(ledger: Ledger, args: AccountPeriod) -> dict:
    return {"account_id": args.account_id, "period": args.period, **ledger.unmatched(args.account_id, args.period)}


def _create_task(ledger: Ledger, args: CreateTask) -> dict:
    task_id = ledger.create_task(args.title, args.account_id, args.assignee)
    return {"task_id": task_id, "status": "created"}


def _post_adjustment(ledger: Ledger, args: PostAdjustment) -> dict:
    entry_id = ledger.post_adjustment(args.account_id, args.entry_date, args.amount_cents, args.memo)
    return {"entry_id": entry_id, "status": "posted", "amount": fmt_money(args.amount_cents)}


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "list_accounts",
            "List the chart of accounts with ids and whether each has a bank feed.",
            NoInput,
            _list_accounts,
        ),
        Tool(
            "get_gl_balance",
            "General-ledger balance of one account at the end of a period.",
            AccountPeriod,
            _gl_balance,
        ),
        Tool(
            "reconcile_account",
            "Compare the GL balance with the bank balance at period end. Returns both balances, "
            "the difference and a status. Only for accounts with a bank feed.",
            AccountPeriod,
            _reconcile,
        ),
        Tool(
            "list_unmatched",
            "List GL entries with no matching bank line and bank lines with no matching GL entry, "
            "up to period end. Use it to explain a reconciliation difference.",
            AccountPeriod,
            _unmatched,
        ),
        Tool(
            "create_close_task",
            "Create a month-end close task assigned to a person. Needs human approval.",
            CreateTask,
            _create_task,
            writes=True,
        ),
        Tool(
            "post_adjusting_entry",
            "Post an adjusting journal entry to the GL. Needs human approval.",
            PostAdjustment,
            _post_adjustment,
            writes=True,
        ),
    ]
}


Approver = Callable[[str, dict], bool]


def deny_all(name: str, args: dict) -> bool:
    return False


def approve_all(name: str, args: dict) -> bool:
    return True


@dataclass
class ToolOutcome:
    content: str
    is_error: bool
    executed: bool
    denied: bool = False


def execute(ledger: Ledger, name: str, raw_input: Any, approver: Approver = deny_all) -> ToolOutcome:
    """Validate and run one tool call. Never raises for model mistakes."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolOutcome(f"unknown tool {name!r}", is_error=True, executed=False)
    try:
        args = tool.input_model.model_validate(raw_input if raw_input is not None else {})
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors())
        return ToolOutcome(f"invalid input: {problems}", is_error=True, executed=False)
    if tool.writes and not approver(name, args.model_dump()):
        return ToolOutcome(
            "NOT EXECUTED: this action needs human approval and was not approved. "
            "Tell the user what you propose instead.",
            is_error=True,
            executed=False,
            denied=True,
        )
    try:
        result = tool.run(ledger, args)
    except LedgerError as exc:
        return ToolOutcome(str(exc), is_error=True, executed=False)
    return ToolOutcome(json.dumps(result), is_error=False, executed=True)


def tool_specs() -> list[dict]:
    return [t.spec() for t in TOOLS.values()]
