import json

import pytest

from close_agent.ledger import Ledger, LedgerError, fmt_money, period_end
from close_agent.tools import TOOLS, approve_all, deny_all, execute, tool_specs


def test_money_format():
    assert fmt_money(4_045_000) == "$40,450.00"
    assert fmt_money(-31_027) == "-$310.27"
    assert fmt_money(0) == "$0.00"


@pytest.mark.parametrize(
    "period,end", [("2026-10", "2026-10-31"), ("2028-02", "2028-02-29"), ("2026-02", "2026-02-28")]
)
def test_period_end(period, end):
    assert period_end(period) == end


@pytest.mark.parametrize("bad", ["Q4", "2026-13", "October", "2026/10", ""])
def test_period_end_rejects(bad):
    with pytest.raises(LedgerError):
        period_end(bad)


def test_balances_are_cumulative_to_period_end(ledger: Ledger):
    assert ledger.gl_balance("1000", "2026-09") == 5_000_000
    assert ledger.gl_balance("1000", "2026-10") == 4_045_000
    assert ledger.bank_balance("1000", "2026-10") == 3_945_500
    # The Nov 2 deposit clears in November.
    assert ledger.bank_balance("1000", "2026-11") == 4_365_500


def test_payroll_reconciles_and_savings_does_not(ledger: Ledger):
    assert ledger.gl_balance("1010", "2026-10") == ledger.bank_balance("1010", "2026-10") == 517_945
    assert ledger.gl_balance("1020", "2026-10") - ledger.bank_balance("1020", "2026-10") == -31_027


def test_unmatched_explains_the_whole_difference(ledger: Ledger):
    um = ledger.unmatched("1000", "2026-10")
    assert [(x["amount"], x["date"]) for x in um["gl_only"]] == [
        ("-$3,250.00", "2026-10-15"),
        ("$4,200.00", "2026-10-30"),
    ]
    assert [x["amount"] for x in um["bank_only"]] == ["-$45.00"]
    # GL-only items minus bank-only items == GL minus bank.
    assert (-325_000 + 420_000) - (-4_500) == ledger.gl_balance("1000", "2026-10") - ledger.bank_balance(
        "1000", "2026-10"
    )


def test_matching_is_one_to_one(ledger: Ledger):
    ledger.conn.execute(
        "INSERT INTO gl_entries (account_id, entry_date, amount_cents, memo) "
        "VALUES ('1000','2026-10-11',-800000,'Duplicate rent')"
    )
    um = ledger.unmatched("1000", "2026-10")
    assert {"date": "2026-10-11", "amount": "-$8,000.00", "memo": "Duplicate rent"} in um["gl_only"]


def test_matching_respects_date_window(ledger: Ledger):
    ledger.conn.execute(
        "INSERT INTO gl_entries (account_id, entry_date, amount_cents, memo) VALUES ('1010','2026-10-01',100,'a')"
    )
    ledger.conn.execute(
        "INSERT INTO bank_lines (account_id, line_date, amount_cents, memo) VALUES ('1010','2026-10-07',100,'b')"
    )
    um = ledger.unmatched("1010", "2026-10")
    assert len(um["gl_only"]) == 1 and len(um["bank_only"]) == 1


def test_specs_are_valid_api_tool_definitions():
    specs = tool_specs()
    assert {s["name"] for s in specs} == set(TOOLS)
    for s in specs:
        assert s["input_schema"]["type"] == "object"
        assert s["input_schema"]["additionalProperties"] is False
        assert "title" not in s["input_schema"]
    rec = next(s for s in specs if s["name"] == "reconcile_account")
    assert rec["input_schema"]["required"] == ["account_id", "period"]


def test_execute_read_tool(ledger):
    out = execute(ledger, "reconcile_account", {"account_id": "1000", "period": "2026-10"})
    assert not out.is_error and out.executed
    assert json.loads(out.content)["difference_gl_minus_bank"] == "$995.00"


@pytest.mark.parametrize(
    "name,inp,needle",
    [
        ("nope", {}, "unknown tool"),
        ("get_gl_balance", {"account_id": "1000"}, "period: Field required"),
        ("get_gl_balance", {"account_id": "1000", "period": "2026-10", "x": 1}, "Extra inputs"),
        ("get_gl_balance", {"account_id": "9999", "period": "2026-10"}, "unknown account"),
        ("get_gl_balance", {"account_id": "1000", "period": "Q4"}, "YYYY-MM"),
        ("reconcile_account", {"account_id": "1200", "period": "2026-10"}, "no bank feed"),
        ("get_gl_balance", None, "Field required"),
    ],
)
def test_execute_turns_mistakes_into_error_results(ledger, name, inp, needle):
    out = execute(ledger, name, inp, approve_all)
    assert out.is_error and not out.executed
    assert needle in out.content


def test_write_tools_are_blocked_without_approval(ledger):
    args = {"title": "Review cash", "account_id": "1000", "assignee": "Jordan"}
    out = execute(ledger, "create_close_task", args, deny_all)
    assert out.denied and out.is_error and not out.executed
    assert ledger.conn.execute("SELECT COUNT(*) FROM tasks").fetchone() == (0,)
    out = execute(ledger, "create_close_task", args, approve_all)
    assert out.executed
    assert ledger.conn.execute("SELECT COUNT(*) FROM tasks").fetchone() == (1,)


def test_approver_is_not_asked_about_invalid_calls(ledger):
    asked = []
    out = execute(ledger, "post_adjusting_entry", {"account_id": "1000"}, lambda n, a: asked.append(n) or True)
    assert out.is_error and not out.denied and asked == []


def test_approver_sees_validated_args(ledger):
    seen = []
    execute(
        ledger,
        "post_adjusting_entry",
        {"account_id": "1000", "entry_date": "2026-10-31", "amount_cents": -4500, "memo": "Bank fee"},
        lambda n, a: seen.append((n, a)) or False,
    )
    assert seen == [
        (
            "post_adjusting_entry",
            {"account_id": "1000", "entry_date": "2026-10-31", "amount_cents": -4500, "memo": "Bank fee"},
        )
    ]


def test_zero_adjustment_rejected(ledger):
    out = execute(
        ledger,
        "post_adjusting_entry",
        {"account_id": "1000", "entry_date": "2026-10-31", "amount_cents": 0, "memo": "noop"},
        approve_all,
    )
    assert out.is_error and "zero" in out.content
