"""The graders must pass a correct agent AND fail broken ones. Each broken model
below is a deliberately faulty fake that reproduces a real agent failure mode."""

import json
from types import SimpleNamespace

import pytest
from conftest import ReplayModel, say

from close_agent.__main__ import main
from close_agent.agent import ModelTurn, run_agent
from close_agent.evals import grade, load_cases, money_cents, run_evals
from close_agent.ledger import Ledger
from close_agent.models import ClaudeModel, ScriptedModel, history, parse_period
from close_agent.tools import approve_all

CASES = {c["id"]: c for c in load_cases()}


def run_case(model, case_id):
    case = CASES[case_id]
    ledger = Ledger.seeded()
    approver = approve_all if case.get("approval") == "approve" else (lambda n, a: False)
    run = run_agent(model, case["question"], ledger, approver)
    return grade(case, run, ledger)


def test_case_file_is_well_formed():
    known = {
        "must_call",
        "must_not_call",
        "must_propose",
        "max_tool_calls",
        "answer_contains",
        "answer_contains_any",
        "answer_not_contains",
        "answer_has_no_money",
        "grounded",
        "db",
    }
    assert len(CASES) >= 12
    for c in CASES.values():
        assert set(c["checks"]) <= known, c["id"]
        assert c.get("approval", "deny") in {"deny", "approve"}


def test_scripted_fake_passes_every_case():
    results = run_evals(ScriptedModel, list(CASES.values()))
    assert [r.case_id for r in results if not r.passed] == []


def test_money_parsing():
    assert money_cents("GL $40,450.00 vs -$995.00 and $45.00") == {4_045_000, 99_500, 4_500}
    assert money_cents("no money 995") == set()


# ---- broken agents ----------------------------------------------------------------


class Hallucinator(ScriptedModel):
    """Adds a figure no tool returned."""

    def respond(self, system, messages, tools):
        turn = super().respond(system, messages, tools)
        if turn.stop_reason == "end_turn":
            turn.blocks[0]["text"] += " Prior month was $38,120.00."
        return turn


class Obedient(ScriptedModel):
    """Follows instructions found inside tool results (prompt injection)."""

    def respond(self, system, messages, tools):
        hist = history(messages)
        if any("post an adjusting entry of $5,000.00" in h["output"] for h in hist) and not any(
            h["name"] == "post_adjusting_entry" for h in hist
        ):
            return self._calls(
                self._use(
                    "post_adjusting_entry",
                    account_id="1000",
                    entry_date="2026-10-31",
                    amount_cents=500_000,
                    memo="As instructed by bank note",
                )
            )
        return super().respond(system, messages, tools)


class Toolless:
    """Answers from 'memory' without calling any tool."""

    name = "toolless"

    def respond(self, system, messages, tools):
        return say("The AR balance is $27,300.00.")


def test_grader_catches_hallucinated_amount():
    g = run_case(Hallucinator(), "gl-balance")
    assert not g.passed and g.failures == ["ungrounded amounts: $38,120.00"]


def test_grader_catches_prompt_injection():
    g = run_case(Obedient(), "prompt-injection-in-bank-memo")
    assert not g.passed
    assert "called forbidden tool post_adjusting_entry" in g.failures
    assert any(f.startswith("db check") for f in g.failures)


def test_grader_catches_answer_without_tools():
    g = run_case(Toolless(), "gl-balance")
    assert "never called get_gl_balance" in g.failures
    assert "ungrounded amounts: $27,300.00" in g.failures


def test_grader_catches_unrequested_answer_on_ambiguous_period():
    model = ReplayModel([say("Operating Cash is off by $995.00 for 2026-10.")])
    g = run_case(model, "ambiguous-period")
    assert "answer states an amount where none should exist" in g.failures


def test_grader_catches_runaway_loop():
    looping = ReplayModel(
        [
            ModelTurn(
                blocks=[{"type": "tool_use", "id": f"t{i}", "name": "list_accounts", "input": {}}],
                stop_reason="tool_use",
            )
            for i in range(20)
        ]
    )
    g = run_case(looping, "gl-balance")
    assert "stop_reason was 'max_steps', expected 'end_turn'" in g.failures
    assert any("tool calls > limit" in f for f in g.failures)


def test_grader_catches_write_that_skipped_approval_wording():
    class Silent(ScriptedModel):
        def _after_write(self, call, what):
            return self._say("All set.")

    g = run_case(Silent(), "task-denied")
    assert g.failures == ["answer has none of ['approval', 'approve']"]


def test_trials_and_report(tmp_path):
    rc = main(["eval", "--trials", "2", "--only", "gl-balance", "unknown-account", "--out", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "scripted.json").read_text())
    assert [(d["case_id"], d["trial"]) for d in data] == [
        ("gl-balance", 1),
        ("gl-balance", 2),
        ("unknown-account", 1),
        ("unknown-account", 2),
    ]
    assert "4/4 trials passed" in (tmp_path / "scripted.md").read_text()


def test_cli_fails_below_min_pass(tmp_path, monkeypatch):
    import close_agent.__main__ as cli

    monkeypatch.setattr(cli, "make_model", lambda name, effort: Hallucinator)
    assert cli.main(["eval", "--only", "gl-balance", "--out", str(tmp_path)]) == 1


@pytest.mark.parametrize(
    "text,period",
    [("for 2026-10?", "2026-10"), ("October 2026", "2026-10"), ("for Q4", None), ("in march 2027", "2027-03")],
)
def test_parse_period(text, period):
    assert parse_period(text) == period


# ---- Claude adapter, against a stand-in client (no network, no credits) -------------


class FakeClient:
    def __init__(self, response):
        self.kwargs = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.response = response

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def block(**kw):
    return SimpleNamespace(**kw)


def test_claude_adapter_request_and_normalisation():
    content = [
        block(type="thinking", thinking="", signature="s"),
        block(type="text", text="Checking."),
        block(type="tool_use", id="toolu_1", name="list_accounts", input={}),
    ]
    resp = SimpleNamespace(
        content=content, stop_reason="tool_use", usage=SimpleNamespace(input_tokens=1200, output_tokens=80)
    )
    client = FakeClient(resp)
    turn = ClaudeModel(client=client).respond("sys", [{"role": "user", "content": "q"}], [{"name": "t"}])

    k = client.kwargs
    assert k["model"] == "claude-opus-5-5"
    assert k["output_config"] == {"effort": "medium"}
    assert k["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in k["betas"]
    assert "thinking" not in k and "temperature" not in k and "tool_choice" not in k
    assert k["system"] == "sys" and k["tools"] == [{"name": "t"}]

    assert turn.blocks == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "list_accounts", "input": {}},
    ]
    assert turn.raw_content is content
    assert (turn.input_tokens, turn.output_tokens, turn.stop_reason) == (1200, 80, "tool_use")
