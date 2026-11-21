from conftest import ReplayModel, calls, say, use

from close_agent.agent import ModelTurn, run_agent
from close_agent.tools import approve_all


def assert_transcript_valid(messages):
    """Roles alternate and every tool_use is answered by a tool_result in the very next message."""
    assert messages[0]["role"] == "user"
    for prev, cur in zip(messages, messages[1:], strict=False):
        assert prev["role"] != cur["role"]
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and isinstance(m["content"], list):
            ids = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
            if ids:
                nxt = messages[i + 1]["content"]
                assert [r["tool_use_id"] for r in nxt] == ids
                assert all(r["type"] == "tool_result" for r in nxt)


def test_plain_answer_needs_no_tools(ledger):
    run = run_agent(ReplayModel([say("Hi")]), "hello", ledger)
    assert run.final_text == "Hi" and run.stop_reason == "end_turn" and run.steps == 1 and run.tool_calls == []


def test_parallel_calls_return_in_one_message(ledger):
    model = ReplayModel(
        [
            calls(
                use("a", "reconcile_account", account_id="1000", period="2026-10"),
                use("b", "reconcile_account", account_id="1020", period="2026-10"),
            ),
            say("done"),
        ]
    )
    run = run_agent(model, "q", ledger)
    second = model.seen[1]
    assert len(second) == 3
    assert [r["tool_use_id"] for r in second[2]["content"]] == ["a", "b"]
    assert_transcript_valid(run.messages)


def test_errors_flow_back_and_model_can_recover(ledger):
    model = ReplayModel(
        [
            calls(use("a", "get_gl_balance", account_id="1000", period="October")),
            calls(use("b", "get_gl_balance", account_id="1000", period="2026-10")),
            say("$40,450.00"),
        ]
    )
    run = run_agent(model, "q", ledger)
    first_result = model.seen[1][2]["content"][0]
    assert first_result["is_error"] is True and "YYYY-MM" in first_result["content"]
    assert "is_error" not in model.seen[2][4]["content"][0]
    assert [c.is_error for c in run.tool_calls] == [True, False]
    assert_transcript_valid(run.messages)


def test_denied_write_is_recorded_as_proposal(ledger):
    model = ReplayModel(
        [
            calls(use("a", "create_close_task", title="Review cash", account_id="1000", assignee="Jordan")),
            say("Needs approval."),
        ]
    )
    run = run_agent(model, "q", ledger)
    assert [c.name for c in run.proposed_actions] == ["create_close_task"]
    assert run.executed_writes == []
    assert "NOT EXECUTED" in model.seen[1][2]["content"][0]["content"]


def test_approved_write_executes(ledger):
    model = ReplayModel(
        [
            calls(use("a", "create_close_task", title="Review cash", account_id="1000", assignee="Jordan")),
            say("Created."),
        ]
    )
    run = run_agent(model, "q", ledger, approver=approve_all)
    assert [c.name for c in run.executed_writes] == ["create_close_task"]


def test_max_steps_stops_a_looping_model(ledger):
    loop = [calls(use(f"t{i}", "list_accounts")) for i in range(10)]
    run = run_agent(ReplayModel(loop), "q", ledger, max_steps=3)
    assert run.stop_reason == "max_steps" and run.steps == 3 and len(run.tool_calls) == 3
    assert run.final_text == ""


def test_refusal_ends_the_run(ledger):
    run = run_agent(ReplayModel([say("", stop="refusal")]), "q", ledger)
    assert run.stop_reason == "refusal"


def test_pause_turn_continues(ledger):
    model = ReplayModel([ModelTurn(blocks=[], stop_reason="pause_turn"), say("ok")])
    run = run_agent(model, "q", ledger)
    assert run.final_text == "ok" and run.steps == 2


def test_raw_content_is_sent_back_unchanged(ledger):
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "x"}]
    model = ReplayModel([ModelTurn(blocks=[{"type": "text", "text": "x"}], stop_reason="end_turn", raw_content=raw)])
    run = run_agent(model, "q", ledger)
    assert run.messages[1]["content"] is raw


def test_tokens_are_summed(ledger):
    t1 = calls(use("a", "list_accounts"))
    t1.input_tokens, t1.output_tokens = 100, 10
    t2 = say("x")
    t2.input_tokens, t2.output_tokens = 150, 20
    run = run_agent(ReplayModel([t1, t2]), "q", ledger)
    assert (run.input_tokens, run.output_tokens) == (250, 30)
