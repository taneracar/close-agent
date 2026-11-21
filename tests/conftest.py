import copy

import pytest

from close_agent.agent import ModelTurn
from close_agent.ledger import Ledger


class ReplayModel:
    """Returns pre-written turns in order and records what it was sent."""

    name = "replay"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def respond(self, system, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        if not self.turns:
            raise AssertionError("ReplayModel ran out of turns")
        return self.turns.pop(0)


def use(id_, name, **inp):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def calls(*uses):
    return ModelTurn(blocks=list(uses), stop_reason="tool_use")


def say(text, stop="end_turn"):
    return ModelTurn(blocks=[{"type": "text", "text": text}], stop_reason=stop)


@pytest.fixture
def ledger():
    return Ledger.seeded()
