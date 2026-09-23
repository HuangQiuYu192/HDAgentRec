"""Small AgentCF-inspired registry for HDAgentRec agents.

Unlike AgentCF's mutable free-text ``update_memory`` lists, this registry owns
one typed ``UserState`` per RecBole internal user ID. The model and trainer may
only ask an agent for a proposed operation; ``apply_memory_update`` is the sole
state mutation point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agent import DynamicUserAgent
from .schemas import UserState


@dataclass
class DynamicUserRegistry:
    agent: DynamicUserAgent | None = None
    states: dict[int, UserState] = field(default_factory=dict)

    def get(self, user_id: int) -> UserState:
        return self.states.setdefault(int(user_id), UserState())

    def replace(self, user_id: int, state: UserState) -> None:
        self.states[int(user_id)] = state

    def reset(self) -> None:
        self.states.clear()
