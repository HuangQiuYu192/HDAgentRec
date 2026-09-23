"""Evaluation bridge modelled on AgentCF's LanguageLossTrainer boundary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent import DynamicUserAgent


@dataclass
class AgentCandidateBatch:
    user_id: int
    candidate_ids: list[int]
    candidate_scores: list[float]
    uncertainty: float


class AgentCFCandidatePool:
    """Read AgentCF's ``<user>\t<negative item ...>`` candidate artifact.

    The protocol mirrors AgentCF's evaluation preparation: remove an accidental
    positive from the sampled pool, retain ``budget - 1`` negatives, and append
    the held-out positive. The caller may shuffle the resulting list with the
    configured experiment seed when matching AgentCF's ``fix_pos=-1`` setting.
    """
    def __init__(self, pools: dict[int, list[int]]): self.pools = pools

    @classmethod
    def from_file(cls, path: str | Path, user_token_to_id: dict, item_token_to_id: dict):
        pools = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                user, items = line.rstrip("\n").split("\t")
                if user in user_token_to_id:
                    pools[user_token_to_id[user]] = [item_token_to_id[item] for item in items.split() if item in item_token_to_id]
        return cls(pools)

    def with_positive(self, user_id: int, positive_id: int, budget: int = 20) -> list[int]:
        if budget < 2: raise ValueError("AgentCF candidate budget must include a positive and a negative")
        negatives = [item for item in self.pools[int(user_id)] if item != int(positive_id)]
        return negatives[: budget - 1] + [int(positive_id)]


class DynamicAgentEvaluator:
    """Calls a frozen agent only after RecBole has produced its top-M candidates."""
    def __init__(self, model, agent: DynamicUserAgent):
        self.model, self.agent = model, agent

    def rerank_one(self, user_id: int, candidate_ids: list[int], prompt: str) -> list[int]:
        # Candidate membership validation is enforced by DynamicUserAgent.
        return self.agent.rerank(candidate_ids, prompt)
