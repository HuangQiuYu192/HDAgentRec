"""Evaluation bridge modelled on AgentCF's LanguageLossTrainer boundary."""
from __future__ import annotations

from dataclasses import dataclass

from .agent import DynamicUserAgent


@dataclass
class AgentCandidateBatch:
    user_id: int
    candidate_ids: list[int]
    candidate_scores: list[float]
    uncertainty: float


class DynamicAgentEvaluator:
    """Calls a frozen agent only after RecBole has produced its top-M candidates."""
    def __init__(self, model, agent: DynamicUserAgent):
        self.model, self.agent = model, agent

    def rerank_one(self, user_id: int, candidate_ids: list[int], prompt: str) -> list[int]:
        # Candidate membership validation is enforced by DynamicUserAgent.
        return self.agent.rerank(candidate_ids, prompt)
