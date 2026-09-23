"""Phase-1, future-safe dynamic-agent evaluation helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .agent import DynamicUserAgent
from .evaluation import ranking_metrics
from .schemas import UserState


def load_item_metadata(path: str | Path) -> dict[str, str]:
    """Load AgentCF/RecBole ``.item`` metadata without assuming fields exist."""
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    if not rows:
        return {}
    fields = [value.split(":", 1)[0] for value in rows[0].split("\t")]
    result: dict[str, str] = {}
    for row in rows[1:]:
        values = row.split("\t")
        if not values:
            continue
        payload = {key: value for key, value in zip(fields, values) if value}
        item_id = payload.pop("item_id", values[0])
        result[item_id] = "; ".join(f"{key}: {value}" for key, value in payload.items()) or f"item_id: {item_id}"
    return result


def _template(name: str) -> str:
    return ((Path(__file__).resolve().parent.parent / "prompts") / name).read_text(encoding="utf-8").strip()


def state_prompt(state: UserState, item_id: int, step: int, metadata: str) -> str:
    payload = {"state": state.model_dump(), "new_interaction": {"item_id": item_id, "step": step, "metadata": metadata}}
    return _template("dynamic_state_v1.txt") + "\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)


def rerank_prompt(state: UserState, recent_items: list[dict], candidates: list[dict], uncertainty: float) -> str:
    payload = {"user_state": state.model_dump(), "recent_interactions": recent_items, "candidates": candidates, "backbone_uncertainty": uncertainty}
    return _template("reranker_v1.txt") + "\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)


def replay_history(agent: DynamicUserAgent, history: Iterable[int], metadata_by_id: dict[int, str]) -> UserState:
    """Build a state strictly from pre-query interactions, in chronological order."""
    state = UserState()
    for step, item_id in enumerate(history, start=1):
        metadata = metadata_by_id.get(int(item_id), f"item_id: {item_id}")
        state = agent.update_state(state, int(item_id), step, metadata, state_prompt(state, int(item_id), step, metadata))
    return state


@dataclass
class Phase1Metrics:
    candidate_size: int = 20
    queries: int = 0
    candidate_hits: int = 0
    backbone: dict[str, float] = field(default_factory=dict)
    agent: dict[str, float] = field(default_factory=dict)

    def add(self, backbone_ranking: list[int], agent_ranking: list[int], target: int) -> None:
        self.queries += 1
        self.candidate_hits += int(target in backbone_ranking)
        for destination, values in ((self.backbone, ranking_metrics(backbone_ranking, target)), (self.agent, ranking_metrics(agent_ranking, target))):
            for key, value in values.items():
                destination[key] = destination.get(key, 0.0) + value

    def report(self) -> dict:
        denominator = max(self.queries, 1)
        return {"queries": self.queries, f"candidate_coverage@{self.candidate_size}": self.candidate_hits / denominator, "backbone": {key: value / denominator for key, value in self.backbone.items()}, "agent": {key: value / denominator for key, value in self.agent.items()}}
