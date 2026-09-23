"""Small, auditable selective-invocation router for phase 1."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# These are available before the LLM is called.  Do not add target rank or any
# relevance signal here: those are label-only quantities and would leak test
# information into the router.
FEATURE_NAMES = ("candidate_entropy", "top1_top2_margin")


def features_from_trace(row: dict[str, Any]) -> list[float]:
    return [float(row[name]) for name in FEATURE_NAMES]


def rank_ndcg(rank: int) -> float:
    """NDCG@10 contribution for the sole held-out positive."""
    return 1.0 / math.log2(rank + 1) if 1 <= rank <= 10 else 0.0


def agent_improved(row: dict[str, Any]) -> int:
    """Binary hindsight label: did the agent strictly improve NDCG@10?"""
    return int(rank_ndcg(int(row["agent_rank"])) > rank_ndcg(int(row["backbone_rank"])))


def invocation_probability(model: Any, entropy: float, margin: float) -> float:
    values = np.asarray([[entropy, margin]], dtype=np.float64)
    return float(model.predict_proba(values)[0, 1])
