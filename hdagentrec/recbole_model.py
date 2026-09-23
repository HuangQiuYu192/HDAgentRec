"""RecBole-native backbone adapter for HDAgentRec.

The class deliberately inherits RecBole's maintained SASRec implementation,
therefore it uses RecBole's SequentialDataset, data loaders, Trainer, temporal
splits, negative-sampling rules and full-sort evaluator unchanged.  LLM calls
remain outside ``full_sort_predict``: this keeps batched GPU evaluation pure and
lets the agent rerank only the exported top-M candidates.
"""
from __future__ import annotations

import torch

try:
    from recbole.model.sequential_recommender import SASRec
except ImportError as error:  # Allows state-only unit tests without RecBole installed.
    SASRec = None
    _RECB0LE_IMPORT_ERROR = error


if SASRec is not None:
    class HDAgentSASRec(SASRec):
        """SASRec with an explicit stable candidate-export interface for agents."""

        def candidate_topk(self, interaction, k: int | None = None):
            """Return top-M internal item IDs, scores, and entropy per query.

            This method intentionally only sees the RecBole test interaction and
            never user future events. It is the sole backbone-to-agent boundary.
            """
            k = k or self.config["agent_candidate_m"]
            scores = self.full_sort_predict(interaction).view(-1, self.n_items)
            scores[:, 0] = -torch.inf  # RecBole padding token
            top_scores, top_items = torch.topk(scores, k=k, dim=-1)
            probabilities = torch.softmax(top_scores, dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            return top_items, top_scores, entropy
else:
    class HDAgentSASRec:  # pragma: no cover - helpful import-time diagnostic
        def __init__(self, *args, **kwargs):
            raise ImportError("Install recbole==1.2.1 to use HDAgentSASRec") from _RECB0LE_IMPORT_ERROR
