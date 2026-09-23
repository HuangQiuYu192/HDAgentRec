"""AgentCF-style RecBole model with a SASRec candidate generator."""
from __future__ import annotations

import torch

from .agentverse import DynamicUserRegistry

try:
    from recbole.model.abstract_recommender import SequentialRecommender
    from recbole.model.sequential_recommender import SASRec
    from recbole.utils import InputType
except ImportError as error:
    SequentialRecommender = None
    _IMPORT_ERROR = error


if SequentialRecommender is not None:
    class HDAgentRec(SequentialRecommender):
        """RecBole model mirroring AgentCF's model→agent-registry→trainer split.

        SASRec remains differentiable and trains only on past sequences.  The
        registry persists typed user agents across agent evaluation; an external
        agent trainer obtains ``candidate_topk`` and invokes the frozen LLM only
        for its top-M candidate set.
        """
        input_type = InputType.POINTWISE

        def __init__(self, config, dataset):
            super().__init__(config, dataset)
            self.config = config
            self.backbone = SASRec(config, dataset)
            self.agent_registry = DynamicUserRegistry()
            self.n_users = dataset.num(self.USER_ID)
            self.item_id_token = dataset.field2id_token[self.ITEM_ID]
            self.user_id_token = dataset.field2id_token[self.USER_ID]

        def calculate_loss(self, interaction):
            return self.backbone.calculate_loss(interaction)

        def predict(self, interaction):
            return self.backbone.predict(interaction)

        def full_sort_predict(self, interaction):
            return self.backbone.full_sort_predict(interaction)

        @torch.no_grad()
        def candidate_topk(self, interaction, k=None):
            k = k or self.config["agent_candidate_m"]
            scores = self.full_sort_predict(interaction).view(-1, self.n_items)
            scores[:, 0] = -torch.inf
            top_scores, top_items = torch.topk(scores, k=k, dim=-1)
            probabilities = torch.softmax(top_scores, dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
            return top_items, top_scores, entropy

        def get_user_state(self, internal_user_id: int):
            return self.agent_registry.get(internal_user_id)
else:
    class HDAgentRec:
        def __init__(self, *args, **kwargs):
            raise ImportError("Install recbole==1.2.1 to use HDAgentRec") from _IMPORT_ERROR
