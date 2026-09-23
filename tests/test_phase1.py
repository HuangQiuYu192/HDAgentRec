import pytest
from hdagentrec.agent import DynamicUserAgent, LLMClient
from hdagentrec.data import temporal_splits
from hdagentrec.memory import apply_memory_update
from hdagentrec.schemas import InteractionType, MemoryUpdate, UserState

def test_chronological_order_and_no_future_leakage():
    splits, _, _ = temporal_splits([("u", "a", 3), ("u", "b", 1), ("u", "c", 2), ("u", "d", 4)])
    assert splits.train[1] == [2, 3]
    assert splits.valid[1] == ([2, 3], 1)
    assert splits.test[1] == ([2, 3, 1], 4)
    assert splits.test[1][1] not in splits.valid[1][0]

def test_memory_update_is_deterministic():
    update = MemoryUpdate(interaction_type=InteractionType.stable_preference, confidence=.8, reason="repeat", concept="books", update_long_term=True, update_short_term=True)
    assert apply_memory_update(UserState(), update, 4, 1).model_dump() == apply_memory_update(UserState(), update, 4, 1).model_dump()

class Stub(LLMClient):
    def _generate(self, prompt): return ('{"ranking": [2, 1]}', 2)

def test_candidate_rankings_are_validated():
    assert DynamicUserAgent(Stub("stub")).rerank([1, 2, 3], "x") == [2, 1, 3]
    agent = DynamicUserAgent(Stub("stub"))
    assert agent.rerank([1, 3], "x") == [1, 3]
    assert agent.cost_metrics["invalid_reranks"] == 1


class DuplicateStub(LLMClient):
    def _generate(self, prompt): return ('{"ranking": [2, 2, 1]}', 3)


def test_duplicate_llm_ranking_is_deduplicated_before_validation():
    assert DynamicUserAgent(DuplicateStub("stub")).rerank([1, 2, 3], "x") == [2, 1, 3]
