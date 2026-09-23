from hdagentrec.trainer import AgentCFCandidatePool

def test_budget_is_twenty_and_positive_once():
    pool = AgentCFCandidatePool({1: list(range(2, 40))})
    candidates = pool.with_positive(1, 9, 20)
    assert len(candidates) == 20 and candidates.count(9) == 1
