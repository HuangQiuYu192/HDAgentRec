from hdagentrec.router import agent_improved, features_from_trace, rank_ndcg
from scripts.train_router import load_rows


def test_router_features_exclude_target_side_ranks():
    row = {"candidate_entropy": 2.1, "top1_top2_margin": 0.3, "backbone_rank": 10, "agent_rank": 1}
    assert features_from_trace(row) == [2.1, 0.3]


def test_hindsight_label_is_strict_ndcg_improvement():
    assert rank_ndcg(1) > rank_ndcg(2)
    assert agent_improved({"backbone_rank": 3, "agent_rank": 2}) == 1
    assert agent_improved({"backbone_rank": 2, "agent_rank": 3}) == 0


def test_legacy_agent_trace_requires_an_explicit_declaration(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"backbone_rank": 2, "agent_rank": 1}\n', encoding="utf-8")
    try:
        load_rows([str(path)], assume_agent_invoked=False)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy rows must not be silently accepted")
    assert len(load_rows([str(path)], assume_agent_invoked=True)) == 1
