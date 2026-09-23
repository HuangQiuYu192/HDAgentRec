from scripts.evaluate_router_offline import ndcg_for_route, pick_threshold, user_splits


def test_router_threshold_selection_uses_the_best_validation_route():
    rows = [
        {"backbone_rank": 2, "agent_rank": 1},
        {"backbone_rank": 1, "agent_rank": 4},
    ]
    threshold, _, rate = pick_threshold(rows, score=[.9, .1], higher=True)
    assert threshold > .1 and rate == .5


def test_user_split_is_disjoint_and_exhaustive():
    rows = [{"user_id": index} for index in range(10)]
    splits = user_splits(rows, 2024)
    assert sum(map(len, splits)) == 10
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
