from __future__ import annotations
import math

def ranking_metrics(ranking: list[int], target: int, cutoffs=(5, 10)) -> dict[str, float]:
    result = {f"HR@{k}": 0.0 for k in cutoffs} | {f"NDCG@{k}": 0.0 for k in cutoffs}
    rank = ranking.index(target) + 1 if target in ranking else 0
    for k in cutoffs:
        if rank and rank <= k: result[f"HR@{k}"], result[f"NDCG@{k}"] = 1.0, 1 / math.log2(rank + 1)
    result["MRR"] = 1 / rank if rank else 0.0
    return result
