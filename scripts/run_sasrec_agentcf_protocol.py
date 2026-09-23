"""SASRec baseline under the paper's sampled-candidate protocol.

The released AgentCF trainer only uses the first nine entries in each supplied
100-item random pool. The paper instead specifies three repetitions with nine
random negatives. This runner samples from that released pool without
replacement for each repetition and reports their mean. It deliberately uses
RecBole's native SASRec for a direct baseline reproduction.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from hdagentrec.trainer import AgentCFCandidatePool
from recbole.model.sequential_recommender import SASRec


def evaluate(model, loader, pool, candidate_size, repeats, seed):
    if candidate_size != 10:
        raise ValueError("The AgentCF paper evaluates one positive against nine negatives.")
    model.eval(); rng = np.random.RandomState(seed); totals = {f"Recall@{k}": 0.0 for k in (1, 3, 5, 7, 10)}
    totals |= {f"NDCG@{k}": 0.0 for k in (1, 3, 5, 7, 10)} | {"MRR@10": 0.0}; count = 0
    with torch.no_grad():
        for interaction, _history, _positive_u, positive_i in loader:
            interaction = interaction.to(model.device)
            all_scores = model.full_sort_predict(interaction).view(len(interaction), model.n_items)
            users = interaction[model.USER_ID].cpu().tolist()
            positives = positive_i.cpu().tolist()
            for row, user, positive in zip(all_scores, users, positives):
                negatives = pool.negatives_for(user, positive)
                if len(negatives) < candidate_size - 1:
                    raise ValueError(f"user {user} has only {len(negatives)} negatives")
                for _ in range(repeats):
                    candidates = rng.choice(negatives, size=candidate_size - 1, replace=False).tolist() + [positive]
                    rng.shuffle(candidates)  # AgentCF's fix_pos=-1 position policy
                    candidate_tensor = torch.tensor(candidates, device=model.device)
                    ranking = candidate_tensor[row[candidate_tensor].argsort(descending=True)].tolist()
                    rank = ranking.index(positive) + 1
                    for k in (1, 3, 5, 7, 10):
                        if rank <= k:
                            totals[f"Recall@{k}"] += 1
                            totals[f"NDCG@{k}"] += 1 / math.log2(rank + 1)
                    if rank <= 10: totals["MRR@10"] += 1 / rank
                    count += 1
    return {key: round(value / count, 4) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default="CDs-100-user-dense")
    parser.add_argument("--config", default="configs/agentcf_cds_100_user_dense.yaml"); args = parser.parse_args()
    config = Config(model=SASRec, dataset=args.dataset, config_file_list=[args.config])
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config); train_data, valid_data, test_data = data_preparation(config, dataset)
    model = SASRec(config, train_data.dataset).to(config["device"])
    path = Path(config["data_path"]) / config["agentcf_candidate_file"]
    pool = AgentCFCandidatePool.from_file(path, dataset.field2token_id[model.USER_ID], dataset.field2token_id[model.ITEM_ID])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, best_epoch, stale, best_state = -1.0, -1, 0, None
    for epoch in range(config["epochs"]):
        model.train()
        for interaction in train_data:
            loss = model.calculate_loss(interaction.to(model.device)); optimizer.zero_grad(); loss.backward(); optimizer.step()
        valid = evaluate(model, valid_data, pool, config["recall_budget"], config["eval_repeats"], config["seed"])
        score = valid["NDCG@10"]
        if score > best:
            best, best_epoch, stale, best_state = score, epoch, 0, copy.deepcopy(model.state_dict())
        else: stale += 1
        if stale >= config["stopping_step"]: break
    model.load_state_dict(best_state)
    test = evaluate(model, test_data, pool, config["recall_budget"], config["eval_repeats"], config["seed"] + 10000)
    print({"best_epoch": best_epoch, "best_valid": best, "test": test, "protocol": "paper: 1 positive + 9 sampled negatives, 3 repetitions"})


if __name__ == "__main__": main()
