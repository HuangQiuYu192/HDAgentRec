"""User-disjoint offline evaluation for a selective Agent router.

All input rows must be full-agent labels.  Users are partitioned before any
model/threshold fitting: 60% router fitting, 20% threshold validation, and
20% one-shot final testing.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hdagentrec.router import FEATURE_NAMES, agent_improved, features_from_trace, rank_ndcg


def load_full_agent_rows(paths: list[str]) -> list[dict]:
    rows, seen = [], set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("agent_invoked") is not True or "user_id" not in row:
                raise ValueError(f"{path} contains a non-full-agent or legacy row")
            user_id = int(row["user_id"])
            if user_id in seen:
                raise ValueError(f"duplicate user_id {user_id}")
            seen.add(user_id); rows.append(row)
    if len(rows) < 10:
        raise ValueError("too few full-agent rows")
    return rows


def user_splits(rows: list[dict], seed: int):
    users = np.asarray(sorted(int(row["user_id"]) for row in rows))
    rng = np.random.RandomState(seed); rng.shuffle(users)
    n_train = int(len(users) * .6); n_validation = int(len(users) * .2)
    return set(users[:n_train]), set(users[n_train:n_train + n_validation]), set(users[n_train + n_validation:])


def ndcg_for_route(rows: list[dict], invoke: np.ndarray) -> tuple[float, float]:
    ranks = np.asarray([row["agent_rank"] if called else row["backbone_rank"] for row, called in zip(rows, invoke)], dtype=int)
    return float(np.mean([rank_ndcg(int(rank)) for rank in ranks])), float(np.mean(invoke))


def pick_threshold(rows: list[dict], score: np.ndarray, higher: bool) -> tuple[float, float, float]:
    score = np.asarray(score, dtype=np.float64)
    candidates = np.unique(np.r_[0.0, score, 1.0])
    options = []
    for threshold in candidates:
        invoke = score >= threshold if higher else score <= threshold
        ndcg, rate = ndcg_for_route(rows, invoke)
        options.append((ndcg, -rate, float(threshold), rate))
    best = max(options)
    return best[2], best[0], best[3]


def summary(rows: list[dict], invoke: np.ndarray) -> dict:
    backbone = float(np.mean([rank_ndcg(int(row["backbone_rank"])) for row in rows]))
    full_agent = float(np.mean([rank_ndcg(int(row["agent_rank"])) for row in rows]))
    routed, rate = ndcg_for_route(rows, invoke)
    return {"NDCG@10_sasrec": backbone, "NDCG@10_full_agent": full_agent, "NDCG@10_router": routed, "agent_invocation_rate": rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    rows = load_full_agent_rows(args.traces)
    train_users, validation_users, test_users = user_splits(rows, args.seed)
    by_split = [[row for row in rows if int(row["user_id"]) in users] for users in (train_users, validation_users, test_users)]
    train, validation, test = by_split
    x_train = np.asarray([features_from_trace(row) for row in train], dtype=np.float64)
    y_train = np.asarray([agent_improved(row) for row in train], dtype=np.int64)
    if len(np.unique(y_train)) != 2:
        raise ValueError("train split has only one hindsight class")
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=args.seed, class_weight="balanced", max_iter=1000))
    model.fit(x_train, y_train)
    validation_probability = model.predict_proba(np.asarray([features_from_trace(row) for row in validation]))[:, 1]
    validation_entropy = np.asarray([row["candidate_entropy"] for row in validation])
    router_threshold, _, _ = pick_threshold(validation, validation_probability, higher=True)
    entropy_threshold, _, _ = pick_threshold(validation, validation_entropy, higher=True)
    test_probability = model.predict_proba(np.asarray([features_from_trace(row) for row in test]))[:, 1]
    test_entropy = np.asarray([row["candidate_entropy"] for row in test])
    report = {
        "protocol": "fixed user-disjoint 60/20/20; thresholds selected on validation only",
        "samples": len(rows), "split_users": {"train": len(train), "validation": len(validation), "test": len(test)},
        "features": list(FEATURE_NAMES), "router_probability_threshold": router_threshold, "entropy_threshold": entropy_threshold,
        "test": {"lr_router": summary(test, test_probability >= router_threshold), "entropy_router": summary(test, test_entropy >= entropy_threshold), "all_agent": summary(test, np.ones(len(test), dtype=bool)), "no_agent": summary(test, np.zeros(len(test), dtype=bool))},
    }
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_model, "wb") as handle:
        pickle.dump(model, handle)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
