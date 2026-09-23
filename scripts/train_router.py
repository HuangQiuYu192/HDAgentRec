"""Train an auditable LR router from full-agent JSONL trace labels.

The input must contain only examples on which the agent was invoked.  The
resulting classifier predicts the probability that invoking it improves the
single-positive NDCG@10 over SASRec.  It deliberately consumes no target-side
features, so it can be used online before invoking the agent.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hdagentrec.router import FEATURE_NAMES, agent_improved, features_from_trace


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("agent_invoked"):
                    rows.append(row)
    if not rows:
        raise ValueError("no agent-invoked trace rows found")
    return rows


def split_indices(rows: list[dict], labels: np.ndarray, seed: int, test_size: float):
    # Future traces include user_id.  Old traces lack it, so stratification is
    # retained but explicitly reported rather than pretending it is user split.
    groups = [row.get("user_id") for row in rows]
    if all(group is not None for group in groups) and len(set(groups)) > 1:
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(rows, labels, groups))
        return train, test, "user-disjoint"
    indices = np.arange(len(rows))
    train, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
    return train, test, "stratified-row (legacy traces without user_id)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    rows = load_rows(args.traces)
    x = np.asarray([features_from_trace(row) for row in rows], dtype=np.float64)
    y = np.asarray([agent_improved(row) for row in rows], dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("router labels contain only one class")
    train, test, split_name = split_indices(rows, y, args.seed, args.test_size)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=args.seed, class_weight="balanced", max_iter=1000))
    model.fit(x[train], y[train])
    probability = model.predict_proba(x[test])[:, 1]
    report = {
        "samples": int(len(rows)), "positive_labels": int(y.sum()), "split": split_name,
        "train_samples": int(len(train)), "test_samples": int(len(test)),
        "test_roc_auc": float(roc_auc_score(y[test], probability)),
        "test_average_precision": float(average_precision_score(y[test], probability)),
        "features": list(FEATURE_NAMES),
        "note": "A small trace set is a pipeline check, not a final performance claim.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(model, handle)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
