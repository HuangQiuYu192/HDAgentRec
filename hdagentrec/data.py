from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class SequenceSplits:
    train: dict[int, list[int]]
    valid: dict[int, tuple[list[int], int]]
    test: dict[int, tuple[list[int], int]]


def temporal_splits(events: list[tuple[str, str, float]]) -> tuple[SequenceSplits, dict[str, int], dict[str, int]]:
    """Sort by timestamp and reserve final two interactions for validation/test."""
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for user, item, timestamp in events:
        grouped[str(user)].append((float(timestamp), str(item)))
    users = {user: index + 1 for index, user in enumerate(sorted(grouped))}
    item_ids = sorted({item for values in grouped.values() for _, item in values})
    items = {item: index + 1 for index, item in enumerate(item_ids)}
    train, valid, test = {}, {}, {}
    for raw_user, values in grouped.items():
        sequence = [items[item] for _, item in sorted(values, key=lambda pair: pair[0])]
        if len(sequence) < 3:
            continue
        user = users[raw_user]
        train[user] = sequence[:-2]
        valid[user] = (sequence[:-2], sequence[-2])
        test[user] = (sequence[:-1], sequence[-1])
    return SequenceSplits(train, valid, test), users, items


def load_amazon_csv(path: Union[str, Path], user_col="user_id", item_col="item_id", time_col="timestamp") -> list[tuple[str, str, float]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [(row[user_col], row[item_col], float(row[time_col])) for row in csv.DictReader(handle)]


def load_movielens_1m(path: Union[str, Path]) -> list[tuple[str, str, float]]:
    with Path(path).open(encoding="latin-1") as handle:
        return [(p[0], p[1], float(p[3])) for line in handle if (p := line.rstrip("\n").split("::"))]
