"""Competition submission validation and writing."""

from __future__ import annotations

import csv
from pathlib import Path


def read_submission_users(sample_path: Path) -> list[int]:
    with sample_path.open(newline="") as source:
        return [int(row["user_id"]) for row in csv.DictReader(source)]


def write_submission(
    path: Path, users: list[int], recommendations: dict[int, list[int]], k: int = 10
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["user_id", "item_id"])
        writer.writeheader()
        for user in users:
            items = recommendations.get(user, [])
            if len(items) != k or len(set(items)) != k:
                raise ValueError(f"user {user} must have exactly {k} unique recommendations")
            writer.writerow({"user_id": user, "item_id": " ".join(map(str, items))})
