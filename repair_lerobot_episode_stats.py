#!/usr/bin/env python3
"""Repair LeRobot v2.1 episode stats missing `count` fields.

Some older local conversions wrote per-episode min/max/mean/std but omitted
`count`, which newer LeRobot versions require when aggregating stats.
This script updates `meta/episodes_stats.jsonl` in place and keeps a backup.
"""

import argparse
import json
import shutil
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Add missing `count` fields to LeRobot episodes_stats.jsonl."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create episodes_stats.jsonl.bak before writing.",
    )
    args = parser.parse_args()

    meta_dir = args.dataset_root / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"

    episodes = read_jsonl(episodes_path)
    episode_lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    rows = read_jsonl(episodes_stats_path)

    changed = 0
    for row in rows:
        episode_index = int(row["episode_index"])
        count = [episode_lengths[episode_index]]
        for feature_stats in row.get("stats", {}).values():
            if "count" not in feature_stats:
                feature_stats["count"] = count
                changed += 1

    if changed == 0:
        print(f"No missing count fields found in {episodes_stats_path}")
        return

    if not args.no_backup:
        backup_path = episodes_stats_path.with_suffix(episodes_stats_path.suffix + ".bak")
        shutil.copy2(episodes_stats_path, backup_path)
        print(f"Backed up original stats to {backup_path}")

    write_jsonl(episodes_stats_path, rows)
    print(f"Added {changed} missing count field(s) in {episodes_stats_path}")


if __name__ == "__main__":
    main()
