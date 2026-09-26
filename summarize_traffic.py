import argparse
import csv
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
args = parser.parse_args()

groups = defaultdict(list)

with (args.results / "episodes.csv").open(
    newline="", encoding="utf-8-sig"
) as file:
    for row in csv.DictReader(file):
        if row["task"] == "intersection":
            groups[(row["policy"], row["training_seed"])].append(row)

fields = [
    "constraint_overrides",
    "wait_limit_overrides",
    "minimum_phase_overrides",
    "traffic_vehicles_served",
    "traffic_overflow",
    "traffic_remaining_at_end",
    "key_metric",
]

required = fields + ["evaluation_seed"]

if not groups:
    raise SystemExit("No traffic rows found in episodes.csv.")

for (policy, seed), rows in groups.items():
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise SystemExit(f"Missing diagnostic columns: {missing}")

    failures = []
    for row in rows:
        total = int(row["constraint_overrides"])
        parts = (
            int(row["wait_limit_overrides"])
            + int(row["minimum_phase_overrides"])
        )
        if total != parts:
            failures.append(row["evaluation_seed"])

    print(f"\nPolicy: {policy}; seed: {seed or 'baseline'}")
    print(f"Episodes: {len(rows)}")
    print(
        "Override accounting:",
        "PASS" if not failures else f"FAIL at episodes {failures}",
    )

    for field in fields:
        values = [
            float(row[field]) for row in rows
            if row[field] != ""
        ]
        average = sum(values) / len(values) if values else None
        print(
            f"{field}: "
            + (f"{average:.4f}" if average is not None else "N/A")
        )