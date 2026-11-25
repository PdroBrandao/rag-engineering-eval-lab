"""Deterministic Development / Frozen split for the 50-question Itaú set.

Q001–Q010 stay in Development because they already informed technical choices.
The other 40 are allocated by category and a recorded difficulty proxy.
The frozen file is immutable: any later change needs a new version.
"""

import hashlib
import json
import random
from pathlib import Path

SEED = "itau-50-v1"
LOCKED_DEVELOPMENT = tuple(f"Q{index:03d}" for index in range(1, 11))
EXPECTED_IDS = tuple(f"Q{index:03d}" for index in list(range(1, 24)) + list(range(26, 53)))
FROZEN_SIZE = 20
DEVELOPMENT_SIZE = 30


def difficulty_of(row: dict) -> str:
    if row.get("table_ambiguous"):
        return "hard"
    mode = (row.get("grading") or {}).get("mode")
    spans = row.get("minimal_support_spans") or []
    if mode in {"judge", "semantic_review"} or len(spans) > 1 or row.get("category") in {
        "desambiguacao",
        "multitrecho",
    }:
        return "medium"
    return "easy"


def hamilton(weights: dict[str, int], seats: int) -> dict[str, int]:
    total = sum(weights.values())
    if seats < 0 or seats > total:
        raise ValueError("Seat count is outside the number of items.")
    if total == 0 or seats == 0:
        return {key: 0 for key in weights}
    raw = {key: weights[key] * seats / total for key in weights}
    base = {key: int(raw[key]) for key in weights}
    leftover = seats - sum(base.values())
    ranked = sorted(weights, key=lambda key: (-(raw[key] - base[key]), key))
    for key in ranked[:leftover]:
        base[key] += 1
    return base


def split_dataset(rows: list[dict], *, seed: str = SEED, frozen_size: int = FROZEN_SIZE) -> dict:
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids.")
    if set(ids) != set(EXPECTED_IDS):
        raise ValueError("Dataset IDs must be Q001–Q023 and Q026–Q052.")
    by_id = {row["question_id"]: row for row in rows}
    missing_locked = [item for item in LOCKED_DEVELOPMENT if item not in by_id]
    if missing_locked:
        raise ValueError("Development lock is missing questions.")
    pool = [by_id[item] for item in ids if item not in set(LOCKED_DEVELOPMENT)]
    categories = sorted({row["category"] for row in pool})
    frozen_by_category = hamilton(
        {category: sum(1 for row in pool if row["category"] == category) for category in categories},
        frozen_size,
    )
    frozen: list[str] = []
    development_extra: list[str] = []
    allocation = []
    for category in categories:
        members = [row for row in pool if row["category"] == category]
        difficulties = sorted({difficulty_of(row) for row in members})
        quota = hamilton(
            {level: sum(1 for row in members if difficulty_of(row) == level) for level in difficulties},
            frozen_by_category[category],
        )
        for level in difficulties:
            group = sorted(row["question_id"] for row in members if difficulty_of(row) == level)
            shuffled = list(group)
            random.Random(f"{seed}|{category}|{level}").shuffle(shuffled)
            take = quota[level]
            chosen = shuffled[:take]
            frozen.extend(chosen)
            development_extra.extend(shuffled[take:])
            allocation.append(
                {
                    "category": category,
                    "difficulty": level,
                    "frozen_ids": chosen,
                    "development_ids": shuffled[take:],
                }
            )
    development = [item for item in LOCKED_DEVELOPMENT] + development_extra
    if set(development) & set(frozen) or set(development) | set(frozen) != set(EXPECTED_IDS):
        raise ValueError("Split does not partition the 50 questions.")
    if len(development) != DEVELOPMENT_SIZE or len(frozen) != frozen_size:
        raise ValueError("Split sizes are not 30 and 20.")
    order = {question_id: index for index, question_id in enumerate(ids)}
    return {
        "seed": seed,
        "difficulty_proxy": "hard=table_ambiguous; medium=judge|semantic_review|multi-span|desambiguacao|multitrecho; else easy",
        "tie_break": "Hamilton largest fractional part, then name ascending; shuffle is Python random.Random of the seed string",
        "development_ids": sorted(development, key=order.get),
        "frozen_ids": sorted(frozen, key=order.get),
        "frozen_by_category": frozen_by_category,
        "allocation": allocation,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_split_files(source: Path, destination: Path) -> dict:
    from retrieval_lab.experiments.ground_truth import load_suite

    rows = load_suite(source)
    plan = split_dataset(rows)
    lines = {
        json.loads(line)["question_id"]: line if line.endswith("\n") else line + "\n"
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    development_path = destination / "development_v1.jsonl"
    frozen_path = destination / "frozen_benchmark_v1.jsonl"
    development_path.write_text("".join(lines[item] for item in plan["development_ids"]), encoding="utf-8")
    frozen_path.write_text("".join(lines[item] for item in plan["frozen_ids"]), encoding="utf-8")
    manifest = {
        "version": "v1",
        "seed": plan["seed"],
        "difficulty_proxy": plan["difficulty_proxy"],
        "tie_break": plan["tie_break"],
        "source": str(source),
        "source_sha256": sha256_file(source),
        "locked_development_ids": list(LOCKED_DEVELOPMENT),
        "development": {
            "path": str(development_path),
            "sha256": sha256_file(development_path),
            "n": len(plan["development_ids"]),
            "ids": plan["development_ids"],
        },
        "frozen_benchmark": {
            "path": str(frozen_path),
            "sha256": sha256_file(frozen_path),
            "n": len(plan["frozen_ids"]),
            "ids": plan["frozen_ids"],
            "immutable": True,
            "change_policy": "Do not edit frozen_benchmark_v1.jsonl. A change requires frozen_benchmark_v2.",
        },
        "frozen_by_category": plan["frozen_by_category"],
        "allocation": plan["allocation"],
        "counts": _counts(rows, plan),
    }
    (destination / "dataset_split_v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _counts(rows: list[dict], plan: dict) -> dict:
    by_id = {row["question_id"]: row for row in rows}
    summary = {}
    for name, ids in (("development", plan["development_ids"]), ("frozen_benchmark", plan["frozen_ids"])):
        categories: dict[str, int] = {}
        difficulties: dict[str, int] = {}
        for question_id in ids:
            row = by_id[question_id]
            categories[row["category"]] = categories.get(row["category"], 0) + 1
            level = difficulty_of(row)
            difficulties[level] = difficulties.get(level, 0) + 1
        summary[name] = {"category": categories, "difficulty": difficulties}
    return summary
