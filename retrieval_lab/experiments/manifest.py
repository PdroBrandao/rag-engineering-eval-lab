import json
from pathlib import Path

INDEX_SETS = (
    {"chunk_size": 256, "chunk_overlap": 26, "index_set_id": "cs256"},
    {"chunk_size": 512, "chunk_overlap": 51, "index_set_id": "cs512"},
    {"chunk_size": 1024, "chunk_overlap": 102, "index_set_id": "cs1024"},
)
TOP_KS = (3, 5, 10)
STRATEGIES = ("lexical", "vector", "hybrid")


def build_configs() -> list[dict]:
    configs = []
    for spec in INDEX_SETS:
        for top_k in TOP_KS:
            for strategy in STRATEGIES:
                configs.append(
                    {
                        **spec,
                        "top_k": top_k,
                        "strategy": strategy,
                        "config_id": f"{spec['index_set_id']}_top{top_k}_{strategy}",
                    }
                )
    return configs


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    retrieval = data["retrieval"]
    if retrieval["candidate_k"] != 20 or retrieval["rrf_k"] != 60:
        raise ValueError("Manifest must freeze candidate_k=20 and rrf_k=60.")
    configs = build_configs()
    if len(configs) != 27:
        raise ValueError("Expected 27 configs.")
    known = {item["config_id"] for item in configs}
    if data["baseline_config_id"] not in known:
        raise ValueError("Baseline config is not in the 27 configs.")
    data["configs"] = configs
    return data


def effective_collections(manifest_collections: list[str], read_collections: list[str]) -> list[str]:
    allowed = [name for name in manifest_collections if name in set(read_collections)]
    if not allowed:
        raise PermissionError("Runner identity cannot read the manifest collection.")
    return allowed
