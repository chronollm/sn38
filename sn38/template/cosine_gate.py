"""Cosine similarity gate — reject models too similar to the baseline.

Loads all baselines into VRAM at startup. Compares candidate models
using batched per-layer cosine similarity on GPU.
"""

import logging
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

logger = logging.getLogger(__name__)

from .model_store import download_model, parse_repo

COSINE_THRESHOLD = 0.999

BASELINES = {
    2013: "manelalab/chrono-gpt-v1-20131231@6f2e595689458b1809d5c6efb9a6095257347ca2",
    2014: "manelalab/chrono-gpt-v1-20141231@4fba07f4ef563b3addf2b05f385d0b347bf1cc0d",
    2015: "manelalab/chrono-gpt-v1-20151231@aacd4c4e8020dd0ad686d36f18bcf34cd8003bc3",
    2016: "manelalab/chrono-gpt-v1-20161231@20d93dc9b103644b212db413db4ab1207063d010",
    2017: "manelalab/chrono-gpt-v1-20171231@4cc4334a2c2d38ae35deb0bb7fcae642d3f73a10",
    2018: "manelalab/chrono-gpt-v1-20181231@17d7de7945199ff03be989ca84d00c0f59a975af",
    2019: "manelalab/chrono-gpt-v1-20191231@7e62517f31b11fad179c79ce79a465aa00c7ee4d",
    2020: "manelalab/chrono-gpt-v1-20201231@c0d2acbd2a378ac79d8a5ae79a9447d23145eb8a",
    2021: "manelalab/chrono-gpt-v1-20211231@a070953708ee809e630e4d9652e9c753d7b6782e",
    2022: "manelalab/chrono-gpt-v1-20221231@993711fdf078740fe1c837a3687528e2173443d2",
    2023: "manelalab/chrono-gpt-v1-20231231@8ac22e54d37df8bb8037622680414118239fbe53",
    2024: "manelalab/chrono-gpt-v1-20241231@1d9f1b8ff50bb45a6fe1402280e617af4c2d805c",
}

_baselines = {}


def load_baselines(all_years, device, cache_dir="/tmp/sn38_baselines"):
    """Load all baseline state dicts into device memory. Call once at stage 1 start."""
    os.makedirs(cache_dir, exist_ok=True)
    for year in all_years:
        repo_str = BASELINES.get(year)
        if not repo_str:
            continue
        repo_id, revision = parse_repo(repo_str)
        path = download_model(repo_id, os.path.join(cache_dir, f"baseline_{year}"), revision=revision)
        sf = os.path.join(path, "model.safetensors")
        if os.path.exists(sf):
            with safe_open(sf, framework="pt") as f:
                state = {k: f.get_tensor(k).to(device) for k in f.keys()}
        else:
            state = {k: v.to(device) for k, v in torch.load(
                os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True
            ).items()}
        _baselines[year] = state
        logger.info(f"Baseline {year} loaded to {device}")
    logger.info(f"All {len(_baselines)} baselines loaded")


def unload_baselines():
    """Free baseline state dicts from memory. Call after stage 1."""
    _baselines.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_cosine_gate(candidate_state, year):
    """Check if a candidate model passes the cosine similarity gate.

    Args:
        candidate_state: model.state_dict() from the already-loaded candidate model
        year: which baseline year to compare against

    Returns (passed, avg_cosine).
    """
    baseline_state = _baselines.get(year)
    if baseline_state is None:
        return True, 0.0

    common = sorted(set(baseline_state.keys()) & set(candidate_state.keys()))
    if not common:
        return False, 1.0

    groups = {}
    for k in common:
        a = baseline_state[k]
        b = candidate_state[k]
        if a.shape != b.shape:
            continue
        size = a.numel()
        if size not in groups:
            groups[size] = {"base": [], "cand": []}
        groups[size]["base"].append(a.flatten().float())
        groups[size]["cand"].append(b.flatten().float())

    sims = []
    for size, group in groups.items():
        base_batch = torch.stack(group["base"])
        cand_batch = torch.stack(group["cand"])
        batch_sims = F.cosine_similarity(base_batch, cand_batch, dim=1)
        sims.extend(batch_sims.tolist())

    if not sims:
        return False, 1.0

    avg = sum(sims) / len(sims)
    return avg < COSINE_THRESHOLD, avg
