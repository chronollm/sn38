"""SVD similarity gate — reject models with spectra too close to the baseline.

Compares singular value spectra of weight matrices. Invariant to
rotation and permutation attacks that bypass cosine similarity.
"""

import logging
import os

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

from .model_store import download_model, parse_repo

SVD_THRESHOLD = 0.01
SVD_TOP_RATIO = 0.25

BASELINES = {
    2013: [
        "manelalab/chrono-gpt-v1-20131231@6f2e595689458b1809d5c6efb9a6095257347ca2",
        "manelalab/chrono-gpt-instruct-v1-20131231@f35f1596d860a797df1c592a5a70bf02a3a00884",
    ],
    2014: [
        "manelalab/chrono-gpt-v1-20141231@4fba07f4ef563b3addf2b05f385d0b347bf1cc0d",
        "manelalab/chrono-gpt-instruct-v1-20141231@e121db790ca77ebb082c025b2438717644ee1cfb",
    ],
    2015: [
        "manelalab/chrono-gpt-v1-20151231@aacd4c4e8020dd0ad686d36f18bcf34cd8003bc3",
        "manelalab/chrono-gpt-instruct-v1-20151231@5a7f3439fd5d782b3780c366160c43177e6f5eba",
    ],
    2016: [
        "manelalab/chrono-gpt-v1-20161231@20d93dc9b103644b212db413db4ab1207063d010",
        "manelalab/chrono-gpt-instruct-v1-20161231@5aec0aacc696f9526e12abe22a3fc96348dfca1d",
    ],
    2017: [
        "manelalab/chrono-gpt-v1-20171231@4cc4334a2c2d38ae35deb0bb7fcae642d3f73a10",
        "manelalab/chrono-gpt-instruct-v1-20171231@5f6b4ab1664bd5e658af44ad6b02183178b81b55",
    ],
    2018: [
        "manelalab/chrono-gpt-v1-20181231@17d7de7945199ff03be989ca84d00c0f59a975af",
        "manelalab/chrono-gpt-instruct-v1-20181231@331c03be137a1a80f1a371232d3d6a9636f6ad9a",
    ],
    2019: [
        "manelalab/chrono-gpt-v1-20191231@7e62517f31b11fad179c79ce79a465aa00c7ee4d",
        "manelalab/chrono-gpt-instruct-v1-20191231@4dfb7817915d07d0ed99815877186f827ec3b88e",
    ],
    2020: [
        "manelalab/chrono-gpt-v1-20201231@c0d2acbd2a378ac79d8a5ae79a9447d23145eb8a",
        "manelalab/chrono-gpt-instruct-v1-20201231@f8020c2c939645abbec9caf8a0cdd1d7806cb42a",
    ],
    2021: [
        "manelalab/chrono-gpt-v1-20211231@a070953708ee809e630e4d9652e9c753d7b6782e",
        "manelalab/chrono-gpt-instruct-v1-20211231@7f3c7d0dccea060d96dfb89391ef830655b8dbaf",
    ],
    2022: [
        "manelalab/chrono-gpt-v1-20221231@993711fdf078740fe1c837a3687528e2173443d2",
        "manelalab/chrono-gpt-instruct-v1-20221231@f1b8c4eb806a9fe7c26b7e5d30cf003304ed9281",
    ],
    2023: [
        "manelalab/chrono-gpt-v1-20231231@8ac22e54d37df8bb8037622680414118239fbe53",
        "manelalab/chrono-gpt-instruct-v1-20231231@2156f3ac9a36916773664266397682b951d43411",
    ],
    2024: [
        "manelalab/chrono-gpt-v1-20241231@1d9f1b8ff50bb45a6fe1402280e617af4c2d805c",
        "manelalab/chrono-gpt-instruct-v1-20241231@c162df20666475d125737e030943e18e10b3d91f",
    ],
}

_baselines = {}


def svd_spectra(state_dict):
    """Extract singular value spectra for all 2D weight matrices."""
    spectra = {}
    for name, param in state_dict.items():
        if param.ndim == 2 and min(param.shape) > 1:
            spectra[name] = torch.linalg.svdvals(param.float())
    return spectra


def _load_state(path, device):
    sf = os.path.join(path, "model.safetensors")
    if os.path.exists(sf):
        with safe_open(sf, framework="pt") as f:
            return {k: f.get_tensor(k).to(device) for k in f.keys()}
    return {k: v.to(device) for k, v in torch.load(
        os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True
    ).items()}


def load_baselines(all_years, device, cache_dir=None):
    """Load all baseline SVD spectra. Call once at stage 1 start."""
    if cache_dir is None:
        cache_dir = os.path.join(os.environ.get("HF_HOME", "/tmp"), "sn38_baselines")
    os.makedirs(cache_dir, exist_ok=True)
    total = 0
    for year in all_years:
        repos = BASELINES.get(year, [])
        _baselines[year] = []
        for i, repo_str in enumerate(repos):
            repo_id, revision = parse_repo(repo_str)
            path = download_model(repo_id, os.path.join(cache_dir, f"baseline_{year}_{i}"), revision=revision)
            state = _load_state(path, device)
            _baselines[year].append(svd_spectra(state))
            del state
            total += 1
            logger.info(f"Baseline {year} variant {i} SVD loaded")
    logger.info(f"All {total} baselines loaded")


def unload_baselines():
    """Free baseline spectra from memory."""
    _baselines.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compare_spectra(candidate_spectra, baseline_spectra):
    """Compare candidate against one baseline. Returns avg distance or None."""
    common = sorted(set(baseline_spectra.keys()) & set(candidate_spectra.keys()))
    if not common:
        return None

    distances = []
    for name in common:
        sv_base = baseline_spectra[name]
        sv_cand = candidate_spectra[name]
        if sv_base.shape != sv_cand.shape:
            continue
        k = max(1, int(len(sv_base) * SVD_TOP_RATIO))
        sv_base_norm = sv_base[:k] / (torch.norm(sv_base[:k]) + 1e-10)
        sv_cand_norm = sv_cand[:k] / (torch.norm(sv_cand[:k]) + 1e-10)
        dist = torch.norm(sv_base_norm - sv_cand_norm)
        distances.append(dist.item())

    if not distances:
        return None
    return sum(distances) / len(distances)


def check_svd_gate(candidate_spectra, year):
    """Check if a candidate model passes the SVD similarity gate.

    Compares against all baselines for the year, uses minimum distance.
    Returns (passed, min_svd_dist).
    """
    baselines = _baselines.get(year, [])
    if not baselines:
        return True, 1.0

    min_dist = None
    for baseline_spectra in baselines:
        dist = _compare_spectra(candidate_spectra, baseline_spectra)
        if dist is not None and (min_dist is None or dist < min_dist):
            min_dist = dist

    if min_dist is None:
        return True, 1.0

    return min_dist >= SVD_THRESHOLD, min_dist


def dedup_by_svd(spectra_dict, submission_times, threshold=SVD_THRESHOLD):
    """Remove duplicate miners by pairwise SVD comparison.

    Sorts by submission time — earliest submitter wins.
    Returns set of accepted UIDs.
    """
    sorted_uids = sorted(spectra_dict.keys(), key=lambda u: submission_times.get(u, "9999"))
    accepted = []

    for uid in sorted_uids:
        is_dup = False
        for accepted_uid in accepted:
            dist = _compare_spectra(spectra_dict[uid], spectra_dict[accepted_uid])
            if dist is not None and dist < threshold:
                logger.warning(f"UID {uid} is a copy of UID {accepted_uid} (svd_dist={dist:.6f}), removing")
                is_dup = True
                break
        if not is_dup:
            accepted.append(uid)

    logger.info(f"SVD dedup: {len(spectra_dict)} miners → {len(accepted)} unique")
    return set(accepted)
