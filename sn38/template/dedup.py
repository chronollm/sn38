"""Dedup orchestration: compare miners against each other within the current round."""

import logging
import os
import shutil
import time

import torch
from safetensors.torch import save_file, load_file

from .svd_gate import svd_spectra, _compare_spectra, _check_weight_cosine, SVD_THRESHOLD, COSINE_THRESHOLD

logger = logging.getLogger(__name__)

CANDIDATES_DIR = os.path.join(os.environ.get("DATA_DIR", "/app/data"), "round_candidates")


def save_model_weights(state_dict, uid, weights_dir=CANDIDATES_DIR):
    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, f"uid_{uid}.safetensors")
    cpu_state = {k: v.cpu() for k, v in state_dict.items()}
    save_file(cpu_state, path)
    spectra = svd_spectra(state_dict)
    torch.save(spectra, os.path.join(weights_dir, f"uid_{uid}.spectra.pt"))
    return path


def _compare_against_dir(state_dict, candidate_spectra, device, directory,
                         svd_threshold=SVD_THRESHOLD, cosine_threshold=COSINE_THRESHOLD):
    if not os.path.exists(directory):
        return True, None, ""

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".safetensors"):
            continue
        saved_uid = int(filename.replace("uid_", "").replace(".safetensors", ""))

        t0 = time.time()
        saved_state = load_file(os.path.join(directory, filename), device=str(device))
        t_load = time.time() - t0

        t0 = time.time()
        cosine_passed, top_cosine = _check_weight_cosine(state_dict, saved_state, threshold=cosine_threshold)
        t_cos = time.time() - t0

        if not cosine_passed:
            del saved_state
            logger.info(f"vs UID {saved_uid}: load={t_load:.1f}s cosine={t_cos:.1f}s | cosine={top_cosine:.6f} DUPLICATE")
            return False, saved_uid, f"cosine={top_cosine:.6f}"

        t0 = time.time()
        spectra_cache = os.path.join(directory, filename.replace(".safetensors", ".spectra.pt"))
        if os.path.exists(spectra_cache):
            saved_spectra = torch.load(spectra_cache, map_location=str(device), weights_only=True)
        else:
            saved_spectra = svd_spectra(saved_state)
        svd_dist = _compare_spectra(candidate_spectra, saved_spectra)
        del saved_spectra, saved_state
        t_svd = time.time() - t0

        logger.info(f"vs UID {saved_uid}: load={t_load:.1f}s cosine={t_cos:.1f}s svd={t_svd:.1f}s | cosine={top_cosine:.6f} svd={svd_dist if svd_dist is None else f'{svd_dist:.6f}'}")

        if svd_dist is not None and svd_dist < svd_threshold:
            return False, saved_uid, f"svd={svd_dist:.6f}"

    return True, None, ""


def check_against_saved(state_dict, svd_threshold=SVD_THRESHOLD, cosine_threshold=COSINE_THRESHOLD):
    """Compare a model against all previously saved candidates in the current round.

    Returns (passed, matched_uid, reason).
    """
    t0 = time.time()
    candidate_spectra = svd_spectra(state_dict)
    device = next(iter(state_dict.values())).device
    logger.info(f"Candidate SVD spectra: {time.time() - t0:.1f}s")

    return _compare_against_dir(
        state_dict, candidate_spectra, device, CANDIDATES_DIR,
        svd_threshold=svd_threshold, cosine_threshold=cosine_threshold
    )


def cleanup():
    if os.path.exists(CANDIDATES_DIR):
        shutil.rmtree(CANDIDATES_DIR)