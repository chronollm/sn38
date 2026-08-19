"""Dedup orchestration: compare miners against each other within the current round."""

import logging
import os
import shutil
import time

import torch
from safetensors.torch import save_file, load_file

from .svd_gate import (
    svd_spectra, _compare_spectra, _check_weight_cosine, _compute_kl_logits, _compare_kl_logits,
    SVD_THRESHOLD, COSINE_THRESHOLD, KL_THRESHOLD,
)

logger = logging.getLogger(__name__)

CANDIDATES_DIR = os.path.join(os.environ.get("DATA_DIR", "/app/data"), "round_candidates")


def save_candidate(state_dict, uid, logits, weights_dir=CANDIDATES_DIR):
    """Save weights, SVD spectra, and KL logits fingerprint."""
    os.makedirs(weights_dir, exist_ok=True)
    cpu_state = {k: v.cpu() for k, v in state_dict.items()}
    save_file(cpu_state, os.path.join(weights_dir, f"uid_{uid}.safetensors"))
    spectra = svd_spectra(state_dict)
    torch.save(spectra, os.path.join(weights_dir, f"uid_{uid}.spectra.pt"))
    torch.save({k: v.cpu() for k, v in logits.items()}, os.path.join(weights_dir, f"uid_{uid}.logits.pt"))


def check_against_saved(model, state_dict, device, probes, svd_threshold=SVD_THRESHOLD,
                        cosine_threshold=COSINE_THRESHOLD, kl_threshold=KL_THRESHOLD):
    """Compare a model against all previously saved candidates.

    Returns (passed, matched_uid, reason, logits).
    """
    candidate_logits = _compute_kl_logits(model, device, probes)

    if not os.path.exists(CANDIDATES_DIR):
        return True, None, "", candidate_logits

    candidate_spectra = svd_spectra(state_dict)

    for filename in sorted(os.listdir(CANDIDATES_DIR)):
        if not filename.endswith(".safetensors"):
            continue
        saved_uid = int(filename.replace("uid_", "").replace(".safetensors", ""))

        t0 = time.time()
        saved_spectra = torch.load(os.path.join(CANDIDATES_DIR, f"uid_{saved_uid}.spectra.pt"), map_location=str(device), weights_only=True)
        svd_dist = _compare_spectra(candidate_spectra, saved_spectra)
        del saved_spectra
        t_svd = time.time() - t0

        # Cosine (from safetensors)
        t0 = time.time()
        saved_state = load_file(os.path.join(CANDIDATES_DIR, filename), device=str(device))
        cosine_passed, avg_cosine = _check_weight_cosine(state_dict, saved_state, threshold=cosine_threshold)
        del saved_state
        t_cos = time.time() - t0

        # KL (from cached logits, no model reload)
        t0 = time.time()
        saved_logits = torch.load(os.path.join(CANDIDATES_DIR, f"uid_{saved_uid}.logits.pt"), map_location=str(device), weights_only=True)
        kl_passed, median_kl = _compare_kl_logits(candidate_logits, saved_logits, threshold=kl_threshold)
        del saved_logits
        t_kl = time.time() - t0

        svd_str = f"{svd_dist:.4f}" if svd_dist is not None else "None"
        logger.info(f"vs UID {saved_uid}: svd={t_svd:.1f}s cos={t_cos:.1f}s kl={t_kl:.1f}s | svd={svd_str} cosine={avg_cosine:.4f} kl={median_kl:.4f}")

        if not kl_passed:
            return False, saved_uid, f"kl={median_kl:.4f} svd={svd_str} cosine={avg_cosine:.4f}", candidate_logits

    return True, None, "", candidate_logits


def cleanup():
    if os.path.exists(CANDIDATES_DIR):
        shutil.rmtree(CANDIDATES_DIR)