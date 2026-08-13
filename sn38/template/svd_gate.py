"""SVD spectral analysis — pairwise dedup between miners.

Compares singular value spectra of weight matrices. Invariant to
rotation and permutation attacks that bypass cosine similarity.
"""

import logging

import torch

logger = logging.getLogger(__name__)

SVD_THRESHOLD = 0.01
SVD_TOP_RATIO = 0.25


def svd_spectra(state_dict):
    """Extract singular value spectra for all 2D weight matrices."""
    spectra = {}
    for name, param in state_dict.items():
        if param.ndim == 2 and min(param.shape) > 1:
            spectra[name] = torch.linalg.svdvals(param.float())
    return spectra


def _compare_spectra(spectra_a, spectra_b):
    """Compare two models' spectra. Returns avg distance or None."""
    common = sorted(set(spectra_b.keys()) & set(spectra_a.keys()))
    if not common:
        return None

    distances = []
    for name in common:
        sv_a = spectra_a[name]
        sv_b = spectra_b[name]
        if sv_a.shape != sv_b.shape:
            continue
        k = max(1, int(len(sv_a) * SVD_TOP_RATIO))
        sv_a_norm = sv_a[:k] / (torch.norm(sv_a[:k]) + 1e-10)
        sv_b_norm = sv_b[:k] / (torch.norm(sv_b[:k]) + 1e-10)
        dist = torch.norm(sv_a_norm - sv_b_norm)
        distances.append(dist.item())

    if not distances:
        return None
    return sum(distances) / len(distances)


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
