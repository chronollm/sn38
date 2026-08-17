"""Weight comparison functions for dedup.

1. SVD spectral distance: catches weight copies and scaling attacks.
2. Weight cosine similarity (top percentile): catches rotation/reshaping
   attacks where some layers must remain unchanged to preserve function.
"""

import torch
import torch.nn.functional as F

SVD_THRESHOLD = 0.01
SVD_TOP_RATIO = 0.25
COSINE_THRESHOLD = 0.999
COSINE_PERCENTILE = 0.25


def svd_spectra(state_dict):
    """Extract singular value spectra for all 2D weight matrices (batched GPU)."""
    groups = {}
    for name, param in state_dict.items():
        if param.ndim == 2 and min(param.shape) > 1:
            shape = param.shape
            if shape not in groups:
                groups[shape] = []
            groups[shape].append((name, param.float()))

    spectra = {}
    with torch.no_grad():
        for shape, items in groups.items():
            batch = torch.stack([p for _, p in items])
            svs = torch.linalg.svdvals(batch)
            for i, (name, _) in enumerate(items):
                spectra[name] = svs[i]
    return spectra


def _compare_spectra(spectra_a, spectra_b):
    """Compare two models' spectra. Returns avg distance or None."""
    common = sorted(set(spectra_b.keys()) & set(spectra_a.keys()))
    if not common:
        return None

    distances = []
    for name in common:
        sv_a = spectra_a[name]
        sv_b = spectra_b[name].to(sv_a.device)
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


def _check_weight_cosine(state_a, state_b, threshold=COSINE_THRESHOLD, percentile=COSINE_PERCENTILE):
    """Check if two models share too many similar weight matrices.

    Batched GPU cosine similarity, then checks if the top percentile
    exceeds the threshold. Architecture-agnostic.
    Returns (passed, top_cosine).
    """
    common = sorted(set(state_a) & set(state_b))
    groups = {}
    with torch.no_grad():
        for key in common:
            a, b = state_a[key], state_b[key]
            if a.shape == b.shape and a.numel() > 1:
                size = a.numel()
                if size not in groups:
                    groups[size] = {"a": [], "b": []}
                groups[size]["a"].append(a.flatten().float())
                groups[size]["b"].append(b.to(a.device).flatten().float())

        cos_sims = []
        for group in groups.values():
            batch_a = torch.stack(group["a"])
            batch_b = torch.stack(group["b"])
            sims = F.cosine_similarity(batch_a, batch_b, dim=1)
            cos_sims.extend(sims.tolist())

    if not cos_sims:
        return True, 0.0
    cos_sims.sort(reverse=True)
    top = cos_sims[:max(1, int(len(cos_sims) * percentile))]
    avg_top = sum(top) / len(top)
    return avg_top < threshold, avg_top