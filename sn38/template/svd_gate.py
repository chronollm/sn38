"""Weight and functional comparison functions for dedup."""

import torch
import torch.nn.functional as F

SVD_THRESHOLD = 0.001
SVD_TOP_RATIO = 0.25
COSINE_THRESHOLD = 0.95
KL_THRESHOLD = 1.0


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


def _compare_spectra(spectra_a, spectra_b, threshold=SVD_THRESHOLD):
    """Compare two models' spectra. Returns (passed, avg_distance)."""
    common = sorted(set(spectra_b.keys()) & set(spectra_a.keys()))
    if not common:
        return True, float("inf")

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
        return True, float("inf")
    avg = sum(distances) / len(distances)
    return avg >= threshold, avg


def _check_weight_cosine(state_a, state_b, threshold=COSINE_THRESHOLD):
    """Check weight cosine similarity (average of all layers).

    Returns (passed, avg_cosine).
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
    avg = sum(cos_sims) / len(cos_sims)
    return avg < threshold, avg


def _to_input_ids(model, data):
    if isinstance(data[0], list):
        return torch.tensor(data)
    all_ids = []
    for text in data:
        all_ids.append(model.encode(text))
    max_len = max(len(ids) for ids in all_ids)
    pad_id = model.pad_token_id
    return torch.tensor([ids + [pad_id] * (max_len - len(ids)) for ids in all_ids])


def _compute_kl_logits(model, device, probes):
    """Compute logits fingerprints from backend probes. Returns dict of logits."""
    logits = {}
    with torch.no_grad():
        for key, data in probes.items():
            input_ids = _to_input_ids(model, data).to(device)
            logits[key] = model(input_ids)[:, -1, :].float()
    return logits


def _compare_kl_logits(logits_dict_a, logits_dict_b, threshold=KL_THRESHOLD):
    """Compare two logit fingerprints via symmetric KL divergence.

    Uses max of medians across probe types.
    Returns (passed, max_median_kl).
    """
    medians = []
    for name in logits_dict_a:
        if name not in logits_dict_b:
            continue
        la = logits_dict_a[name]
        lb = logits_dict_b[name].to(la.device)

        if la.shape[-1] != lb.shape[-1]:
            return True, float("inf")

        log_probs_a = F.log_softmax(la, dim=-1)
        log_probs_b = F.log_softmax(lb, dim=-1)
        probs_a = F.softmax(la, dim=-1)
        probs_b = F.softmax(lb, dim=-1)

        kl_ab = F.kl_div(log_probs_a, probs_b, reduction="none").sum(dim=-1)
        kl_ba = F.kl_div(log_probs_b, probs_a, reduction="none").sum(dim=-1)
        symmetric_kl = (kl_ab + kl_ba) / 2

        medians.append(symmetric_kl.median().item())

    if not medians:
        return True, float("inf")

    max_median = max(medians)
    return max_median >= threshold, max_median