"""Weight and functional comparison functions for dedup."""

import hashlib

import torch
import torch.nn.functional as F

SVD_THRESHOLD = 0.001
SVD_TOP_RATIO = 0.25
COSINE_THRESHOLD = 0.95
KL_THRESHOLD = 1.0
MIN_SHARED_TOKENS = 1024


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
    """Pad a batch of probes to a rectangle. Returns (input_ids, real lengths).

    The lengths matter: the padding is filler, and reading the logits at the
    padded end would measure the pad token rather than the model.
    """
    if isinstance(data[0], list):
        all_ids = [list(ids) for ids in data]
    else:
        all_ids = [model.encode(text) for text in data]
    max_len = max(len(ids) for ids in all_ids)
    pad_id = model.pad_token_id
    padded = torch.tensor([ids + [pad_id] * (max_len - len(ids)) for ids in all_ids])
    return padded, torch.tensor([len(ids) for ids in all_ids])



# Key under which the vocabulary fingerprint travels with the logits. It is a
# tensor so the saved .pt stays loadable with weights_only=True.
VOCAB_KEY = "__vocab_hash__"


def _vocab_hashes(model):
    """int64 hash of each token's string, indexed by token id, or None."""
    tok = getattr(model, "tokenizer", None)
    get_vocab = getattr(tok, "get_vocab", None)
    if get_vocab is None:
        return None
    vocab = get_vocab()
    if not vocab:
        return None
    out = torch.zeros(max(vocab.values()) + 1, dtype=torch.int64)
    for s, i in vocab.items():
        d = hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest()
        out[i] = int.from_bytes(d, "big", signed=True)
    return out


def _align_columns(ha, hb):
    """Column indices (ia, ib) selecting the tokens the two vocabularies share."""
    pos_b = {}
    for i, h in enumerate(hb.tolist()):
        pos_b.setdefault(h, i)
    ia, ib = [], []
    for i, h in enumerate(ha.tolist()):
        j = pos_b.get(h)
        if j is not None:
            ia.append(i)
            ib.append(j)
    return ia, ib


def _compute_kl_logits(model, device, probes):
    """Compute logits fingerprints from backend probes. Returns dict of logits.

    Also carries the vocabulary fingerprint so a later comparison can line the
    columns up by token string rather than trusting the token ids to agree.
    """
    logits = {}
    with torch.no_grad():
        for key, data in probes.items():
            input_ids, lengths = _to_input_ids(model, data)
            out = model(input_ids.to(device))
            rows = torch.arange(out.shape[0], device=out.device)
            logits[key] = out[rows, (lengths - 1).to(out.device), :].float()
    h = _vocab_hashes(model)
    if h is not None:
        logits[VOCAB_KEY] = h
    return logits


def _compare_kl_logits(logits_dict_a, logits_dict_b, threshold=KL_THRESHOLD):
    """Compare two logit fingerprints via symmetric KL divergence.

    Uses max of medians across probe types.
    Returns (passed, max_median_kl).
    """
    ha = logits_dict_a.get(VOCAB_KEY)
    hb = logits_dict_b.get(VOCAB_KEY)
    if ha is not None:
        ha = ha.cpu()
    if hb is not None:
        hb = hb.cpu()
    cols = None
    if ha is not None and hb is not None and not (
            ha.shape == hb.shape and bool(torch.equal(ha, hb))):
        # different token ids or vocabulary size: reindex both onto the tokens
        # they share, by string. Identical tokenizers skip this entirely.
        ia, ib = _align_columns(ha, hb)
        if len(ia) < MIN_SHARED_TOKENS:
            # too little overlap to judge -- do not hand out a free pass
            return False, 0.0
        cols = (torch.tensor(ia), torch.tensor(ib))

    medians = []
    for name in logits_dict_a:
        if name == VOCAB_KEY or name not in logits_dict_b:
            continue
        la = logits_dict_a[name]
        lb = logits_dict_b[name].to(la.device)

        if cols is not None:
            la = la.index_select(-1, cols[0].to(la.device))
            lb = lb.index_select(-1, cols[1].to(lb.device))
        elif la.shape[-1] != lb.shape[-1]:
            # no fingerprint (checkpoint written before this change): fall back
            # to the shared prefix rather than passing, as the old code did.
            n = min(la.shape[-1], lb.shape[-1])
            la, lb = la[..., :n], lb[..., :n]

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