"""Model storage layer — commit metadata on-chain, download models from HuggingFace."""

import json
import re
import torch
import bittensor as bt
from huggingface_hub import snapshot_download, hf_hub_download, HfApi

from .constants import ALL_YEARS

SHA_PATTERN = re.compile(r"^[^/]+/[^@]+@[0-9a-f]{40}$")


def validate_models_json(models: dict) -> list[int]:
    """Validate the models JSON structure. Raises ValueError on invalid input.
    Returns list of missing years (for warnings)."""
    if not isinstance(models, dict):
        raise ValueError("models must be a dict")
    for year_str, repo_str in models.items():
        year = int(year_str)
        if year not in ALL_YEARS:
            raise ValueError(f"Year {year} not in {ALL_YEARS[0]}-{ALL_YEARS[-1]}")
        if not isinstance(repo_str, str) or not SHA_PATTERN.match(repo_str):
            raise ValueError(f"Invalid repo format: {repo_str} (expected owner/repo@<40-char commit SHA>)")
    return [y for y in ALL_YEARS if str(y) not in models]


def upload_models_json(models: dict, dataset_repo: str, token: str = None):
    """Upload models.json to a HuggingFace dataset repo."""
    api = HfApi(token=token)
    api.create_repo(dataset_repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=json.dumps(models, indent=2).encode(),
        path_in_repo="models.json",
        repo_id=dataset_repo,
        repo_type="dataset",
    )
    bt.logging.info(f"Uploaded models.json to {dataset_repo}")


def fetch_models_json(dataset_repo: str) -> dict:
    """Fetch models.json from a HuggingFace dataset repo."""
    path = hf_hub_download(repo_id=dataset_repo, filename="models.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def commit_metadata(subtensor, wallet, netuid: int, data: str):
    """Commit model metadata on-chain."""
    subtensor.set_commitment(wallet=wallet, netuid=netuid, data=data)
    bt.logging.info(f"Committed on-chain: {data[:80]}...")


def download_model(repo_id: str, local_dir: str, revision: str = None) -> str:
    """Download a model from HuggingFace. Returns the local path."""
    return snapshot_download(repo_id=repo_id, local_dir=local_dir, revision=revision)


def parse_repo(repo_str):
    """Parse 'owner/repo@revision' → (repo_id, revision). No @ means main."""
    if "@" in repo_str:
        repo_id, revision = repo_str.rsplit("@", 1)
        return repo_id, revision
    return repo_str, None


def get_repo_file_size(repo_id, revision=None):
    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        return sum(s.size for s in (info.siblings or []) if s.rfilename.endswith((".safetensors", ".bin")))
    except Exception:
        return 0


def count_model_params(model):
    return sum(p.numel() for p in model.parameters())


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
