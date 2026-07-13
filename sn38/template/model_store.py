"""Model storage layer — commit metadata on-chain, download models from HuggingFace."""

import json
import logging
import multiprocessing as mp
import os
import re
import time
from typing import Optional

import bittensor as bt
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .constants import ALL_YEARS

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

if os.environ.get("HF_DEBUG", "").lower() in ("1", "true"):
    logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)

SHA_PATTERN = re.compile(r"^[^/]+/[^@]+@[0-9a-f]{40}$")


def verify_commit_sha(repo_id: str, revision: str) -> bool:
    """Verify that a revision resolves to a real commit SHA, not a branch with a SHA-like name."""
    info = HfApi().repo_info(repo_id, revision=revision)
    return info.sha == revision


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


def upload_models_json(models: dict, dataset_repo: str, token: Optional[str] = None):
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


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PRIVATE_OR_MISSING = 10
EXIT_GATED = 11
EXIT_BAD_REVISION = 12


def _do_download_model(
    repo_id: str,
    revision: Optional[str],
    local_dir: str,
):
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
    try:
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)
    except GatedRepoError:
        os._exit(EXIT_GATED)
    except RepositoryNotFoundError:
        os._exit(EXIT_PRIVATE_OR_MISSING)
    except RevisionNotFoundError:
        os._exit(EXIT_BAD_REVISION)
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in (401, 403):
            os._exit(EXIT_PRIVATE_OR_MISSING)

        os._exit(EXIT_ERROR)
    except Exception:
        os._exit(EXIT_ERROR)

    # bt.logging is holding a thread hostage which raises an exception on exit...
    os._exit(EXIT_OK)


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass

    return total


def download_model(
    repo_id: str,
    local_dir: str,
    revision: Optional[str] = None,
    stall_timeout: int = 240,
    max_retries: int = 5,
    watchdog_interval: int = 5,
) -> str:
    for _ in range(max_retries):
        ctx = mp.get_context("spawn")
        process = ctx.Process(target=_do_download_model, args=(repo_id, revision, local_dir))
        process.start()

        last_size = -1
        last_progress_time = time.time()

        while process.is_alive():
            process.join(timeout=watchdog_interval)
            if not process.is_alive():
                continue

            size = _dir_size(local_dir)
            if size != last_size:
                last_size = size
                last_progress_time = time.time()

            elif time.time() - last_progress_time > stall_timeout:
                bt.logging.info(f"Stalled ({size} bytes, no growth for {stall_timeout}s), killing and retrying...")
                process.kill()
                process.join()
                break

        else:
            process.join()

            exit_code = process.exitcode
            if exit_code == EXIT_OK:
                return local_dir

            elif exit_code == EXIT_PRIVATE_OR_MISSING:
                raise FileNotFoundError(f"{repo_id} does not exist or is private (no access with current token)")
            elif exit_code == EXIT_GATED:
                raise PermissionError(f"{repo_id} is gated (no access with current token)")
            elif exit_code == EXIT_BAD_REVISION:
                raise ValueError(f"Revision {revision!r} not found for {repo_id}")

            bt.logging.warning(f"Download process exited with code {exit_code}, retrying...")

    raise RuntimeError(f"Download of {repo_id} failed after {max_retries} attempts")


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
