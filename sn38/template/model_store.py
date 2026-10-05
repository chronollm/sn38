"""Model storage layer — commit metadata on-chain, download models from HuggingFace."""

import json
import logging
import multiprocessing as mp
import os
import re
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

from .constants import ALL_YEARS

if os.environ.get("HF_DEBUG", "").lower() in ("1", "true"):
    logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)

SHA_PATTERN = re.compile(r"^[^/]+/[^@]+@[0-9a-f]{40}$")


def verify_commit_sha(repo_id: str, revision: str) -> bool:
    """Verify that a revision resolves to a real commit SHA, not a branch with a SHA-like name."""
    from huggingface_hub import HfApi
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
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(dataset_repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=json.dumps(models, indent=2).encode(),
        path_in_repo="models.json",
        repo_id=dataset_repo,
        repo_type="dataset",
    )
    logger.info(f"Uploaded models.json to {dataset_repo}")


def fetch_models_json(dataset_repo: str) -> dict:
    """Fetch models.json from a HuggingFace dataset repo."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=dataset_repo, filename="models.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def commit_metadata(subtensor, wallet, netuid: int, data: str):
    """Commit model metadata on-chain."""
    subtensor.set_commitment(wallet=wallet, netuid=netuid, data=data)
    logger.info(f"Committed on-chain: {data[:80]}...")


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PRIVATE_OR_MISSING = 10
EXIT_GATED = 11
EXIT_BAD_REVISION = 12


def _do_download_model(
    repo_id: str,
    revision: Optional[str],
    local_dir: str,
    disable_xet: bool = False,
):
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import snapshot_download
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
        import traceback
        traceback.print_exc()
        os._exit(EXIT_ERROR)

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


def _wipe_partial_state(local_dir: str):
    """Remove resumable partial-download state so the next attempt starts fresh."""
    import shutil
    partial = os.path.join(local_dir, ".cache")
    if os.path.isdir(partial):
        shutil.rmtree(partial, ignore_errors=True)
        logger.info(f"Wiped partial download state in {partial}")


def _wipe_xet_cache():
    """Remove the global xet chunk cache."""
    import shutil
    xet_cache = os.environ.get(
        "HF_XET_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "xet"),
    )
    if os.path.isdir(xet_cache):
        shutil.rmtree(xet_cache, ignore_errors=True)
        logger.info(f"Wiped xet chunk cache at {xet_cache}")


def retrying_download_model(
    repo_id: str,
    local_dir: str,
    revision: Optional[str] = None,
    stall_timeout: int = 240,
    max_retries: int = 5,
    watchdog_interval: int = 5,
) -> str:
    for attempt in range(max_retries):
        disable_xet = False

        if attempt == 1:
            _wipe_partial_state(local_dir)
        if attempt == 2:
            _wipe_partial_state(local_dir)
            _wipe_xet_cache()
        if attempt >= 3:
            disable_xet = True
            logger.info(f"Attempt {attempt}: falling back to non-xet download for {repo_id}")

        ctx = mp.get_context("spawn")
        process = ctx.Process(target=_do_download_model, args=(repo_id, revision, local_dir, disable_xet))
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
                logger.info(f"Stalled ({size} bytes, no growth for {stall_timeout}s), terminating...")
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
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

            logger.warning(f"Download process exited with code {exit_code}, retrying (attempt {attempt + 1}/{max_retries})...")

    raise RuntimeError(f"Download of {repo_id} failed after {max_retries} attempts")


def direct_download_model(repo_id: str, local_dir: Optional[str] = None, revision: Optional[str] = None) -> str:
    """Download a model snapshot and return its path. The hub handles retries and resume.

    Swap the body for retrying_download_model(...) if the hub stalls come back.
    """
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)


_HF_BASE = os.environ.get("HF_HOME", "/app/data/hf")


def set_round_cache(eval_round):
    """Point the HF caches at this round's dir and drop every other round.

    Must run before the first model_store call: huggingface_hub derives every path
    from HF_HOME at import, and that import is deferred until then.
    """
    import shutil
    current = f"round{eval_round}"
    removed = []
    for name in (os.listdir(_HF_BASE) if os.path.isdir(_HF_BASE) else []):
        if name != current:
            shutil.rmtree(os.path.join(_HF_BASE, name), ignore_errors=True)
            removed.append(name)
    home = os.path.join(_HF_BASE, current)
    os.environ["HF_HOME"] = home
    if removed:
        logger.info(f"Cleared old cache: {', '.join(sorted(removed))}")
    logger.info(f"HF cache: {home}")
    for root, dirs, files in os.walk(home):
        for name in sorted(dirs + files):
            logger.debug(f"  {os.path.relpath(os.path.join(root, name), home)}")


def delete_models(revisions):
    """Delete cached model revisions by commit sha."""
    from huggingface_hub import scan_cache_dir
    if revisions:
        scan_cache_dir().delete_revisions(*revisions).execute()


def parse_repo(repo_str):
    """Parse 'owner/repo@revision' → (repo_id, revision). No @ means main."""
    if "@" in repo_str:
        repo_id, revision = repo_str.rsplit("@", 1)
        return repo_id, revision
    return repo_str, None


def get_repo_file_size(repo_id, revision=None):
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        return sum(s.size for s in (info.siblings or []) if s.rfilename.endswith((".safetensors", ".bin")))
    except Exception:
        return 0


def count_model_params(model):
    return sum(p.numel() for p in model.parameters())


def get_device():
    logger.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    logger.info(f"torch.version.cuda={torch.version.cuda}")
    logger.info(f"torch.backends.cudnn.enabled={torch.backends.cudnn.enabled}")
    if hasattr(torch.cuda, "device_count"):
        logger.info(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning("No GPU detected, falling back to CPU")
    return torch.device("cpu")
