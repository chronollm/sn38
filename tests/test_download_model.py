from pathlib import Path

import pytest
from sn38.template.model_store import download_model

# Small public repo, always accessible, safe to hit repeatedly
VALID_REPO = "hf-internal-testing/tiny-random-bert"
GATED_REPO = "meta-llama/Llama-2-7b-hf"
NONEXISTENT_REPO = "this-org-does-not-exist-12345/no-such-model-98765"


def test_gated_repo_raises_permission_error(tmp_path: Path):
    with pytest.raises(PermissionError, match="gated"):
        download_model(
            repo_id=GATED_REPO,
            local_dir=str(tmp_path / "model"),
        )


def test_nonexistent_repo_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist or is private"):
        download_model(
            repo_id=NONEXISTENT_REPO,
            local_dir=str(tmp_path / "model"),
        )


def test_bad_revision_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError, match="revision"):
        download_model(
            repo_id=VALID_REPO,
            local_dir=str(tmp_path / "model"),
            revision="this-revision-does-not-exist",
        )


def test_download_model_real(tmp_path: Path):
    local_dir = tmp_path / "model"
    result = download_model(
        repo_id=VALID_REPO,
        local_dir=str(local_dir),
    )

    assert result == str(local_dir)
    files = list(local_dir.rglob("*"))
    assert any(f.is_file() for f in files)
