from pathlib import Path
from unittest.mock import patch

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


def _make_fake_process(exit_code):
    """Create a fake process that returns the given exit code."""
    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.exitcode = exit_code
        def start(self): pass
        def join(self, timeout=None): pass
        def is_alive(self): return False
        def kill(self): pass
    return FakeProcess


def test_escalation_on_repeated_failures(tmp_path: Path):
    """Verify escalation: wipe partial state, wipe xet cache, then disable xet."""
    fake_ctx = type("FakeCtx", (), {"Process": _make_fake_process(exit_code=1)})()
    local_dir = str(tmp_path / "model")

    with patch("sn38.template.model_store._wipe_partial_state") as mock_wipe_partial, \
         patch("sn38.template.model_store._wipe_xet_cache") as mock_wipe_xet, \
         patch("sn38.template.model_store.mp") as mock_mp:
        mock_mp.get_context.return_value = fake_ctx

        with pytest.raises(RuntimeError, match="failed after"):
            download_model(
                repo_id="fake/repo",
                local_dir=local_dir,
                max_retries=5,
            )

        # Partial state wiped on attempts 1 and 2
        assert mock_wipe_partial.call_count == 2
        # Xet cache wiped once on attempt 2
        mock_wipe_xet.assert_called_once()
