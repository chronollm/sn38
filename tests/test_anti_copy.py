"""
Tests for anti-copy protections.

Usage:
    python -m pytest tests/test_anti_copy.py -v
"""

import hashlib
import os
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test")

import pytest
from sn38.template.model_store import validate_models_json
from sn38.neurons.validator import check_duplicate_weights, run_stage1

VALID_SHA = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"


# ─── Fixtures ───

@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"fake model weights")
    return str(tmp_path)

@pytest.fixture
def api_allowed():
    api = MagicMock()
    api.check_hash.return_value = {"allowed": True}
    return api

@pytest.fixture
def patched_stage1(monkeypatch, tmp_path):
    """Patch all heavy dependencies for run_stage1. Returns tmp_path for model dirs."""
    for uid in [1, 2]:
        d = tmp_path / str(uid)
        d.mkdir(exist_ok=True)
        (d / "model.safetensors").write_bytes(f"weights_{uid}".encode())

    def fake_download(repo_id, local_dir, revision=None):
        uid = int(repo_id.split("-")[1])
        return str(tmp_path / str(uid))

    eval_calls = iter([(False, -8.0), (True, -3.0)] * 100)

    monkeypatch.setattr("sn38.neurons.validator.download_model", fake_download)
    monkeypatch.setattr("sn38.neurons.validator.load_model", lambda p, d: MagicMock())
    monkeypatch.setattr("sn38.neurons.validator.count_model_params", lambda m: 1_000_000)
    monkeypatch.setattr("sn38.neurons.validator.get_repo_file_size", lambda r, v: 1000)
    monkeypatch.setattr("sn38.neurons.validator.get_device", lambda: "cpu")
    monkeypatch.setattr("sn38.neurons.validator.evaluate", lambda m, d, b: next(eval_calls))
    monkeypatch.setattr("sn38.neurons.validator.get_cached_result", lambda c, u, y, r: None)
    monkeypatch.setattr("sn38.neurons.validator.save_result", lambda *a: None)
    monkeypatch.setattr("sn38.neurons.validator.verify_commit_sha", lambda r, v: True)


# ─── SHA validation ───

def test_valid_sha_accepted():
    missing = validate_models_json({"2013": f"user/model@{VALID_SHA}"})
    assert isinstance(missing, list)

@pytest.mark.parametrize("repo", ["user/model@main", "user/model", "user/model@abc123"])
def test_invalid_repo_rejected(repo):
    with pytest.raises(ValueError):
        validate_models_json({"2013": repo})


# ─── Weight hash check ───

def test_first_submission_allowed(api_allowed, model_dir):
    assert check_duplicate_weights(api_allowed, model_dir, uid=1, snapshot_at="2026-07-01") is True

def test_duplicate_rejected(model_dir):
    api = MagicMock()
    api.check_hash.return_value = {"allowed": False, "owner_uid": 1}
    assert check_duplicate_weights(api, model_dir, uid=2, snapshot_at="2026-07-01") is False

def test_hash_is_sha256(api_allowed, tmp_path):
    content = b"test weights"
    (tmp_path / "model.safetensors").write_bytes(content)
    check_duplicate_weights(api_allowed, str(tmp_path), uid=1, snapshot_at="2026-07-01")
    actual_hash = api_allowed.check_hash.call_args[0][0]
    assert actual_hash == hashlib.sha256(content).hexdigest()

def test_falls_back_to_pytorch_bin(api_allowed, tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"pytorch weights")
    assert check_duplicate_weights(api_allowed, str(tmp_path), uid=1, snapshot_at="2026-07-01") is True


# ─── Stage 1 integration ───

def test_stage1_evaluates_oldest_first(patched_stage1, monkeypatch):
    """Oldest submitter is evaluated first and claims the weight hash."""
    eval_order = []

    def fake_check(api, path, uid, snapshot_at):
        eval_order.append(uid)
        return True

    monkeypatch.setattr("sn38.neurons.validator.check_duplicate_weights", fake_check)

    submissions = {
        1: {"2013": f"user/m-1@{VALID_SHA}"},
        2: {"2013": f"user/m-2@{VALID_SHA}"},
    }
    submission_times = {1: "2026-07-01T12:00:00", 2: "2026-07-01T08:00:00"}
    config = {"max_model_bytes": 10**10, "max_parameters": 10**10, "max_eval_seconds": 999}

    run_stage1(MagicMock(), submissions, submission_times, config, [2013], conn=MagicMock())

    assert eval_order == [2, 1]


def test_stage1_duplicate_gets_worst_score(patched_stage1, monkeypatch):
    """A miner with duplicate weights gets WORST_SCORE."""
    call_count = [0]

    def fake_check(api, path, uid, snapshot_at):
        call_count[0] += 1
        return call_count[0] == 1

    monkeypatch.setattr("sn38.neurons.validator.check_duplicate_weights", fake_check)

    submissions = {
        1: {"2013": f"user/m-1@{VALID_SHA}"},
        2: {"2013": f"user/m-2@{VALID_SHA}"},
    }
    submission_times = {1: "2026-07-01T08:00:00", 2: "2026-07-01T10:00:00"}
    config = {"max_model_bytes": 10**10, "max_parameters": 10**10, "max_eval_seconds": 999}

    scores = run_stage1(MagicMock(), submissions, submission_times, config, [2013], conn=MagicMock())

    assert scores[1] < 0
    assert scores[2] == 0.0
