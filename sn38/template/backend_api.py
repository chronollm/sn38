"""Backend API client for the validator."""

import logging

from .tee import ValidatorSession

logger = logging.getLogger(__name__)


class BackendAPI:
    def __init__(self, backend_url: str, hotkey: str = "unknown"):
        self.session = ValidatorSession(backend_url, hotkey=hotkey)
        logger.info(f"Backend URL: {backend_url}")
        logger.info(f"TEE status: {self.session.is_tee}")
        logger.info(f"TLS verify: {self.session.session.verify}")

    def get_config(self):
        return self.session.get("/config").json()

    def get_years(self, round_num=None):
        params = {"round_num": round_num} if round_num is not None else {}
        return self.session.get("/years", params=params).json()["years"]

    def get_eval_round(self):
        return self.session.get("/rounds/current").json()["eval_round"]

    def get_submission_round(self):
        return self.session.get("/rounds/current").json()["current_round"]

    def get_benchmark(self, cutoff_year, known=False):
        resp = self.session.get(f"/benchmark/{cutoff_year}", params={"known": known})
        if resp.status_code != 200:
            logger.warning(f"Backend /benchmark/{cutoff_year}?known={known} returned {resp.status_code}, closing connection and retrying...")
            resp.close()
            self.session.session.close()
            resp = self.session.get(f"/benchmark/{cutoff_year}", params={"known": known})
            if resp.status_code != 200:
                raise RuntimeError(f"Backend /benchmark/{cutoff_year}?known={known} returned {resp.status_code}")
        return resp.json()

    def get_submissions(self, round_num):
        resp = self.session.get(f"/submissions/{round_num}")
        if resp.status_code != 200:
            logger.error(f"Backend /submissions/{round_num} returned {resp.status_code}")
            return {}, {}
        data = resp.json()
        models = {int(uid): sub["models"] for uid, sub in data.get("submissions", {}).items()}
        timestamps = {int(uid): sub["snapshot_at"] for uid, sub in data.get("submissions", {}).items()}
        return models, timestamps

    def check_hash(self, weight_hash, uid, snapshot_at):
        resp = self.session.post("/models/check-hash", json_data={
            "weight_hash": weight_hash,
            "uid": uid,
            "snapshot_at": snapshot_at,
        })
        if resp.status_code == 401:
            raise RuntimeError("Backend rejected request (401 Unauthorized). Check TEE authentication.")
        if resp.status_code != 200:
            logger.error(f"check_hash returned {resp.status_code}: {resp.text[:200]}")
            return {"allowed": True}
        return resp.json()

    def submit_eval_results(self, round_num, results):
        resp = self.session.post("/eval/results", json_data={
            "round": round_num,
            "results": results,
        })
        if resp.status_code != 200:
            logger.error(f"Failed to submit eval results: {resp.status_code}")
        else:
            logger.info(f"Eval results submitted for round {round_num}")

    def submit_eval_detail(self, round_num, uid, year, repo_id, passed, score, score_unknown, score_known):
        resp = self.session.post("/eval/detail", json_data={
            "round": round_num,
            "uid": uid,
            "year": year,
            "repo_id": repo_id,
            "passed": passed,
            "score": score,
            "score_unknown": score_unknown,
            "score_known": score_known,
        })
        return resp.status_code == 200

    def get_selftest_benchmark(self, cutoff_year, known=False):
        resp = self.session.get(f"/benchmark-selftest/{cutoff_year}", params={"known": known})
        if resp.status_code == 503:
            raise RuntimeError("Self-test dataset not available")
        if resp.status_code != 200:
            raise RuntimeError(f"Backend /benchmark-selftest/{cutoff_year} returned {resp.status_code}")
        return resp.json()

    def check_leak_test_rate_limit(self, hotkey, repo):
        """Check if miner can run a leak test. Raises on rate limit or error."""
        resp = self.session.post("/self-test", json_data={"hotkey": hotkey, "repo": repo})
        if resp.status_code == 429:
            raise RuntimeError(resp.json().get("detail", "Rate limited"))
        if resp.status_code != 200:
            raise RuntimeError(resp.json().get("detail", f"Backend error: {resp.status_code}"))

    def get_quality_questions(self):
        resp = self.session.get("/quality/questions")
        if resp.status_code != 200:
            logger.error(f"Backend /quality/questions returned {resp.status_code}")
            return []
        return resp.json().get("questions", [])
