#!/usr/bin/env python3
"""
Compute the compose-hash for the chrono llm subnet CVMs on Phala Cloud.

Usage:
    python scripts/compute-compose-hash.py                    # validator (default)
    python scripts/compute-compose-hash.py --target self-test # self-test
    python scripts/compute-compose-hash.py --verify
"""

import argparse
import json
import os
import subprocess

from dstack_sdk import get_compose_hash

SCRIPT_DIR = os.path.dirname(__file__)
PRELAUNCH_PATH = os.path.join(SCRIPT_DIR, "prelaunch.sh")

TARGETS = {
    "validator": {
        "compose": os.path.join(SCRIPT_DIR, "..", "docker-compose.validator.yml"),
        "allowed_envs": ["HOTKEY_FILE_CONTENT", "OPENAI_API_KEY", "HF_TOKEN"],
    },
    "self-test": {
        "compose": os.path.join(SCRIPT_DIR, "..", "docker-compose.self-test.yml"),
        "allowed_envs": ["HOTKEY_FILE_CONTENT", "HF_TOKEN", "HF_REPO"],
    },
}

PHALA_DEFAULTS = {
    "runner": "docker-compose",
    "manifest_version": 2,
    "name": "",
    "kms_enabled": True,
    "local_key_provider_enabled": False,
    "no_instance_id": False,
    "public_logs": True,
    "public_sysinfo": True,
    "public_tcbinfo": True,
    "gateway_enabled": True,
    "tproxy_enabled": True,
    "features": ["kms", "tproxy-net"],
    "secure_time": False,
    "storage_fs": "zfs",
}


def main():
    parser = argparse.ArgumentParser(description="Compute Phala CVM compose-hash")
    parser.add_argument("--target", choices=list(TARGETS.keys()), default="validator")
    parser.add_argument("--verify", action="store_true", help="Verify against live CVM")
    args = parser.parse_args()

    target = TARGETS[args.target]
    with open(target["compose"]) as f:
        docker_compose = f.read()

    with open(PRELAUNCH_PATH) as f:
        pre_launch_script = f.read()

    app_compose = dict(PHALA_DEFAULTS)
    app_compose["allowed_envs"] = target["allowed_envs"]
    app_compose["docker_compose_file"] = docker_compose
    app_compose["pre_launch_script"] = pre_launch_script

    print(f"Target: {args.target}")

    h = get_compose_hash(app_compose)
    print(f"compose-hash: {h}")
    print(f"ALLOWED_COMPOSE_HASHES={h}")

    if args.verify:
        result = subprocess.run(["phala", "cvms", "get", "--json"],
                                capture_output=True, text=True)
        live = json.loads(result.stdout)
        print(f"Live CVM hash: {live['compose_hash']}")
        print(f"Match: {h == live['compose_hash']}")


if __name__ == "__main__":
    main()
