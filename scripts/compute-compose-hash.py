#!/usr/bin/env python3
"""
Compute the compose-hash for the chrono llm subnet validator CVM on Phala Cloud.

Reproduces the exact app-compose structure that Phala Cloud generates,
using the pinned pre_launch_script (v0.0.15) from scripts/prelaunch.sh.

Deploy with:
    phala deploy -c docker-compose.validator.yml \
        --pre-launch-script scripts/prelaunch.sh \
        --env OPENAI_API_KEY=sk-xxx \
        --env HOTKEY_FILE_CONTENT="$(cat ~/.bittensor/wallets/testnet-sn38-validator1/hotkeys/default)"

Usage:
    python scripts/compute-compose-hash.py
    python scripts/compute-compose-hash.py --verify
"""

import argparse
import json
import os
import subprocess

from dstack_sdk import get_compose_hash

SCRIPT_DIR = os.path.dirname(__file__)
COMPOSE_PATH = os.path.join(SCRIPT_DIR, "..", "docker-compose.validator.yml")
PRELAUNCH_PATH = os.path.join(SCRIPT_DIR, "prelaunch.sh")

# Phala Cloud app-compose properties
PHALA_DEFAULTS = {
    "runner": "docker-compose",
    "manifest_version": 2,
    "name": "",
    "allowed_envs": ["HOTKEY_FILE_CONTENT", "OPENAI_API_KEY"],
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
    parser.add_argument("--verify", action="store_true", help="Verify against live CVM")
    args = parser.parse_args()

    with open(COMPOSE_PATH) as f:
        docker_compose = f.read()

    with open(PRELAUNCH_PATH) as f:
        pre_launch_script = f.read()

    app_compose = dict(PHALA_DEFAULTS)
    app_compose["docker_compose_file"] = docker_compose
    app_compose["pre_launch_script"] = pre_launch_script

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
