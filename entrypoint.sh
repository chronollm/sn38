#!/bin/bash
set -e

# Reconstruct the bittensor hotkey file from the encrypted secret
if [ -n "$HOTKEY_FILE_CONTENT" ]; then
    # We use the environment variables provided in docker-compose
    WALLET_NAME="${WALLET_NAME:-validator}"
    WALLET_HOTKEY="${WALLET_HOTKEY:-default}"
    WALLET_PATH="/root/.bittensor/wallets/${WALLET_NAME}/hotkeys"
    mkdir -p "$WALLET_PATH"
    echo "$HOTKEY_FILE_CONTENT" > "$WALLET_PATH/${WALLET_HOTKEY}"
    echo "✅ Hotkey file successfully provisioned at $WALLET_PATH/${WALLET_HOTKEY}"
else
    echo "⚠️  HOTKEY_FILE_CONTENT not set. Ensure the wallet exists at the default path."
fi

exec "$@"
