"""Custom architecture registry.

Each subdirectory registers an architecture with HuggingFace AutoModel, so
custom architectures load with trust_remote_code=False. Registration is lazy:
only the architecture a checkpoint actually declares is imported.
"""

import importlib
import json
import os
import pkgutil

_DIR = os.path.dirname(__file__)
_PREFIX = "sn38-"


def register_all():
    """Import every architecture."""
    for _, name, is_pkg in pkgutil.iter_modules([_DIR]):
        if is_pkg:
            importlib.import_module("." + name, __name__)


def register_for(model_path):
    """Import only the architecture this checkpoint's config declares."""
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            model_type = json.load(f).get("model_type", "")
    except (OSError, ValueError):
        register_all()
        return
    if model_type.startswith(_PREFIX):
        importlib.import_module("." + model_type[len(_PREFIX):], __name__)
