"""Custom architecture registry.

Each subdirectory contains a custom model architecture that is
registered with HuggingFace AutoModel at import time. This allows
loading custom architectures with trust_remote_code=False.
"""

import importlib
import os
import pkgutil

for _, name, is_pkg in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if is_pkg:
        importlib.import_module(f".{name}", __name__)
