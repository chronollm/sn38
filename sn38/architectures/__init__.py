"""Custom architecture registry.

Each subdirectory contains a custom model architecture that is
registered with HuggingFace AutoModel at import time. This allows
loading custom architectures with trust_remote_code=False.

To add a new architecture, create a subdirectory with:
  - configuration_<name>.py (required)
  - modeling_<name>.py (required)
  - tokenization_<name>.py (optional, only if tokenizer.json is not enough)

Then add the registration calls below.
"""
