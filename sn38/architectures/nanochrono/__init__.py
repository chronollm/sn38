from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_nanochrono import NanochronoConfig
from .modeling_nanochrono import NanochronoForCausalLM

AutoConfig.register(NanochronoConfig.model_type, NanochronoConfig)
AutoModelForCausalLM.register(NanochronoConfig, NanochronoForCausalLM)
