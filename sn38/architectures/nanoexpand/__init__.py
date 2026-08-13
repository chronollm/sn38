from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_nanoexpand import NanoExpandConfig
from .modeling_nanoexpand import NanoExpandForCausalLM

AutoConfig.register(NanoExpandConfig.model_type, NanoExpandConfig)
AutoModelForCausalLM.register(NanoExpandConfig, NanoExpandForCausalLM)
