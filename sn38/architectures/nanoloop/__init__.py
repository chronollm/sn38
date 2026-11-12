from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_nanoloop import NanoLoopConfig
from .modeling_nanoloop import NanoLoopForCausalLM

AutoConfig.register(NanoLoopConfig.model_type, NanoLoopConfig)
AutoModelForCausalLM.register(NanoLoopConfig, NanoLoopForCausalLM)
