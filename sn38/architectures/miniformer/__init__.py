from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_miniformer import MiniformerConfig
from .modeling_miniformer import MiniformerForCausalLM

AutoConfig.register(MiniformerConfig.model_type, MiniformerConfig)
AutoModelForCausalLM.register(MiniformerConfig, MiniformerForCausalLM)
