from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_chronogrow import ChronoGrowConfig
from .modeling_chronogrow import ChronoGrowForCausalLM

AutoConfig.register(ChronoGrowConfig.model_type, ChronoGrowConfig)
AutoModelForCausalLM.register(ChronoGrowConfig, ChronoGrowForCausalLM)
