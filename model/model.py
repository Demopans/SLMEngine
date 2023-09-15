# General torch model bindings. infrence.py and fineTune.py contain the bit that actually run infrence and fineTune respectively
import torch
import sys
import safetensors.torch, safetensors
from auto_gptq import AutoGPTQForCausalLM


class LM:
    model: torch.nn.Module

    def __init__(self, config: str):
        # attach safetensored model to blank model
        # split config file on newlines

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("model/Wizard-Vicuna-7B-Uncensored-GPTQ")

        model = AutoGPTQForCausalLM.from_quantized("model/Wizard-Vicuna-7B-Uncensored-GPTQ",
                                                   model_basename="model",
                                                   use_safetensors=True,
                                                   trust_remote_code=True,
                                                   device=dev,
                                                   use_triton=False,
                                                   quantize_config=None)

        self.model = torch.nn.Module()
        pass

    def infer(self):
        pass

    def tune(self):
        pass
