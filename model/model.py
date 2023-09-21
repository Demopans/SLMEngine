# General torch model bindings. infrence.py and fineTune.py contain the bit that actually run infrence and fineTune respectively

import torch



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

        self.model = AutoModelForCausalLM.from_pretrained("./model/Wizard-Vicuna-7B-Uncensored-GPTQ", device_map="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("./model/Wizard-Vicuna-7B-Uncensored-GPTQ")
        self.memory: str


    def infer(self):

        pass

    def tune(self):
        pass
