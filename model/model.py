# General torch model bindings. infrence.py and fineTune.py contain the bit that actually run infrence and fineTune respectively
import torch
import sys

class lm:
    model: torch.nn.Module

    def initModel(self, config: str):
        # attach safetensored model to blank model

        pass

    def infer(self):
        pass

    def tune(self):
        pass

