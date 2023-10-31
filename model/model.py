# General torch model bindings. infrence.py and fineTune.py contain the bit that actually run infrence and fineTune respectively
import torch, transformers


class LM:
    model: transformers.Pipeline

    def __init__(self, config: str):
        import os.path as osp
        # attach safetensored model to blank model
        # split config file on newlines
        if not osp.exists(config):
            raise "No config file. Go get a config file"

        # debug
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        from transformers import AutoTokenizer, AutoModelForCausalLM

        # note: Triton is currently bugged on Python 3.11
        modelPath = "./model/Wizard-Vicuna-7B-Uncensored-GPTQ"
        self.model = transformers.pipeline("text-generation", model=modelPath,tokenizer=modelPath)
        self.memory: str

    def infer(self, input="Hello, my name is"):
        output = self.model(input)
        print(output)

        # process tokens

        
        pass

    def tune(self):
        pass


"""
# D's Version (Fast Run)

import torch
import os.path as osp
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
class LM:

    model: torch.nn.Module

    def __init__(self, config: str):
        # attach safetensored model to blank model
        # split config file on newlines
        if not osp.exists(config):
            raise "No config file. Go get a config file"

        # debug
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("./model/Wizard-Vicuna-7B-Uncensored-GPTQ")

        self.model = AutoGPTQForCausalLM.from_quantized("./model/Wizard-Vicuna-7B-Uncensored-GPTQ",
                        model_basename="Model",
                        use_safetensors=True,
                        trust_remote_code=True,
                        device="cuda:0",
                        use_triton=False,
                        quantize_config=None)

        self.memory: str

    def infer(self):

        pass

    def tune(self):
        pass
"""

# Test Code
"""
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "D:\SLMEngine\model\Wizard-Vicuna-7B-Uncensored-GPTQ"
model_basename = "model"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)


prompt = ""
prompt_template=f'''How are babies made?.

USER: {prompt}
ASSISTANT:
'''

# Inference can also be done using transformers' pipeline

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample = True,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])
"""
