import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model.sps.decoding_batched import sps_generate
# from evaluation.inference_sps import sps_forward as sps_forward_test
# from model.sps.decoding_batched import _speculative_sampling

# from model.sps.decoding import assisted_decoding
from fastchat.utils import str_to_torch_dtype
from transformers.tokenization_utils_base import BatchEncoding

class ResponseChecker:
    def __init__(self, tokenizer, prompts):
        self.prompts = prompts
        self.target = []
        self.pred = []
        self.tokenizer = tokenizer
        self.pad_token = tokenizer.pad_token
    
    def ids_to_str(self, ids):
        return self.tokenizer.decode(ids.cpu()).replace(self.pad_token, "").split("</s>")[0]
    
    def add_target(self, ids):
        for sample in ids:
            s = self.ids_to_str(sample)
            self.target.append(s)
    
    def add_pred(self, ids):
        for sample in ids:
            s = self.ids_to_str(sample)
            self.pred.append(s)

    def run_check(self):
        assert len(self.target) == len(self.prompts), f"{len(self.target)} {len(self.prompts)}"
        assert len(self.pred) == len(self.prompts), f"{len(self.pred)} {len(self.prompts)}"
        for prompt, pred, target in zip(self.prompts, self.pred, self.target):
            length = min(len(pred), len(target))
            if pred[:length] != target[:length]:
                print(  f"Error on prompt: {prompt}\n" \
                        f"Target    : {target}\n" \
                        f"Prediction: {pred}\n")
        print("Check finished")
    
    @staticmethod
    def output_token_by_token(output_ids, tokenizer):
        print("OUTPUT:")
        for sample in output_ids:
            print(tokenizer.decode(sample.cpu()).replace(pad_token, "").split("</s>")[0])
            print("------------------------------")

model_path = "meta-llama/Llama-2-7b-chat-hf"
drafter_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = "cuda:0"
dtype = "float16"
temperature = 0
max_new_tokens = 128

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    torch_dtype=str_to_torch_dtype(dtype),
    low_cpu_mem_usage=True,
    device_map = device,
)

drafter = AutoModelForCausalLM.from_pretrained(
    drafter_path,
    torch_dtype=str_to_torch_dtype(dtype),
    low_cpu_mem_usage=True,
    device_map = device,
)

model.eval()
drafter.eval()

if temperature > 0:
    do_sample = True
else:
    do_sample = False
tokenizer = AutoTokenizer.from_pretrained(model_path)
pad_token = "[PAD]"
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': pad_token})
    model.resize_token_embeddings(len(tokenizer))
    drafter.resize_token_embeddings(len(tokenizer))

pad_token = tokenizer.pad_token
pad_token_id = tokenizer.pad_token_id

prompts = [
    "Translate German to English: „ Das verändert die Zukunft meiner Familie “ , meinte der Mann .",
    "Translate German to English: \" Das habe ich verboten \" , erklärte er .",
    "Translate German to English: Als Zeuge war der damals ermittelnde Kommissar geladen .",
    "Translate German to English: Je dunkler das Fleisch , desto höher der ph-Wert .",
    "Random test 1",
    "Something to generate",
    "Blabla bla",
    "ewfj2nfo f32ipojd;qnf 32oirj0[923jfqe  39r0uq23'j]",
]
prompt_template=[f"[INST] {prompt} [/INST]" for prompt in prompts]

checker = ResponseChecker(tokenizer, prompt_template)

# My batched sps
inputs = tokenizer( prompt_template, 
                        add_special_tokens=True,
                        # truncation=True, 
                        padding="longest", 
                        return_attention_mask=True, 
                        return_tensors="pt",
                        padding_side="left"
                        ).to(device)
input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
input_ids, step, accept_length = sps_generate(
                        input_ids, attention_mask,
                        model,
                        tokenizer,
                        max_new_tokens,
                        drafter=drafter,
                            do_sample=do_sample,
                            temperature=temperature,
                    )
print("AVG ACCEPT:", accept_length)
checker.add_pred(input_ids)

# Just model generation
for prompt in prompt_template:
    input_ids = tokenizer([prompt], return_tensors="pt").to(device).input_ids
    inputs = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    do_sample=do_sample,
    temperature=temperature,
    return_dict_in_generate=True,
    output_scores=True)
    checker.add_target(inputs.sequences)
checker.run_check()