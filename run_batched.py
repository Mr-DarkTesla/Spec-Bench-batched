import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model.sps.decoding_batched import sps_generate
# from evaluation.inference_sps import sps_forward as sps_forward_test
# from model.sps.decoding_batched import _speculative_sampling

# from model.sps.decoding import assisted_decoding
from fastchat.utils import str_to_torch_dtype
from transformers.tokenization_utils_base import BatchEncoding

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
max_new_tokens = 30

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
]
prompt_template=[f"[INST] {prompt} [/INST]" for prompt in prompts]

# My batched sps
print("\nMy SPS:")
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
output_token_by_token(input_ids, tokenizer)

# Just model generation
print("\nModel Generation:")
for prompt in prompt_template:
    input_ids = tokenizer([prompt], return_tensors="pt").to(device).input_ids
    inputs = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    do_sample=do_sample,
    temperature=temperature,
    return_dict_in_generate=True,
    output_scores=True)
    # input_ids, logits = inputs.sequences, torch.stack(inputs.scores, dim=1)
    # print(torch.softmax(logits[0][0], dim=-1))
    # print(torch.argmax(logits[0][0], dim=-1))
    output_token_by_token(inputs.sequences, tokenizer)
