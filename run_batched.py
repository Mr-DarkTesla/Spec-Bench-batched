import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model.sps.decoding_batched import sps_forward
from evaluation.inference_sps import sps_forward as sps_forward_test
# from model.sps.decoding_batched import _speculative_sampling

# from model.sps.decoding import assisted_decoding
from fastchat.utils import str_to_torch_dtype
from transformers.tokenization_utils_base import BatchEncoding

def output_token_by_token(output_ids, tokenizer):
    print("OUTPUT:")
    for sample in output_ids:
        # replace_pad = lambda s : s if s != "</s>" else ""
        print(tokenizer.decode(sample.cpu()).replace(pad_token, "").split("</s>")[0])
        print("------------------------------")


# "meta-llama/Llama-2-7b-chat-hf" "TheBloke/Llama-2-7B-Chat-GPTQ" "meta-llama/Llama-3.1-8B-Instruct"
model_path = "meta-llama/Llama-2-7b-chat-hf"
# model_path = "meta-llama/Llama-3.1-8B-Instruct"
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  "JackFram/llama-68m" "meta-llama/Llama-3.2-1B-Instruct" 
drafter_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# drafter_path = "meta-llama/Llama-3.2-1B-Instruct" 

device = "cuda:4"
dtype = "float16"
temperature = 0
max_new_tokens = 4

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=str_to_torch_dtype(dtype),
    low_cpu_mem_usage=True,
    device_map = device,
)

drafter = AutoModelForCausalLM.from_pretrained(
    drafter_path,
    # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=str_to_torch_dtype(dtype),
    low_cpu_mem_usage=True,
    device_map = device,
)

from transformers import GenerationMixin
from model.sps.decoding import assisted_decoding
GenerationMixin.assisted_decoding = assisted_decoding

model.eval()
drafter.eval()

if temperature > 0:
    do_sample = True
else:
    do_sample = False
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    pad_token = "<s>"
    tokenizer.add_special_tokens({'pad_token': pad_token})
    model.resize_token_embeddings(len(tokenizer))
pad_token_id = tokenizer.pad_token_id

# prompts = [
#             # "What is the capital city of California state?",
#             # "What do you think about global warming?",
#             # "WTF is going on?",
#             "Who plays young Damon in The Vampire Diaries?",
#             "Tell me a story.",
#             "Continue sequence 1 2 3 4 5 6:",
#             # "Count from 1 to 10:",
#             # "9 8 7 6 5",
#             # "a b c d"
#           ]

prompts = [
    "Translate German to English: „ Das verändert die Zukunft meiner Familie “ , meinte der Mann .",
    "Translate German to English: \" Das habe ich verboten \" , erklärte er .",
    "Translate German to English: Als Zeuge war der damals ermittelnde Kommissar geladen .",
    "Translate German to English: Je dunkler das Fleisch , desto höher der ph-Wert .",
]
# SPACE IN THE END IS IMPORTANT FOR SPECBENCH SPS
prompt_template=[f"[INST] {prompt} [/INST]" for prompt in prompts]

iterations = 15


# My batched sps
print("\nMy SPS:")
inputs = tokenizer( prompt_template, 
                        add_special_tokens=True,
                        # truncation=True, 
                        padding="longest", 
                        return_attention_mask=True, 
                        return_tensors="pt",
                        padding_side="right").to(device)
input_ids, attention_mask = inputs.input_ids, inputs.attention_mask

for _ in range(iterations):
    input_ids, step, _, accept_length_tree, attention_mask = sps_forward(
                            input_ids, attention_mask,
                            model,
                            tokenizer,
                            max_new_tokens,
                            drafter=drafter,
                            do_sample=do_sample,
                            temperature=temperature,
                        )
    # print(accept_length_tree)
output_token_by_token(input_ids, tokenizer)


# # Original sps frmo spec bench
# print("\nOriginal SPS frmo spec bench:")
# for prompt in prompt_template:
#     inputs = tokenizer([prompt], return_tensors="pt").to(device).input_ids
#     for i in range(iterations):
#         inputs = sps_forward_test(
#                     inputs,
#                     model,
#                     tokenizer,
#                     max_new_tokens,
#                     drafter=drafter,
#                     do_sample=do_sample,
#                     temperature=temperature)
#     output_token_by_token(inputs, tokenizer)

# Just model 
print("\nModel Generation:")
for prompt in prompt_template:
    inputs = tokenizer([prompt], return_tensors="pt").to(device).input_ids
    for i in range(2 * iterations):
        inputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature)
    output_token_by_token(inputs, tokenizer)