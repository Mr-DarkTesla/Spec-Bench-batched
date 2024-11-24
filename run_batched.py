import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model.sps.decoding_batched import sps_forward
# from model.sps.decoding_batched import _speculative_sampling

# from model.sps.decoding import assisted_decoding
from fastchat.utils import str_to_torch_dtype
from transformers.tokenization_utils_base import BatchEncoding

def output_token_by_token(output_ids, tokenizer):
    print("OUTPUT:")
    for sample in output_ids:
        replace_pad = lambda s : s if s != "</s>" else ""
        print(*[replace_pad(tokenizer.decode(tok)) for tok in sample.cpu()], sep=" ")
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

model.eval()
drafter.eval()

if temperature > 0:
    do_sample = True
else:
    do_sample = False
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '</s>'})
    model.resize_token_embeddings(len(tokenizer))
pad_token_id = tokenizer.pad_token_id

#  Spaces after prompts
prompts = [
            # "The capital city of California state is ",
            # "What do you think about global warming? ",
            # "WTF is going on? ",
            "Tell me a story ",
            "Continue sequence 1 2 3 4 5 6 ",
            # "Count from 1 to 10: ",
            # "9 8 7 6 5 ",
            # "a b c d "
          ]
prompt_template=[f'''<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST]''' for prompt in prompts]

inputs = tokenizer( prompts, 
                        add_special_tokens=True,
                        # truncation=True, 
                        padding="longest", 
                        return_attention_mask=True, 
                        return_tensors="pt",
                        padding_side="right").to(device)
input_ids, attention_mask = inputs.input_ids, inputs.attention_mask

for _ in range(3):
    input_ids, step, _, accept_length_tree, attention_mask = sps_forward(
                            # inputs,
                            input_ids, attention_mask,
                            model,
                            tokenizer,
                            max_new_tokens,
                            drafter=drafter,
                            do_sample=do_sample,
                            temperature=temperature,
                        )

    output_token_by_token(input_ids, tokenizer)
    print(accept_length_tree)

