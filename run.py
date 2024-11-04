# Load model directly
import torch
# from evaluation.inference_sps import sps_forward
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from model.sps.decoding import assisted_decoding
from fastchat.utils import str_to_torch_dtype
from transformers.tokenization_utils_base import BatchEncoding

def sps_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None):
    input_ids = inputs.input_ids
    model.generation_config.max_new_tokens = max_new_tokens
    output_ids, idx, accept_length_list = model.generate(
        **inputs, generation_config=model.generation_config, assistant_model=drafter, do_sample=do_sample, temperature=temperature)
    new_token = len(output_ids[0][len(input_ids[0]):])
    return output_ids, new_token, idx+1, accept_length_list

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
drafter_path = "JackFram/llama-68m"
dtype = "float16"
temperature = 0.0
max_new_tokens = 50

GenerationMixin.assisted_decoding = assisted_decoding

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=str_to_torch_dtype(dtype),
    low_cpu_mem_usage=True,
    device_map="auto"
)

drafter = AutoModelForCausalLM.from_pretrained(
    drafter_path,
    torch_dtype=str_to_torch_dtype(dtype),
    low_cpu_mem_usage=True,
    device_map="auto"
)

model.eval()
drafter.eval()

if temperature > 0:
    do_sample = True
else:
    do_sample = False
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = ["The capital city of California state is",
          "What do you think about",
          "WTF is going on?"]
inputs = tokenizer.batch_encode_plus(prompt, 
                                    add_special_tokens=True,
                                    # truncation=True, 
                                    padding=True, 
                                    return_attention_mask=True, 
                                    return_tensors="pt",
                                    pad_to_multiple_of=4).to("cuda")
input_ids = inputs.input_ids
output_ids, new_token, idx, accept_length_list = sps_forward(inputs, model, tokenizer, max_new_tokens,
 do_sample=do_sample, temperature=temperature, drafter=drafter)
# print(output_ids, new_token, idx, accept_length_list)
print(f"Input: {[tokenizer.decode(sample) for sample in input_ids.cpu()]}")
print(f"Output: {[tokenizer.decode(sample) for sample in output_ids.cpu()]}")
