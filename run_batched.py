import torch
import numpy as np
# from evaluation.inference_sps import sps_forward
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin

# from run import sps_forward
from model.sps.decoding_batched import _speculative_sampling

from model.sps.decoding import assisted_decoding
from fastchat.utils import str_to_torch_dtype
from transformers.tokenization_utils_base import BatchEncoding

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
drafter_path = "JackFram/llama-68m"
device = "cuda:6"
dtype = "float16"
temperature = 0.1
max_new_tokens = 8

GenerationMixin.assisted_decoding = assisted_decoding

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=str_to_torch_dtype(dtype),
    low_cpu_mem_usage=True,
).to(device)

drafter = AutoModelForCausalLM.from_pretrained(
    drafter_path,
    torch_dtype=str_to_torch_dtype(dtype),
    low_cpu_mem_usage=True,
).to(device)

model.eval()
drafter.eval()

if temperature > 0:
    do_sample = True
else:
    do_sample = False
tokenizer = AutoTokenizer.from_pretrained(model_path)
pad_token_id = tokenizer.pad_token_id

prompt = [
            "The capital city of California state is",
            "What do you think about",
            "WTF is going on?"
          ]
inputs = tokenizer.batch_encode_plus(prompt, 
                                    add_special_tokens=True,
                                    # truncation=True, 
                                    padding=True, 
                                    return_attention_mask=True, 
                                    return_tensors="pt",
                                    pad_to_multiple_of=8).to(device)

def sps_forward(inputs, verifier, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None):
    assert drafter is not None
    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    drafter_outputs = drafter.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens - 1,
        do_sample=do_sample,
        temperature=temperature,
        return_dict_in_generate=True,
        output_logits=True,)
    # drafter_logits = drafter.compute_transition_scores(
    #     drafter_outputs.sequences, drafter_outputs.scores, normalize_logits=True
    # )
    
    batch_size, input_len = input_ids.shape
    # draftedted_tokens = drafter_outputs.sequences[:, input_len:]

    attention_mask_ver = torch.cat((attention_mask,
                                torch.ones((batch_size, max_new_tokens - 1)).to(device)), 1)
    verifier_outputs = verifier(drafter_outputs.sequences, attention_mask=attention_mask_ver)

    # last_assistant_token_is_eos = (
    #         ~draftedted_tokens[:, -1]
    #         .tile(eos_token_id_tensor.shape[0], 1)
    #         .ne(eos_token_id_tensor.unsqueeze(1))
    #         .prod(dim=0)
    #         .bool()
    #     )
    
    valid_tokens, n_matches = _speculative_sampling(
        drafter_outputs.sequences,
        torch.stack(drafter_outputs.logits, dim=1),
        max_new_tokens - 1,
        verifier_outputs.logits[:, 1-max_new_tokens:],
        None,
        max_new_tokens,
    )
    output_ids = torch.cat((input_ids, valid_tokens), dim=-1)
    new_token, idx, accept_length_list = None, None, None
    return output_ids, new_token, idx+1, accept_length_list

output_ids, step, accept_length_tree = sps_forward(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        drafter=drafter
                    )
print("OUTPUT:")
for sample in output_ids.cpu():
    print("OUTPUT:", *[tokenizer.decode(sample) for sample in output_ids.cpu()], sep="\n")
