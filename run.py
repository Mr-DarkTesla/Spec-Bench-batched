import torch
import numpy as np
# from evaluation.inference_sps import sps_forward
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from model.sps.decoding import assisted_decoding
from fastchat.utils import str_to_torch_dtype
from transformers.tokenization_utils_base import BatchEncoding

def sps_forward(
    # inputs,
    input_ids, attention_mask,
     verifier, drafter, max_new_tokens, do_sample=False, temperature=0.0):
    # input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    # print("INPUT :", *[tokenizer.decode(sample) for sample in inputs.input_ids.cpu()], sep="\n")

    print("BEFORE DRAFTER:", *[tokenizer.decode(sample) for sample in input_ids.cpu()], sep="\n")
    drafter_outputs = drafter.generate(
        # **inputs,
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        return_dict_in_generate=True,
        output_scores=True)

    transition_scores = drafter.compute_transition_scores(
        drafter_outputs.sequences, drafter_outputs.scores, normalize_logits=True
    )
    batch_size, input_len = input_ids.shape
    draftedted_tokens = drafter_outputs.sequences[:, input_len:]
    print("DRAFTER:", *[tokenizer.decode(sample) for sample in draftedted_tokens.cpu()], sep="\n")

    # Drafter probs for rejections
    drafter_probs = torch.exp(transition_scores)

    # Adjust attention mask
    attention_mask_ver = torch.cat((attention_mask,
                                torch.ones((batch_size, max_new_tokens)).to(device)), 1)
    verifier_logits = verifier(drafter_outputs.sequences, attention_mask=attention_mask_ver).logits
    # find_token = lambda idx: verifier_logits[*idx, draftedted_tokens[*idx]]
    # verifier_probs = .apply_(find_token).to(device)
    indices = np.indices(drafter_probs.shape)
    verifier_probs = torch.tensor([[verifier_logits[batch, seq, draftedted_tokens[batch, seq]]
        for seq in range(max_new_tokens)]
        for batch in range(batch_size)]).to(device) # todo optimize (torch gather)

    reject_rate = torch.max(torch.zeros_like(verifier_probs), 1 - verifier_probs / drafter_probs)
    rand = torch.rand(reject_rate.shape).to(device)
    accept = (reject_rate < rand)

    # todo optimize 
    # n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1
    for i, lst in enumerate(accept):
        for j, el in enumerate(lst):
            accept[i, j] = False if (j > 0 and not lst[j - 1]) else accept[i, j]
    accept = accept.to(int)
    attention_mask = torch.cat((attention_mask, accept), 1)

    output_ids = torch.cat((input_ids, (draftedted_tokens - pad_token_id) * accept + pad_token_id), 1)
    return output_ids, attention_mask

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
    # device_map="auto"
).to(device)

drafter = AutoModelForCausalLM.from_pretrained(
    drafter_path,
    torch_dtype=str_to_torch_dtype(dtype),
    low_cpu_mem_usage=True,
    # device_map="auto"
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
# print(inputs.input_ids.shape, inputs.attention_mask.shape)
input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
print("INPUT :", *[tokenizer.decode(sample) for sample in input_ids.cpu()], sep="\n")

for _ in range(4):
    output_ids, attention_mask = sps_forward(
        input_ids, attention_mask,
        model, drafter, max_new_tokens, do_sample=do_sample, temperature=temperature)
    input_ids, attention_mask = output_ids, attention_mask
    print("OUTPUT:", *[tokenizer.decode(sample) for sample in output_ids.cpu()], sep="\n")

