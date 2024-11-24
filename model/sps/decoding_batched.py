import torch
import numpy as np

@torch.no_grad()
def sps_forward(
    input_ids, drafter_attention_mask,
    # inputs,
    verifier, tokenizer, max_new_tokens, drafter=None, do_sample=False, temperature=0.0):
    # input_ids, drafter_attention_mask = inputs.input_ids, inputs.attention_mask
    drafter_position_ids = torch.cumsum(drafter_attention_mask, dim=-1)
    batch_size, input_len = input_ids.shape
    pad_token_id = tokenizer.pad_token_id
    device = input_ids.device

    # sps_sampling -> output_ids, idx, accept_length_list
    drafter_outputs = drafter.generate(
        input_ids=input_ids,
        attention_mask=drafter_attention_mask,
        position_ids=drafter_position_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        return_dict_in_generate=True,
        output_scores=True)
    drafter_input_ids, drafter_logits = drafter_outputs.sequences, torch.stack(drafter_outputs.scores, dim=1)

    # print("DRAFTER:")
    # for sample in drafter_input_ids:
    #     print(*[tokenizer.decode(tok) for tok in sample.cpu()], sep=" ")
    #     print("--------")



    verifier_attention_mask = torch.cat((drafter_attention_mask,
                                torch.ones((batch_size, max_new_tokens)).to(device)), 1)
    verifier_position_ids = torch.cumsum(verifier_attention_mask, dim=-1)
    verifier_logits = verifier( drafter_input_ids,
                                attention_mask=verifier_attention_mask,
                                position_ids=verifier_position_ids,
                                do_sample=do_sample,
                                temperature=temperature,
                                ).logits[:, -1-max_new_tokens:, :]                                    

    verifier_outputs = verifier.generate(
            input_ids=drafter_input_ids,
            attention_mask=verifier_attention_mask,
            position_ids=verifier_position_ids,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True)
    print("VERIFIER:")
    for sample in verifier_outputs.sequences:
        print(*[tokenizer.decode(tok) for tok in sample.cpu()], sep=" ")
        print("--------")

    drafter_length = drafter_logits.shape[1]
    _, n_matches, free_token = _speculative_sampling(
                drafter_input_ids,
                drafter_logits,
                drafter_length,
                verifier_logits,
                do_sample=do_sample,
                temperature=temperature,
            )

    # add padding
    max_n = max(n_matches)
    mask = torch.arange(max_n)[None, :].to(device) < n_matches[:, None]
    if max_n > 0:
        valid_tokens = (drafter_input_ids[:, -max_n:] - pad_token_id) * mask + pad_token_id
        output_ids = torch.cat((input_ids, valid_tokens, free_token), dim=-1)
        attention_mask = torch.cat([drafter_attention_mask, mask], dim=1)
    else:
        valid_tokens = torch.zeros(batch_size, 0).to(device)
        output_ids = torch.cat((input_ids, free_token), dim=-1)
        attention_mask = drafter_attention_mask

    attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1)).to(device)), dim=1)
    assert output_ids.shape == attention_mask.shape
    return output_ids, valid_tokens, None, n_matches, attention_mask


@torch.no_grad()
def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    do_sample,
    temperature=0,
):
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]

    q = candidate_logits.softmax(dim=-1)
    q_i = torch.gather(q, 2, new_candidate_input_ids[:, :, None]).squeeze(-1) #q[:, torch.arange(candidate_length), new_candidate_input_ids]
    if not do_sample:
        q_i = torch.ones_like(q_i)

    p = new_logits.softmax(dim=-1)
    p_i = torch.gather(p, 2, new_candidate_input_ids[:, :, None]).squeeze(-1) #p[:, torch.arange(candidate_length), new_candidate_input_ids]

    probability_ratio = torch.clamp(p_i / q_i, min=0, max=1)
    # print(p_i, q_i, probability_ratio, sep="\n\n")

    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum(dim=-1)
    
    p_n_plus_1 = p[torch.arange(p.size(0)), n_matches]
    zeros = torch.zeros_like(q[:, 0, :]).unsqueeze(1)
    q_n_plus_1 = torch.cat((q, zeros), dim=1)[torch.arange(q.size(0)), n_matches]
    p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)

    p_sum = p_prime.sum(dim=-1)
    p_prime = torch.div(p_prime.T, p_sum).T
    
    if not do_sample:
        t = torch.multinomial(p_prime, num_samples=1)
    else:
        t = torch.argmax(p_prime, dim=-1)[:, None]
    return None, n_matches, t
