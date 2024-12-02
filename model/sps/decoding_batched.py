import torch

def temperature_scaled_softmax(logits, temperature=0.0, dim=0):
    assert not logits.isnan().any()
    # if temperature > 0:
    temperature = max(temperature, 1e-3)
    res = torch.softmax(logits / temperature, dim=dim)
    # else:
    #     res = torch.argmax(logits, dim=dim)
    return res

def check_mask(ids, mask, token_id):
    batch_size, seq_len = mask.shape
    for b in range(batch_size):
        for idx, (i, m) in enumerate(zip(ids[b], mask[b])):
            assert (m and i != token_id) or (not m and i == token_id), f"Batch: {b}, Index: {idx}, TokenId: {i}, Mask: {m}"

@torch.no_grad()
def sps_generate(
    input_ids, 
    attention_mask,
    model, 
    tokenizer, 
    max_new_tokens,
    tokens_per_forward=4,
    drafter=None, 
    do_sample=False, 
    temperature=0.0
):  
    cum_accept = torch.zeros(len(input_ids)).to(input_ids.device)
    step = 0
    while True:
        step += 1
        input_ids, accept_length_tree, attention_mask, finished = sps_forward(
            input_ids, attention_mask,
            model,
            tokenizer,
            tokens_per_forward,
            drafter=drafter,
            do_sample=do_sample,
            temperature=temperature,
        )
        cum_accept += accept_length_tree
        if finished.all() or (cum_accept + step >= max_new_tokens).any():
            return input_ids, step, cum_accept / step

@torch.no_grad()
def sps_forward(
    input_ids, 
    drafter_attention_mask,
    verifier, 
    tokenizer, 
    max_new_tokens,
    drafter=None, 
    do_sample=False, 
    temperature=0.0
):  
    drafter_position_ids = torch.cumsum(drafter_attention_mask, dim=-1)
    batch_size, input_len = input_ids.shape
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    device = input_ids.device

    check_mask(input_ids, drafter_attention_mask, pad_token_id)

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
    drafter_input_ids, drafter_logits = drafter_outputs.sequences, torch.stack(drafter_outputs.scores, dim=1).to(dtype=drafter.dtype)
    drafter_length = drafter_logits.shape[1]

    verifier_attention_mask = torch.cat((drafter_attention_mask,
                                torch.ones((batch_size, max_new_tokens)).to(device)), 1)
    verifier_position_ids = torch.cumsum(verifier_attention_mask, dim=-1)

    verifier_logits = verifier( drafter_input_ids,
                                attention_mask=verifier_attention_mask,
                                position_ids=verifier_position_ids,
                                do_sample=do_sample,
                                temperature=temperature,
                                ).logits
    # print(verifier_attention_mask[:, -2*max_new_tokens:])
    # print(drafter_input_ids[:, -2*max_new_tokens:])
    # print(verifier_logits[:, -1-max_new_tokens:, :])
    # print(torch.softmax(verifier_logits[:, -2, :], dim=-1))
    # print(torch.argmax(verifier_logits[:, -2, :], dim=-1))
    assert not verifier_logits.isnan().any()
    # breakpoint()
    n_matches, free_token = _speculative_sampling(
                drafter_input_ids,
                drafter_logits,
                drafter_length,
                verifier_logits[:, -1-max_new_tokens:-1, :],
                max_new_tokens-1,
                do_sample=do_sample,
                temperature=temperature,
            )
    
    # print(free_token)

    def pad_by_n(ids, n, max_tok):
        max_n = max(n)
        assert max_n < max_tok
        mask = torch.arange(max_n)[None, :].to(device) < n[:, None]
        padded = (ids[:, -max_tok:max_n-max_tok] - pad_token_id) * mask + pad_token_id
        return padded, mask
    
    # add paddings
    max_n = max(n_matches)
    if max_n > 0:
        valid_tokens, mask = pad_by_n(drafter_input_ids, n_matches, max_new_tokens)
        new_tokens = torch.cat((valid_tokens, free_token), dim=-1)

        output_ids = torch.cat((input_ids, new_tokens), dim=-1)
        attention_mask = torch.cat([drafter_attention_mask, mask], dim=1)
    else:
        output_ids = torch.cat((input_ids, free_token), dim=-1)
        attention_mask = drafter_attention_mask

    finished = ((output_ids == eos_token_id).sum(dim=-1) > 0)
    attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1)).to(device, dtype=int)), dim=1)
    check_mask(output_ids, attention_mask, pad_token_id)
    return output_ids, n_matches, attention_mask, finished


@torch.no_grad()
def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    max_new_tokens,
    do_sample=False,
    temperature=0,
):
    batch_size = candidate_input_ids.shape[0]
    device = candidate_input_ids.device
    
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]

    q = temperature_scaled_softmax(candidate_logits, temperature, dim=-1)
    q_i = torch.gather(q, 2, new_candidate_input_ids[:, :, None]).squeeze(-1)

    p = temperature_scaled_softmax(new_logits, temperature, dim=-1)
    p_i = torch.gather(p, 2, new_candidate_input_ids[:, :, None]).squeeze(-1)

    probability_ratio = torch.clamp(p_i / q_i, min=0, max=1)

    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio

    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum(dim=-1)
    max_new_tok_tensor = torch.ones((batch_size), dtype=int).to(device) * max_new_tokens
    n_matches = torch.minimum(n_matches, max_new_tok_tensor)
    
    p_n_plus_1 = torch.take_along_dim(p, n_matches.resize(batch_size, 1, 1), 1)
    q_n_plus_1 = torch.take_along_dim(q, n_matches.resize(batch_size, 1, 1), 1)
    p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1 * (n_matches < max_new_tokens).reshape(batch_size, 1, 1)), min=0).squeeze(1)
    if do_sample:
        t = torch.multinomial(p_prime, num_samples=1)
    else:
        t = torch.argmax(p_prime, dim=-1)[:, None]
    return n_matches, t
