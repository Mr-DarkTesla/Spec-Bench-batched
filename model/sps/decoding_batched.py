import torch

def attention_mask_4d(attention_mask_2d):
    attention_mask_2d = attention_mask_2d.to(dtype=torch.float16)
    device = attention_mask_2d.device
    batch_size, seq_len = attention_mask_2d.shape

    mask = torch.bmm(attention_mask_2d[:, :, None], attention_mask_2d[:, None, :])
    tri = torch.ones(seq_len, seq_len).triu(diagonal=1) * float("-inf")
    tri = torch.nan_to_num(tri).to(device)
    mask += tri[None, :, :].repeat(batch_size, 1, 1)
    return mask[:, None, :, :]


def temperature_scaled_softmax(logits, temperature=1.0, dim=0):
    temperature = max(1e-3, temperature)
    logits = logits / temperature
    return torch.softmax(logits, dim=dim)

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
                                attention_mask=attention_mask_4d(verifier_attention_mask),
                                position_ids=verifier_position_ids,
                                do_sample=do_sample,
                                temperature=temperature,
                                ).logits[:, -1-max_new_tokens:-1, :]                                    

    # verifier_outputs = verifier.generate(
    #         input_ids=drafter_input_ids,
    #         attention_mask=verifier_attention_mask,
    #         position_ids=verifier_position_ids,
    #         do_sample=do_sample,
    #         temperature=temperature,
    #         max_new_tokens=max_new_tokens-1,
    #         return_dict_in_generate=True,
    #         output_scores=True)
    # print("VERIFIER:")
    # for sample in verifier_outputs.sequences:
    #     print(*[tokenizer.decode(tok) for tok in sample.cpu()], sep=" ")
    #     print("--------")

    drafter_length = drafter_logits.shape[1]
    _, n_matches, free_token = _speculative_sampling(
                drafter_input_ids,
                drafter_logits,
                drafter_length,
                verifier_logits,
                max_new_tokens-1,
                do_sample=do_sample,
                temperature=temperature,
            )

    # add padding
    max_n = max(n_matches)
    if max_n > 0:
        mask = torch.arange(max_n)[None, :].to(device) < n_matches[:, None]
        valid_tokens = (drafter_input_ids[:, -max_new_tokens:max_n-max_new_tokens] - pad_token_id) * mask + pad_token_id
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
    max_new_tokens,
    do_sample=False,
    temperature=0,
):
    batch_size = candidate_input_ids.shape[0]
    device = candidate_input_ids.device
    
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]

    q = temperature_scaled_softmax(candidate_logits, temperature, dim=-1)
    q_i = torch.gather(q, 2, new_candidate_input_ids[:, :, None]).squeeze(-1) #q[:, torch.arange(candidate_length), new_candidate_input_ids]

    p = temperature_scaled_softmax(new_logits, temperature, dim=-1)
    p_i = torch.gather(p, 2, new_candidate_input_ids[:, :, None]).squeeze(-1) #p[:, torch.arange(candidate_length), new_candidate_input_ids]

    probability_ratio = torch.clamp(p_i / q_i, min=0, max=1)
    # print(p_i, q_i, probability_ratio, sep="\n\n")

    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = torch.minimum(((~is_accepted).cumsum(dim=-1) < 1).sum(dim=-1), torch.ones((batch_size), dtype=int).to(device) * max_new_tokens)
    
    p_n_plus_1 = torch.take_along_dim(p, n_matches.resize(batch_size, 1, 1), 1)
    q_n_plus_1 = torch.take_along_dim(q, n_matches.resize(batch_size, 1, 1), 1)
    p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1 * (n_matches < max_new_tokens).reshape(batch_size, 1, 1)), min=0).squeeze(1)

    # print(p_prime)
    if not do_sample:
        t = torch.multinomial(p_prime, num_samples=1)
    else:
        t = torch.argmax(p_prime, dim=-1)[:, None]
    return None, n_matches, t
