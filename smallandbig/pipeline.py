import torch

from .compression import (
    create_compatible_cache,
    prune_kv_layerwise,
    select_visual_tokens,
)
from .config import SmallAndBigConfig
from .keyframe import select_keyframes


def register_attention_hooks(model):
    total = [None]
    hooks = []

    def make_hook():
        def hook(module, inp, out):
            attn = out[1]
            attn_mean = attn.mean(dim=1)
            if total[0] is None:
                total[0] = attn_mean
            else:
                total[0] = total[0] + attn_mean.to(total[0].device)

        return hook

    for layer in model.language_model.layers:
        hooks.append(layer.self_attn.register_forward_hook(make_hook()))

    def get_scores():
        s = total[0]
        s = s.sum(dim=0).sum(dim=0)
        return s

    def remove():
        for h in hooks:
            h.remove()

    return get_scores, remove


def compress_kv_cache(model, past_key_values, total_scores, config: SmallAndBigConfig):
    key_cache, value_cache, token_scores, num_select = select_visual_tokens(
        past_key_values, total_scores, config.beta, config.sys_token_num
    )
    return prune_kv_layerwise(
        model,
        key_cache,
        value_cache,
        config.layer_id1,
        num_select,
        token_scores,
        config.compression_ratio,
        config.alloc_steepness,
        config.min_tokens,
        config.fusion_ratio,
        config.fusion_alpha,
        config.fusion_temperature,
        config.sys_token_num,
    )


def run(model, processor, messages, video_tensor, clip_model, device, config: SmallAndBigConfig,
        max_new_tokens=128, stopping_criteria=None):
    prompt = next(
        c["text"] for c in messages[0]["content"] if c["type"] == "text"
    )

    keyframes = select_keyframes(
        prompt, clip_model, video_tensor, device, config.alpha, config.mmr_lambda
    )
    video_tensor = video_tensor[keyframes]

    inputs = processor(
        text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
        images=None,
        videos=[[video_tensor]],
        fps=1,
        padding=True,
        return_tensors="pt",
    ).to(device)

    get_scores, remove_hooks = register_attention_hooks(model)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=True, return_dict=True)
    total_scores = get_scores()
    remove_hooks()

    past_key_values = outputs.past_key_values
    del outputs

    new_cache = compress_kv_cache(model, past_key_values, total_scores, config)
    del past_key_values

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        past_key_values=new_cache,
        do_sample=False,
        temperature=0.0,
    )
    if stopping_criteria is not None:
        gen_kwargs["stopping_criteria"] = stopping_criteria

    with torch.no_grad():
        generated_ids = model.generate(**gen_kwargs)

    out = generated_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(out, skip_special_tokens=True)
