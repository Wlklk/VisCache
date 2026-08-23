import math

import torch
import torch.nn.functional as F


def create_compatible_cache(model, key_cache, value_cache):
    try:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for i, (k, v) in enumerate(zip(key_cache, value_cache)):
            cache.update(k, v, i)
        return cache
    except (ImportError, AttributeError):
        return tuple((k, v) for k, v in zip(key_cache, value_cache))


def monotonic_parabolic_allocation(
    num_layers, original_tokens, compression_ratio=0.75, steepness=0.5, min_tokens=1
):
    weights = []
    a = steepness / (num_layers ** 2)
    for i in range(num_layers):
        w = max(0.0, min(1.0, -a * (i ** 2) + 1.0))
        weights.append(w)
    norm = num_layers / max(sum(weights), 1e-8)
    weights = [w * norm for w in weights]

    target = round(original_tokens * compression_ratio * num_layers)
    alloc = [max(min_tokens, int(w * original_tokens)) for w in weights]
    return _adjust_monotonic(alloc, target, min_tokens)


def _adjust_monotonic(alloc, target, min_tokens=1):
    alloc = list(alloc)
    for i in range(1, len(alloc)):
        alloc[i] = min(alloc[i], alloc[i - 1])

    diff = target - sum(alloc)
    if diff > 0:
        i = 0
        while diff > 0:
            if i == 0 or alloc[i] < alloc[i - 1] - 1:
                alloc[i] += 1
                diff -= 1
            i = (i + 1) % len(alloc)
    elif diff < 0:
        for i in reversed(range(len(alloc))):
            while diff < 0 and alloc[i] > min_tokens:
                alloc[i] -= 1
                diff += 1
    for i in range(1, len(alloc)):
        alloc[i] = min(alloc[i], alloc[i - 1])
    return alloc


def selective_attention_fusion(
    v_vis, keep_idx, attention_scores, fusion_ratio=0.3, alpha=0.2, temperature=1.0
):
    B, H, L, D = v_vis.shape
    keep_idx = keep_idx.long().to(v_vis.device)
    K = keep_idx.shape[-1]

    if attention_scores.dim() == 1:
        attention_scores = attention_scores.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
    elif attention_scores.dim() == 2:
        attention_scores = attention_scores.unsqueeze(1).expand(B, H, L)
    elif attention_scores.dim() == 3 and attention_scores.shape[1] == 1:
        attention_scores = attention_scores.expand(B, H, L)

    v_enhanced = torch.zeros(B, H, K, D, device=v_vis.device, dtype=v_vis.dtype)
    all_idx = torch.arange(L, device=v_vis.device)

    for b in range(B):
        for h in range(H):
            ki = keep_idx[b, h]
            drop_mask = torch.ones(L, dtype=torch.bool, device=v_vis.device)
            drop_mask[ki] = False
            drop_idx = all_idx[drop_mask].to(attention_scores.device)
            if drop_idx.numel() == 0:
                v_enhanced[b, h] = v_vis[b, h, ki]
                continue

            drop_scores = attention_scores[b, h, drop_idx]
            n_fuse = max(1, int(drop_idx.numel() * fusion_ratio))
            _, top_drop = torch.topk(drop_scores, k=n_fuse, dim=-1)
            fuse_idx = drop_idx[top_drop].to(v_vis.device)

            V_keep = v_vis[b, h, ki]
            V_fuse = v_vis[b, h, fuse_idx]
            fuse_scores = attention_scores[b, h, fuse_idx].to(V_fuse.device)
            w = F.softmax(fuse_scores / temperature, dim=-1)
            weighted = torch.sum(w.unsqueeze(-1) * V_fuse, dim=0)
            for k_idx in range(K):
                v_enhanced[b, h, k_idx] = V_keep[k_idx] + alpha * (weighted - V_keep[k_idx])
    return v_enhanced


def select_visual_tokens(past_key_values, total_scores, beta, sys_token_num=15):
    L = total_scores.shape[0] - sys_token_num
    num_select = round(beta * L)

    score_vis = total_scores[sys_token_num:sys_token_num + L]
    topk_scores, topk_idx = torch.topk(score_vis, k=num_select, dim=-1)
    order = torch.argsort(topk_idx)
    topk_idx = topk_idx[order]
    topk_scores = topk_scores[order]

    key_cache, value_cache = [], []
    for k, v in past_key_values:
        topk_idx = topk_idx.to(k.device)
        k_sys, v_sys = k[:, :, :sys_token_num, :], v[:, :, :sys_token_num, :]
        k_vis, v_vis = k[:, :, sys_token_num:sys_token_num + L, :], v[:, :, sys_token_num:sys_token_num + L, :]
        k_text, v_text = k[:, :, sys_token_num + L:, :], v[:, :, sys_token_num + L:, :]

        gather_idx = topk_idx.view(1, 1, -1, 1).expand(k_vis.size(0), k_vis.size(1), -1, k_vis.size(-1))
        k_sel = torch.gather(k_vis, 2, gather_idx)
        v_sel = torch.gather(v_vis, 2, gather_idx)

        key_cache.append(torch.cat([k_sys, k_sel, k_text], dim=2))
        value_cache.append(torch.cat([v_sys, v_sel, v_text], dim=2))

    return key_cache, value_cache, topk_scores, num_select


def prune_kv_layerwise(
    model,
    key_cache,
    value_cache,
    layer_id1,
    num_select,
    token_scores,
    compression_ratio=0.75,
    steepness=0.5,
    min_tokens=1,
    fusion_ratio=0.3,
    fusion_alpha=0.2,
    fusion_temperature=1.0,
    sys_token_num=15,
):
    L_new = num_select
    layer_tokens = monotonic_parabolic_allocation(
        layer_id1, int(L_new * 0.75), compression_ratio, steepness, min_tokens
    )

    k0 = key_cache[0]
    B, H, _, _ = k0.shape
    device = k0.device
    batch_ar = torch.arange(B, device=device).view(B, 1, 1)
    head_ar = torch.arange(H, device=device).view(1, H, 1)

    new_keys, new_values = [], []
    for i, (k, v) in enumerate(zip(key_cache, value_cache)):
        k_sys, v_sys = k[:, :, :sys_token_num, :], v[:, :, :sys_token_num, :]
        k_vis, v_vis = k[:, :, sys_token_num:sys_token_num + L_new, :], v[:, :, sys_token_num:sys_token_num + L_new, :]
        k_text, v_text = k[:, :, sys_token_num + L_new:, :], v[:, :, sys_token_num + L_new:, :]

        if i < layer_id1:
            keep = layer_tokens[i]
            _, topk_idx = torch.topk(token_scores, k=keep, dim=-1)
            topk_idx = topk_idx.view(1, 1, -1).expand(B, H, -1).to(k_vis.device)

            batch_idx = batch_ar.expand(B, H, keep).to(k_vis.device)
            head_idx = head_ar.expand(B, H, keep).to(k_vis.device)
            k_top = k_vis[batch_idx, head_idx, topk_idx]
            v_top = selective_attention_fusion(
                v_vis, topk_idx, token_scores, fusion_ratio, fusion_alpha, fusion_temperature
            )
            new_keys.append(torch.cat([k_sys, k_top, k_text], dim=2))
            new_values.append(torch.cat([v_sys, v_top, v_text], dim=2))
        else:
            new_keys.append(torch.cat([k_sys, k_text], dim=2))
            new_values.append(torch.cat([v_sys, v_text], dim=2))

    return create_compatible_cache(model, new_keys, new_values)
