"""Efficiency measurement for VisCache.

Provides three complementary metrics used throughout the paper:

* ``measure_generate`` / ``measure_decode`` -- wall-clock latency measured
  with CUDA events (prefill, per-token decode / TPOT, end-to-end total).
* ``kv_cache_memory_mb`` -- peak KV-cache memory footprint in megabytes.
* ``estimate_decode_flops`` -- a closed-form theoretical FLOPs estimate for the
  autoregressive decode stage (attention + MLP + LM head).

All functions are model-agnostic: they read architecture hyperparameters from
``model.config`` and operate on standard ``past_key_values`` caches.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def _cache_layers(past_key_values: Any) -> List[Any]:
    if hasattr(past_key_values, "layers"):
        return list(past_key_values.layers)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return [past_key_values[i] for i in range(len(past_key_values))]


def _layer_kv(layer: Any) -> (torch.Tensor, torch.Tensor):
    if isinstance(layer, (tuple, list)):
        return layer[0], layer[1]
    for kn, vn in (("keys", "values"), ("key_cache", "value_cache")):
        if hasattr(layer, kn) and hasattr(layer, vn):
            return getattr(layer, kn), getattr(layer, vn)
    raise TypeError("Unsupported cache layer type: %r" % (type(layer),))


def kv_cache_memory_mb(past_key_values: Any) -> float:
    """Total KV-cache memory in megabytes (all layers, keys + values)."""
    total_bytes = 0
    for layer in _cache_layers(past_key_values):
        k, v = _layer_kv(layer)
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024.0 ** 2)


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
def measure_generate(
    model: Any,
    generate_inputs: Dict[str, Any],
    max_new_tokens: int,
    device: Optional[torch.device] = None,
    repeats: int = 1,
) -> Dict[str, float]:
    """End-to-end generation latency (ms) for one ``model.generate`` call.

    Returns the averaged total / prefill / decode (TPOT) timings across
    ``repeats`` runs. The first ``warmup`` runs are discarded to stabilise
    measurement on CUDA.
    """
    device = device or next(model.parameters()).device
    gen_kwargs = dict(generate_inputs)
    gen_kwargs["max_new_tokens"] = max_new_tokens
    gen_kwargs["use_cache"] = True

    def _one_run():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        end.record()
        torch.cuda.synchronize()
        return out, start.elapsed_time(end)

    # warmup
    for _ in range(max(0, repeats - 1)):
        _one_run()
    total_ms = 0.0
    generated_tokens = 0
    for _ in range(repeats):
        out, ms = _one_run()
        total_ms += ms
        generated_tokens = int(out.shape[1] - gen_kwargs.get("input_ids").shape[1])
    total_ms /= max(repeats, 1)

    prompt_len = int(gen_kwargs.get("input_ids").shape[1])
    decode_tokens = max(generated_tokens - 1, 0)
    tpot_ms = total_ms / generated_tokens if generated_tokens > 0 else 0.0
    prefill_ms = total_ms - tpot_ms * decode_tokens if decode_tokens > 0 else total_ms
    return {
        "total_ms": total_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": tpot_ms,
        "generated_tokens": float(generated_tokens),
    }


def measure_decode(
    model: Any,
    past_key_values: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int = 1,
    repeats: int = 10,
) -> float:
    """Per-token decode latency (TPOT, ms) measured by stepping one token.

    Useful for isolating the decode-stage cost as a function of cache length.
    """
    device = next(model.parameters()).device
    past_len = max(
        int(_layer_kv(layer)[0].shape[2]) for layer in _cache_layers(past_key_values)
    )

    def _step():
        pos = torch.arange(past_len, past_len + 1, device=device)
        attn = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(
                input_ids=input_ids[:, -1:],
                attention_mask=attn,
                past_key_values=past_key_values,
                cache_position=pos,
                use_cache=True,
            )
        end.record()
        torch.cuda.synchronize()

    for _ in range(3):
        _step()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(repeats, 1)


# --------------------------------------------------------------------------- #
# Theoretical FLOPs
# --------------------------------------------------------------------------- #
def _candidate_text_configs(model: Any) -> List[Any]:
    candidates = []
    seen = set()

    def add(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        candidates.append(obj)

    config = getattr(model, "config", None)
    add(config)
    for name in ("text_config", "llm_config", "language_config"):
        if isinstance(config, dict):
            add(config.get(name))
        else:
            add(getattr(config, name, None))
    add(getattr(getattr(model, "language_model", None), "config", None))
    inner = getattr(model, "model", None)
    add(getattr(inner, "config", None))
    add(getattr(getattr(inner, "language_model", None), "config", None))
    return candidates


def _config_value(configs: Sequence[Any], names: Sequence[str], default=None):
    for cfg in configs:
        for name in names:
            if isinstance(cfg, dict):
                value = cfg.get(name)
            else:
                value = getattr(cfg, name, None)
            if value is not None:
                return value
    return default


def estimate_decode_flops(
    model: Any,
    initial_cache_tokens: int,
    generated_tokens: int,
    batch_size: int = 1,
    count_prefill_sample: bool = False,
    include_lm_head: bool = True,
) -> Dict[str, float]:
    """Closed-form FLOPs estimate for the autoregressive decode stage.

    Counts per-layer projection + MLP FLOPs and the quadratic attention FLOPs
    that scale with the (compressed) KV-cache length. Returns total, attention
    and per-stage token counts.
    """
    generated_tokens = int(generated_tokens)
    initial_cache_tokens = int(initial_cache_tokens)
    batch_size = int(batch_size)
    decode_tokens = generated_tokens if count_prefill_sample else max(generated_tokens - 1, 0)
    if decode_tokens <= 0:
        return {"total": 0.0, "attention": 0.0, "decode_tokens": float(decode_tokens)}

    configs = _candidate_text_configs(model)
    num_layers = _config_value(configs, ("num_hidden_layers", "num_layers", "n_layer"))
    hidden_size = _config_value(configs, ("hidden_size", "n_embd"))
    intermediate_size = _config_value(configs, ("intermediate_size", "ffn_hidden_size"))
    num_heads = _config_value(configs, ("num_attention_heads", "n_head"))
    num_kv_heads = _config_value(configs, ("num_key_value_heads", "num_kv_heads"), num_heads)
    vocab_size = _config_value(configs, ("vocab_size",), 0)
    if None in (num_layers, hidden_size, intermediate_size, num_heads):
        return {"total": 0.0, "attention": 0.0, "decode_tokens": float(decode_tokens)}

    num_layers = int(num_layers)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    num_heads = int(num_heads)
    num_kv_heads = int(num_kv_heads)
    vocab_size = int(vocab_size or 0)
    head_dim = hidden_size // max(num_heads, 1)
    kv_hidden = num_kv_heads * head_dim

    projection_flops = 2.0 * hidden_size * (hidden_size + kv_hidden + kv_hidden + hidden_size)
    mlp_flops = 6.0 * hidden_size * intermediate_size
    lm_head_flops = 2.0 * hidden_size * vocab_size if include_lm_head and vocab_size > 0 else 0.0

    first_kv_len = initial_cache_tokens if count_prefill_sample else initial_cache_tokens + 1
    sum_kv_len = decode_tokens * first_kv_len + decode_tokens * (decode_tokens - 1) / 2.0
    attention_flops = 4.0 * hidden_size * sum_kv_len * num_layers
    layer_static_flops = (projection_flops + mlp_flops) * num_layers
    total_flops = (layer_static_flops * decode_tokens + attention_flops + lm_head_flops * decode_tokens) * batch_size

    return {
        "total": float(total_flops),
        "attention": float(attention_flops * batch_size),
        "decode_tokens": float(decode_tokens),
    }


def format_flops(flops: float) -> str:
    return "%.4f TFLOPs (%.4e FLOPs)" % (float(flops) / 1e12, float(flops))


# --------------------------------------------------------------------------- #
# Aggregated report
# --------------------------------------------------------------------------- #
@dataclass
class EfficiencyReport:
    total_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    generated_tokens: int = 0
    kv_memory_mb: float = 0.0
    est_flops: float = 0.0
    est_flops_str: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "total_time_ms": self.total_time_ms,
            "prefill_time_ms": self.prefill_time_ms,
            "decode_time_ms": self.decode_time_ms,
            "generated_tokens": self.generated_tokens,
            "kv_memory_mb": self.kv_memory_mb,
            "est_flops": self.est_flops,
            "est_flops_str": self.est_flops_str,
        }
        d.update(self.extra)
        return d
