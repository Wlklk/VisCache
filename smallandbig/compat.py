import torch


def find_transformer_layers(model):
    """Return the list of decoder layers for various MLLM backbones."""
    candidates = [
        "language_model.layers",   # Qwen2-VL / Qwen2.5-VL / Qwen3-VL, InternVL
        "model.layers",            # LLaVA, standard Llama-like
        "model.model.layers",      # some wrapper stacks
        "llm.layers",              # MiniCPM-V
    ]
    for path in candidates:
        mod = model
        ok = True
        for attr in path.split("."):
            if hasattr(mod, attr):
                mod = getattr(mod, attr)
            else:
                ok = False
                break
        if ok and isinstance(mod, (list, tuple)) and len(mod) > 0:
            return mod
    raise AttributeError(
        "Could not locate transformer layers. Tried: " + ", ".join(candidates)
    )


def resolve_num_layers(config, model):
    """Fill num_layers / layer_id1 from the model when not explicitly set."""
    if config.num_layers is None:
        config.num_layers = model.config.num_hidden_layers
    if config.layer_id1 is None:
        config.layer_id1 = round(0.75 * config.num_layers)
    return config


DEFAULT_VISION_TOKENS = [
    "<|image_pad|>", "<image>", "<|IMAGE_TOKEN|>", "<img>",
    "<|vision_pad|>", "<image_pad>", "<IMG_CONTEXT>",
]


def locate_vision_tokens(input_ids, tokenizer, vision_token_ids=None):
    """Return sorted global indices of visual placeholder tokens in input_ids.

    input_ids: 1-D LongTensor or list.
    tokenizer: tokenizer with convert_tokens_to_ids; used only when vision_token_ids is None.
    vision_token_ids: explicit set of token ids treated as visual; None -> auto-detect
        from common MLLM image placeholders.
    """
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)

    if vision_token_ids is None and tokenizer is not None:
        vision_token_ids = set()
        unk = getattr(tokenizer, "unk_token_id", None)
        for name in DEFAULT_VISION_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                vision_token_ids.add(tid)

    if not vision_token_ids:
        raise ValueError(
            "No visual token ids found. Pass vision_token_ids explicitly "
            "(e.g. {model.config.image_token_index})."
        )

    idx = [i for i, t in enumerate(ids) if t in vision_token_ids]
    if not idx:
        raise ValueError("Visual tokens not present in input_ids.")
    return torch.tensor(idx, dtype=torch.long)
