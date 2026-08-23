# SmallAndBig — Attention-Guided KV Cache Compression for Video LLMs

A clean reimplementation of the "Small and Big" KV-cache compression pipeline for video
multimodal LLMs (e.g. Qwen2.5-VL). Visual tokens are pruned in two stages — a **global
attention-based selection** that keeps the most informative visual tokens, followed by a
**layerwise progressive prune** where shallow layers keep many tokens ("big") and deep
layers keep few or none ("small"). Dropped tokens are fused into the survivors via a
selective attention fusion, so information is preserved without growing the cache.

## Method

Given a video and a prompt, the pipeline runs in four steps:

1. **Keyframe selection** (`keyframe.py`)
   CLIP encodes every frame and the text prompt. Frames are scored by MMR
   (relevance to text − diversity against already-selected frames) and the top
   `alpha · T` frames are kept, shrinking the video before it ever reaches the LLM.

2. **Prefill with attention accumulation** (`pipeline.py`)
   A forward hook on every decoder-layer `self_attn` (auto-located for Qwen-VL /
   LLaVA / InternVL / MiniCPM-V) sums the mean attention maps across all layers into
   one per-token importance score.

3. **Global visual-token selection** (`compression.select_visual_tokens`)
   Visual tokens are auto-located by their placeholder ids (e.g. `<|image_pad|>`),
   independent of how many system/text tokens precede them. The top `beta · V` visual
   tokens by accumulated attention are kept; all non-visual tokens are preserved.

4. **Layerwise progressive prune + fusion** (`compression.prune_kv_layerwise`)
   A monotonic parabola assigns each of the first `layer_id1` layers a decreasing
   keep-count; within each layer the top-scoring surviving tokens are kept and the
   dropped ones are fused into them (`selective_attention_fusion`). Layers
   `>= layer_id1` discard all visual tokens entirely ("small").

The compressed `past_key_values` is then fed straight into `model.generate`.

## Multi-model support

The pipeline is model-agnostic — nothing is hard-coded to Qwen2.5-VL:

- **Transformer layers** are auto-located by trying `language_model.layers`
  (Qwen-VL / InternVL), `model.layers` (LLaVA), `model.model.layers`, then
  `llm.layers` (MiniCPM-V). Override via `find_transformer_layers`.
- **Number of layers** is read from `model.config.num_hidden_layers` when
  `num_layers=None`; `layer_id1` then defaults to `round(0.75 · num_layers)`.
- **Visual tokens** are found by their placeholder ids. When `vision_token_ids`
  is `None`, common image placeholders are auto-detected
  (`<|image_pad|>`, `<image>`, `<|IMAGE_TOKEN|>`, `<img>`, …). Pass an explicit
  set (e.g. `{model.config.image_token_index}`) for unusual tokenizers.
- **Input building** differs per architecture, so `run()` accepts a `build_inputs`
  callback. The default handles Qwen2.5-VL; LLaVA / others supply their own.

## Project layout

```
SmallAndBig/                 # repository root
├── smallandbig/             # the package
│   ├── config.py        # SmallAndBigConfig: all hyperparameters
│   ├── keyframe.py      # CLIP + MMR keyframe selection
│   ├── compression.py   # global token selection, layerwise prune, fusion
│   ├── pipeline.py      # prefill hooks + compress + generate
│   └── __init__.py
├── run_demo.py          # minimal end-to-end usage example
├── requirements.txt
└── README.md
```

## Hyperparameters (`SmallAndBigConfig`)

| Param               | Default | Meaning                                                  |
|---------------------|---------|----------------------------------------------------------|
| `alpha`             | 0.5     | fraction of frames kept by keyframe selection            |
| `mmr_lambda`        | 0.7     | MMR trade-off: relevance vs. diversity                   |
| `beta`              | 0.67    | fraction of visual tokens kept after global selection    |
| `num_layers`        | auto    | total model layers; None -> read from model.config.num_hidden_layers |
| `layer_id1`         | auto    | layers 0..layer_id1-1 keep tokens; None -> round(0.75·num_layers): 27 (3B) / 48 (32B) |
| `compression_ratio` | 0.75    | target keep-ratio used by the parabolic allocation      |
| `alloc_steepness`   | 0.5     | how fast the per-layer keep-count decreases             |
| `min_tokens`        | 1       | lower bound on kept tokens per layer                     |
| `fusion_ratio`      | 0.3     | fraction of dropped tokens fused into each survivor      |
| `fusion_alpha`      | 0.2     | fusion strength                                          |
| `fusion_temperature`| 1.0     | softmax temperature over dropped-token attention        |
| `vision_token_ids`  | auto    | token ids marking visual tokens; None -> auto-detect common placeholders |

## Usage

```python
from smallandbig import SmallAndBigConfig, run

config = SmallAndBigConfig()   # num_layers / layer_id1 auto-derived from the model

answer = run(
    model=model,                # Qwen2.5-VL
    processor=processor,
    messages=messages,          # chat template with the video + prompt
    video_tensor=video_tensor,  # raw frames [T, 3, H, W]
    clip_model=clip_model,      # CLIP for keyframe selection
    device=device,
    config=config,
    max_new_tokens=128,
)
```

For non-Qwen processors, pass a `build_inputs` callback that returns the model
inputs (the clip-based keyframe selection still runs upstream):

```python
def llava_build_inputs(processor, messages, video_tensor, device):
    prompt = next(c["text"] for c in messages[0]["content"] if c["type"] == "text")
    inputs = processor(images=video_tensor, text=prompt, return_tensors="pt")
    return inputs.to(device)

run(..., build_inputs=llava_build_inputs)   # vision_token_ids auto-detected
```

See `run_demo.py` for a complete example.
