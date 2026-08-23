"""MVBench evaluation for VisCache (Qwen2.5-VL family).

For every MVBench sub-task the pipeline is identical: compress the video into
key-frames, prefill while accumulating per-token attention, prune the visual KV
cache with VisCache, decode the answer, and score it as multiple-choice. The
only thing that changes per task is *which* json/video folder to read, which is
supplied declaratively by ``MVBENCH_TASKS`` in :mod:`benchmarks.loader`.

Run::

    python -m benchmarks.mvbench \
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \
        --mvbench_root /data/MVBench --json_root /data/MVBench/JSON \
        --clip_model ViT-B/32 --tasks AS AP AC
"""

import argparse
import os
import time

import torch
from tqdm import tqdm

from smallandbig import (
    SmallAndBigConfig,
    compress_kv_cache,
    locate_vision_tokens,
    register_attention_hooks,
    resolve_num_layers,
    select_keyframes,
)
from smallandbig.keyframe import encode_text

from .loader import MVBENCH_TASKS, build_mcq_prompt, load_mvbench_samples, mcq_acc


def _import_process_vision_info():
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        try:
            from qwen_vl_utils_ours import process_vision_info
        except Exception as exc:
            raise RuntimeError("Need qwen_vl_utils.process_vision_info.") from exc
    return process_vision_info


def run_sample(model, processor, clip_model, device, config, sample, args):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample.video_path,
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 256 * 28 * 28,
                    "total_pixels": 20480 * 28 * 28,
                },
                {"type": "text", "text": build_mcq_prompt(sample)},
            ],
        }
    ]
    prompt = build_mcq_prompt(sample)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    process_vision_info = _import_process_vision_info()
    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs is None or len(video_inputs) == 0:
        raise RuntimeError("No video tensor produced by process_vision_info.")

    frames = video_inputs[0]
    if args.frame_stride > 1:
        frames = frames[:: args.frame_stride]
    if args.max_frames and frames.shape[0] > args.max_frames:
        idx = torch.linspace(0, frames.shape[0] - 1, args.max_frames).long()
        frames = frames[idx]

    keyframes = select_keyframes(prompt, clip_model, frames, device, config.alpha, config.mmr_lambda)
    video_inputs[0] = frames[keyframes]

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=args.fps,
        padding=True,
        return_tensors="pt",
    ).to(device)

    get_scores, remove_hooks = register_attention_hooks(model)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=True, return_dict=True)
    total_scores = get_scores()
    remove_hooks()

    tokenizer = getattr(processor, "tokenizer", None) or getattr(model, "tokenizer", None)
    vision_idx = locate_vision_tokens(inputs["input_ids"][0], tokenizer, config.vision_token_ids)
    resolve_num_layers(config, model)

    new_cache = compress_kv_cache(model, outputs.past_key_values, total_scores, vision_idx, config)
    del outputs

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            past_key_values=new_cache,
        )
    out = generated[0][inputs["input_ids"].shape[1]:]
    pred = processor.decode(out, skip_special_tokens=True).strip()
    return pred


def evaluate(model, processor, clip_model, device, config, args):
    task_keys = args.tasks or list(MVBENCH_TASKS.keys())
    summary_lines = []
    grand_correct, grand_total = 0, 0

    for task in task_keys:
        samples = load_mvbench_samples(task, args.json_root, args.mvbench_root, args.limit)
        if not samples:
            print("No samples for task %s, skipping." % task)
            continue
        correct, total = 0, 0
        for sample in tqdm(samples, desc=task):
            try:
                pred = run_sample(model, processor, clip_model, device, config, sample, args)
                correct += mcq_acc(sample.answer, pred)
                total += 1
            except Exception as e:
                print("Error on %s/%s: %s" % (task, sample.video_id, e))
                continue
        acc = correct / total if total else 0.0
        grand_correct += correct
        grand_total += total
        line = "%s: %.2f%% (%d/%d)" % (task, 100.0 * acc, correct, total)
        summary_lines.append(line)
        print(line)

    overall = grand_correct / grand_total if grand_total else 0.0
    summary_lines.append("OVERALL: %.2f%% (%d/%d)" % (100.0 * overall, grand_correct, grand_total))

    model_name = os.path.basename(args.model_path.rstrip("/"))
    out_path = args.output or ("mvbench_results_ours_%s.txt" % model_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("Saved ->", out_path)
    return overall


def parse_args():
    p = argparse.ArgumentParser(description="MVBench evaluation for VisCache")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--clip_model", type=str, default="ViT-B/32")
    p.add_argument("--mvbench_root", type=str, required=True, help="root containing task video folders")
    p.add_argument("--json_root", type=str, required=True, help="folder with the MVBench *.json files")
    p.add_argument("--tasks", nargs="*", default=None, help="task keys, e.g. AS AP AC (default: all 20)")
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--frame_stride", type=int, default=4)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--beta", type=float, default=0.67)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--num_layers", type=int, default=None, help="model layers; auto-detected if omitted")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import clip as openclip

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto", attn_implementation="eager"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    device = next(model.parameters()).device
    clip_model, _ = openclip.load(args.clip_model, device=str(device))

    config = SmallAndBigConfig(beta=args.beta, alpha=args.alpha, num_layers=args.num_layers)
    evaluate(model, processor, clip_model, device, config, args)


if __name__ == "__main__":
    main()
