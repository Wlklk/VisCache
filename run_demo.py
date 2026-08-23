import torch

from smallandbig import SmallAndBigConfig, run


def main():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    clip_model, _ = clip.load("ViT-B/32", device=device)

    video_tensor = torch.rand(64, 3, 224, 224, device=device)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "dummy.mp4"},
                {"type": "text", "text": "Describe the content of this video in detail."},
            ],
        }
    ]

    config = SmallAndBigConfig(layer_id1=36)

    answer = run(
        model=model,
        processor=processor,
        messages=messages,
        video_tensor=video_tensor,
        clip_model=clip_model,
        device=device,
        config=config,
        max_new_tokens=128,
    )
    print(answer)


if __name__ == "__main__":
    main()
