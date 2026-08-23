import time

import torch
import torch.nn.functional as F
from torchvision import transforms

_CLIP_NORMALIZE = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26579941, 0.26158784),
)


def encode_text(text, clip_model, device):
    if isinstance(text, str):
        text = [text]
    tokens = clip_model.tokenize(text, truncate=True).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def _mmr_select(frame_feats, num_keyframes, text_feat, lambda_param):
    N = frame_feats.shape[0]
    if N <= num_keyframes:
        return list(range(N))

    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    sim_to_text = F.cosine_similarity(frame_feats, text_feat)

    selected = [0, N - 1]
    while len(selected) < num_keyframes:
        best_score, best_idx = -1e9, -1
        for i in range(N):
            if i in selected:
                continue
            relevance = sim_to_text[i].item()
            redundancy = max(
                F.cosine_similarity(
                    frame_feats[i].unsqueeze(0), frame_feats[j].unsqueeze(0)
                ).item()
                for j in selected
            )
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
    return sorted(selected)


def select_keyframes(text, clip_model, video_tensor, device, alpha=0.5, mmr_lambda=0.7):
    num_frames = video_tensor.shape[0]
    num_keyframes = max(2, round(num_frames * alpha))

    resize = transforms.Resize((224, 224))
    frames = torch.stack([resize(img) for img in video_tensor]).to(device, dtype=torch.float16)
    frames = _CLIP_NORMALIZE(frames.float()).to(torch.float16)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        feats = clip_model.encode_image(frames)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    text_feat = encode_text(text, clip_model, device)
    if num_frames <= num_keyframes:
        return list(range(num_frames))
    return _mmr_select(feats, num_keyframes, text_feat, mmr_lambda)
