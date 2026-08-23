"""MVBench dataset loading and scoring utilities.

MVBench is a multiple-choice video benchmark: each sample carries a video, a
question, 4-5 candidate options (``candidates``) and the correct option letter
(``answer``). The 20 sub-tasks share an identical I/O contract, so they are
described by a single declarative registry (``MVBENCH_TASKS``) instead of 20
near-duplicate code branches.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple


# task_key -> (json_filename, relative_video_subdir, keyframe_mode)
# video_subdir is relative to ``--mvbench_root``; json lives under ``--json_root``.
MVBENCH_TASKS: Dict[str, Tuple[str, str, str]] = {
    "AS": ("action_sequence.json",        "star/Charades_v1_480", "mmr"),
    "AP": ("action_prediction.json",     "star/Charades_v1_480", "mmr"),
    "AA": ("action_antonym.json",        "star/Charades_v1_480", "mmr"),
    "FA": ("fine_grained_action.json",   "star/Charades_v1_480", "mmr"),
    "UA": ("unexpected_action.json",     "star/Charades_v1_480", "mmr"),
    "OE": ("object_existence.json",      "star/Charades_v1_480", "mmr"),
    "OI": ("object_interaction.json",    "star/Charades_v1_480", "mmr"),
    "OS": ("object_shuffle.json",        "star/Charades_v1_480", "mmr"),
    "MD": ("moving_direction.json",      "star/Charades_v1_480", "mmr"),
    "AL": ("action_localization.json",   "star/Charades_v1_480", "mmr"),
    "ST": ("state_change.json",          "star/Charades_v1_480", "mmr"),
    "AC": ("action_count.json",          "star/Charades_v1_480", "mmr"),
    "MC": ("moving_count.json",          "star/Charades_v1_480", "mmr"),
    "MA": ("moving_attribute.json",      "star/Charades_v1_480", "mmr"),
    "SC": ("scene_transition.json",      "star/Charades_v1_480", "mmr"),
    "FP": ("fine_grained_pose.json",     "something/v1-20",      "mmr"),
    "CO": ("counterfactual_inference.json", "clever",            "mmr"),
    "EN": ("egocentric_navigation.json", "egocentric",           "mmr"),
    "ER": ("episodic_reasoning.json",    "tvqa",                 "mmr"),
    "CI": ("character_order.json",       "moviech",              "mmr"),
}


@dataclass
class MVBenchSample:
    task: str
    video_id: str
    video_path: str
    question: str
    candidates: List[str]
    answer: str


def create_filename_mapping(video_root: str) -> Dict[str, str]:
    mapping = {}
    for root, _, files in os.walk(video_root):
        for name in files:
            mapping[re.sub(r"\s+", "", name).lower()] = os.path.join(root, name)
    return mapping


def load_mvbench_samples(
    task_key: str,
    json_root: str,
    video_root_base: str,
    limit: Optional[int] = None,
) -> List[MVBenchSample]:
    """Load all samples for one MVBench sub-task."""
    if task_key not in MVBENCH_TASKS:
        raise ValueError("Unknown MVBench task: %s" % task_key)
    json_name, video_subdir, _ = MVBENCH_TASKS[task_key]
    json_path = os.path.join(json_root, json_name)
    video_root = os.path.join(video_root_base, video_subdir)
    mapping = create_filename_mapping(video_root)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data[: limit]:
        video_id = item["video"]
        path = mapping.get(re.sub(r"\s+", "", video_id).lower())
        if path is None:
            print("Skipping %s: video %s not found under %s" % (task_key, video_id, video_root))
            continue
        samples.append(
            MVBenchSample(
                task=task_key,
                video_id=video_id,
                video_path=path,
                question=item["question"],
                candidates=list(item["candidates"]),
                answer=str(item["answer"]).strip().upper(),
            )
        )
    return samples


def build_mcq_prompt(sample: MVBenchSample) -> str:
    options = " ".join(sample.candidates)
    return "Question:" + sample.question + "\nOption:\n" + options + "\nOnly give the best option.\n"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def mcq_acc(answer: str, pred: str) -> int:
    period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    comma_strip = re.compile(r"(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_",
             "-", ">", "<", "@", "`", ",", "?", "!"]

    def process_punctuation(text: str) -> str:
        out = text
        for p in punct:
            if (p + " " in text or " " + p in text) or re.search(comma_strip, text):
                out = out.replace(p, "")
            else:
                out = out.replace(p, " ")
        return period_strip.sub("", out, re.UNICODE)

    def process(text: str) -> str:
        opt = re.match(r"^([A-E])\.\s*(.+)$", text.strip(), re.IGNORECASE)
        if opt:
            return opt.group(1).upper()
        text = text.replace("\n", " ").replace("\t", " ").strip()
        text = process_punctuation(text).strip("'\"").strip("()").strip().lower()
        letter = re.search(r"\b([A-E])\b", text, re.IGNORECASE)
        return letter.group(1).upper() if letter else text

    return 1 if process(pred) == process(answer) else 0


def normalize_text(text) -> str:
    text = text if isinstance(text, str) else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def rouge_l_f1(reference, prediction: str) -> float:
    ref = normalize_text(reference).split()
    pred = normalize_text(prediction).split()

    def lcs_len(a, b):
        if not a or not b:
            return 0
        prev = [0] * (len(b) + 1)
        for x in a:
            curr = [0]
            for j, y in enumerate(b, start=1):
                curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[-1]))
            prev = curr
        return prev[-1]

    lcs = lcs_len(ref, pred)
    if lcs == 0 or not ref or not pred:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) * 100.0
