"""
GazeLens proactive task evaluation.

Evaluates Gaze-Triggered Alert (GTA) and Object Appearance Alert (OAA) tasks.
These are binary Yes/No detection tasks evaluated at multiple time points per question.

Usage:
    python -m gazelens.eval_proactive \
        --mode gazelens \
        --qa_file dataset/qa/proactive_gaze_triggered_alert.json \
        --task_type GTA \
        --video_root dataset/videos/original_video \
        --fixation_root "pipeline/final_data/egoexo/metadata pipeline/final_data/egtea/metadata" \
        --checkpoint checkpoints/gazelens/best_model.pt \
        --output_file results/GazeLens/proactive_gta_gazelens.json
"""

import argparse
import gc
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from gazeqwen.hooks import GazeLensContext, register_gazelens_hooks, remove_gazelens_hooks
from gazeqwen.model import GazeLens, _CONFIG_KEYS
from gazeqwen.train import (
    compute_frame_times,
    create_video_clip,
    get_abcd_token_ids,
    pick_fps,
)
from gazeqwen.lora import apply_lora, load_lora_state_dict

logger = logging.getLogger(__name__)

SUPPORTED_MODES = ("no_gaze", "gazelens", "vjepa_coordpe_perlayer_lora")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: str) -> float:
    """Parse 'MM:SS' or 'HH:MM:SS' to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Cannot parse timestamp: {ts}")


def get_ab_token_ids(tokenizer) -> Tuple[int, int]:
    """Return token IDs for A (Yes) and B (No)."""
    abcd = get_abcd_token_ids(tokenizer)
    return abcd[0], abcd[1]  # A=Yes, B=No


def build_proactive_prompt(task_type: str, question: str) -> str:
    """Build an MCQA prompt for proactive tasks (A=Yes, B=No).

    Uses MCQA format to match the LoRA fine-tuning distribution.
    """
    if task_type == "GTA":
        context = (
            f"You are monitoring a user's gaze in a video stream. "
            f"The user's instruction is: \"{question}\""
        )
    else:  # OAA
        context = (
            f"You are monitoring a video stream. "
            f"The user's instruction is: \"{question}\""
        )
    return (
        f"Question: {context} Based on the video so far, should the alert be triggered?\n"
        f"Options:\n"
        f"A. Yes\n"
        f"B. No\n\n"
        f"The best option is:"
    )


def _find_fixation_csv(video_path: str, fixation_root: Optional[str]) -> Optional[str]:
    """Find fixation CSV for a video, same logic as data.py."""
    from pathlib import Path
    stem = Path(video_path).stem
    candidates = []
    if fixation_root:
        roots = re.split(r'[:\s]+', fixation_root.strip())
        for root in roots:
            if root:
                candidates.append(os.path.join(root, f"{stem}_fixation_filtered.csv"))
                candidates.append(os.path.join(root, stem, f"{stem}_fixation_filtered.csv"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_scanpath(csv_path: Optional[str], start_time: float, end_time: float) -> torch.Tensor:
    """Load and clip scanpath from CSV. Returns (N, 4) tensor or empty."""
    if csv_path is None or not os.path.exists(csv_path):
        return torch.zeros(0, 4)
    try:
        import csv
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row.get("timestamp", row.get("time", 0)))
                if start_time <= t <= end_time:
                    x = float(row.get("x", row.get("norm_x", 0)))
                    y = float(row.get("y", row.get("norm_y", 0)))
                    dur = float(row.get("duration", 0))
                    rows.append([t, x, y, dur])
        if rows:
            return torch.tensor(rows, dtype=torch.float32)
    except Exception as e:
        logger.warning("Failed to load scanpath from %s: %s", csv_path, e)
    return torch.zeros(0, 4)


# ---------------------------------------------------------------------------
# Forward pass (adapted from eval.py)
# ---------------------------------------------------------------------------

def _forward_one_binary(
    model,
    processor,
    mode: str,
    ctx: Optional[GazeLensContext],
    temp_video: str,
    scanpath: torch.Tensor,
    prompt: str,
    start_time: float,
    end_time: float,
    a_id: int,
    b_id: int,
    device: torch.device,
    fps: float,
    max_frames: int,
    vjepa_extractor,
) -> Optional[bool]:
    """Forward pass returning True (Yes) or False (No), or None on failure."""
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{temp_video}",
                    "fps": fps,
                    "max_frames": max_frames,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Frame times for hooks
    frame_times: List[float] = []
    video_grid_thw = inputs.get("video_grid_thw")
    if video_grid_thw is not None and len(video_grid_thw) > 0:
        T = int(video_grid_thw[0][0])
        frame_times = compute_frame_times(start_time, T, fps)

    scanpath_dev = scanpath.to(device)

    # V-JEPA backbone features
    backbone_features = None
    if video_grid_thw is not None and len(video_grid_thw) > 0:
        T = int(video_grid_thw[0][0])
        llm_H_est = int(video_grid_thw[0][1]) // 2
        llm_W_est = int(video_grid_thw[0][2]) // 2
        if video_inputs is not None and len(video_inputs) > 0 and vjepa_extractor is not None:
            raw_frames = video_inputs[0]
            backbone_features = vjepa_extractor.extract_from_raw_frames(
                raw_frames, T, llm_H_est, llm_W_est,
            )

    del image_inputs, video_inputs
    gc.collect()

    torch.cuda.empty_cache()
    with torch.no_grad():
        if mode != "no_gaze" and ctx is not None:
            with ctx.active(scanpath_dev, frame_times, backbone_features=backbone_features):
                output = model(**inputs)
        else:
            output = model(**inputs)

    # Extract A(Yes)/B(No) prediction from ABCD logits
    logits_at_ans = output.logits[0, -1, :]
    a_logit = logits_at_ans[a_id].item()
    b_logit = logits_at_ans[b_id].item()
    return a_logit > b_logit


def eval_one_test_point(
    model,
    processor,
    mode: str,
    ctx: Optional[GazeLensContext],
    video_path: str,
    scanpath: torch.Tensor,
    prompt: str,
    start_time: float,
    end_time: float,
    a_id: int,
    b_id: int,
    device: torch.device,
    max_frames: int = 8,
    vjepa_extractor=None,
    oom_cache: Optional[Dict[str, int]] = None,
) -> Optional[bool]:
    """Evaluate one test point with OOM retry. Returns True/False/None."""
    duration = end_time - start_time
    fps = pick_fps(duration)
    temp_video = create_video_clip(video_path, start_time, end_time, target_fps=fps)

    if temp_video is None:
        return None

    try:
        current_frames = max_frames
        if oom_cache is not None and video_path in oom_cache:
            current_frames = oom_cache[video_path]
            if current_frames == 0:
                return None

        while True:
            try:
                result = _forward_one_binary(
                    model, processor, mode, ctx, temp_video, scanpath,
                    prompt, start_time, end_time, a_id, b_id,
                    device, fps, current_frames, vjepa_extractor,
                )
                if oom_cache is not None:
                    oom_cache[video_path] = current_frames
                return result
            except torch.cuda.OutOfMemoryError:
                reduced = max(1, current_frames // 2)
                if reduced >= current_frames:
                    logger.warning("CUDA OOM for %s at min frames, skipping", video_path)
                    if oom_cache is not None:
                        oom_cache[video_path] = 0
                    return None
                logger.warning("CUDA OOM for %s frames=%d, retrying %d", video_path, current_frames, reduced)
                gc.collect()
                torch.cuda.empty_cache()
                current_frames = reduced
    except Exception as exc:
        logger.warning("Exception for %s: %s", video_path, exc)
        return None
    finally:
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def load_proactive_qa(qa_file: str, split_file: Optional[str], split: Optional[str]) -> List[Dict]:
    """Load proactive QA entries, optionally filtered by split."""
    with open(qa_file) as f:
        data = json.load(f)

    if split_file and split:
        with open(split_file) as f:
            splits = json.load(f)
        allowed = set(splits[split])
        data = [q for q in data if q["video_path"] in allowed]

    return data


def run_proactive_eval(
    model,
    processor,
    mode: str,
    ctx: Optional[GazeLensContext],
    qa_data: List[Dict],
    task_type: str,
    video_root: str,
    fixation_root: Optional[str],
    a_id: int,
    b_id: int,
    device: torch.device,
    *,
    output_file: Optional[str] = None,
    save_every: int = 10,
    max_frames: int = 8,
    vjepa_extractor=None,
    skip_keys: Optional[Set[tuple]] = None,
    prior_results: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Run proactive eval over all QA entries and test points."""
    all_results = list(prior_results) if prior_results else []
    _skip = skip_keys or set()
    newly_processed = 0
    n_skipped = 0
    oom_cache: Dict[str, int] = {}

    # Counters
    tp = fp = tn = fn = 0
    # Seed from prior
    if prior_results:
        for r in prior_results:
            pred = r["pred_alert"]
            gt = r["ground_truth"]
            if pred and gt: tp += 1
            elif pred and not gt: fp += 1
            elif not pred and gt: fn += 1
            else: tn += 1

    total_points = sum(
        len(ti) for q in qa_data for qi in q["questions"] for ti in [qi["test_info"]]
    )
    logger.info("Total test points: %d (from %d questions)", total_points, len(qa_data))

    for entry in qa_data:
        video_name = entry["video_path"]
        video_path = os.path.join(video_root, os.path.basename(video_name))
        if not os.path.exists(video_path):
            logger.warning("Video not found: %s", video_path)
            continue

        csv_path = _find_fixation_csv(video_path, fixation_root)

        for qi in entry["questions"]:
            question_text = qi["question"]
            prompt = build_proactive_prompt(task_type, question_text)

            for ti in qi["test_info"]:
                realtime = ti["realtime"]
                ground_truth = ti["type"]  # 0 or 1

                # Clip range: use explicit field if present, else derive from realtime
                if "input_video_clip" in ti:
                    clip_range = ti["input_video_clip"]
                else:
                    # Streaming: clip starts at 0, ends at realtime (seconds)
                    end_sec = _parse_timestamp(realtime)
                    clip_range = [0, int(end_sec)]

                # Resume key
                rkey = (video_name, question_text, realtime)
                if rkey in _skip:
                    continue

                # Skip if video already OOMed at minimum
                if oom_cache.get(video_path) == 0:
                    n_skipped += 1
                    continue

                start_time = float(clip_range[0])
                end_time = float(clip_range[1])
                if end_time <= start_time:
                    end_time = start_time + 1.0

                # Load scanpath for this clip window
                scanpath = _load_scanpath(csv_path, start_time, end_time)

                pred = eval_one_test_point(
                    model=model,
                    processor=processor,
                    mode=mode,
                    ctx=ctx,
                    video_path=video_path,
                    scanpath=scanpath,
                    prompt=prompt,
                    start_time=start_time,
                    end_time=end_time,
                    a_id=a_id,
                    b_id=b_id,
                    device=device,
                    max_frames=max_frames,
                    vjepa_extractor=vjepa_extractor,
                    oom_cache=oom_cache,
                )

                if pred is None:
                    n_skipped += 1
                    continue

                gt_bool = bool(ground_truth)
                if pred and gt_bool: tp += 1
                elif pred and not gt_bool: fp += 1
                elif not pred and gt_bool: fn += 1
                else: tn += 1

                all_results.append({
                    "video_path": video_name,
                    "task_type": task_type,
                    "question_text": question_text,
                    "realtime": realtime,
                    "ground_truth": ground_truth,
                    "pred_alert": pred,
                    "is_correct": pred == gt_bool,
                    "clip_range": clip_range,
                })

                newly_processed += 1
                if output_file and save_every > 0 and newly_processed % save_every == 0:
                    result_dict = _build_proactive_result(
                        task_type, mode, all_results, tp, fp, tn, fn, n_skipped
                    )
                    _save_json(output_file, result_dict)
                    logger.info(
                        "Intermediate save: %d processed, %d total -> %s",
                        newly_processed, len(all_results), output_file,
                    )

    return _build_proactive_result(task_type, mode, all_results, tp, fp, tn, fn, n_skipped)


def _build_proactive_result(
    task_type: str, mode: str, results: List[Dict],
    tp: int, fp: int, tn: int, fn: int, n_skipped: int,
) -> Dict[str, Any]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0.0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "task_type": task_type,
        "mode": mode,
        "metrics": {
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
        },
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_test_points": total,
        "n_skipped": n_skipped,
        "results": results,
    }


def _save_json(path: str, data: Dict) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate GazeLens on proactive tasks (GTA/OAA)")
    p.add_argument("--mode", required=True, choices=list(SUPPORTED_MODES))
    p.add_argument("--qa_file", required=True, help="Proactive QA JSON file")
    p.add_argument("--task_type", required=True, choices=["GTA", "OAA"])
    p.add_argument("--video_root", required=True)
    p.add_argument("--fixation_root", default=None)
    p.add_argument("--output_file", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--vjepa_checkpoint", default=None)
    p.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--split_file", default=None)
    p.add_argument("--split", default=None, choices=["train", "val", "test"])
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--n_samples_max", type=int, default=None)
    p.add_argument("--no_lora", action="store_true",
                   help="Skip LoRA adapter (use GazeLens hooks only, base LLM output)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_every", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    mode = args.mode
    is_gaze_mode = mode != "no_gaze"

    if is_gaze_mode and not args.checkpoint:
        raise ValueError(f"--checkpoint required for {mode} mode")

    # Resumption
    prior_results = None
    skip_keys = None
    if args.resume and os.path.exists(args.output_file):
        with open(args.output_file) as f:
            prev = json.load(f)
        prior = prev.get("results", [])
        if prior:
            prior_results = prior
            skip_keys = {
                (r["video_path"], r["question_text"], r["realtime"])
                for r in prior
            }
            logger.info("Resume: %d prior results", len(prior))

    # Load model
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    logger.info("Loading %s (eager attention) ...", args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    processor = AutoProcessor.from_pretrained(args.model_path)
    device = next(model.parameters()).device
    a_id, b_id = get_ab_token_ids(processor.tokenizer)
    logger.info("A(Yes)/B(No) token IDs: %d / %d", a_id, b_id)

    # LoRA
    if args.checkpoint:
        _ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(_ckpt, dict) and "lora_state_dict" in _ckpt:
            lora_cfg = _ckpt.get("lora_config", {"rank": 8, "alpha": 16.0})
            apply_lora(model, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"])
            load_lora_state_dict(model, _ckpt["lora_state_dict"])
            logger.info("LoRA loaded: rank=%d alpha=%.1f", lora_cfg["rank"], lora_cfg["alpha"])
        del _ckpt

    # GazeLens hooks
    ctx = None
    vjepa_extractor = None
    if is_gaze_mode:
        raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            ckpt_config = raw.get("config", {})
            state = raw["state_dict"]
        else:
            ckpt_config = {}
            state = raw
        ckpt_config.pop("version", None)
        valid_keys = set(_CONFIG_KEYS)
        valid_keys.add("amplitude_init")
        ckpt_config = {k: v for k, v in ckpt_config.items() if k in valid_keys}

        gazelens_module = GazeLens(**ckpt_config).to(device)
        gazelens_module.load_state_dict(state)
        gazelens_module.eval()
        ctx = GazeLensContext(gazelens_module)
        register_gazelens_hooks(model, ctx)

        from gazeqwen.vjepa_features import VJEPAFeatureExtractor
        vjepa_extractor = VJEPAFeatureExtractor(
            device, checkpoint_path=getattr(args, "vjepa_checkpoint", None)
        )
        logger.info("GazeLens: checkpoint=%s config=%s", args.checkpoint, gazelens_module.get_config())
        del raw

    # Load QA data
    qa_data = load_proactive_qa(args.qa_file, args.split_file, args.split)
    if args.n_samples_max and len(qa_data) > args.n_samples_max:
        qa_data = qa_data[:args.n_samples_max]
    logger.info("QA entries: %d, task=%s, mode=%s", len(qa_data), args.task_type, mode)

    # Run eval
    results = run_proactive_eval(
        model=model,
        processor=processor,
        mode=mode,
        ctx=ctx,
        qa_data=qa_data,
        task_type=args.task_type,
        video_root=args.video_root,
        fixation_root=args.fixation_root,
        a_id=a_id,
        b_id=b_id,
        device=device,
        output_file=args.output_file,
        save_every=args.save_every,
        max_frames=args.max_frames,
        vjepa_extractor=vjepa_extractor,
        skip_keys=skip_keys,
        prior_results=prior_results,
    )

    _save_json(args.output_file, results)
    logger.info("Results saved to %s", args.output_file)

    # Print summary
    m = results["metrics"]
    c = results["confusion"]
    print(f"\n=== {args.task_type} ({mode}) ===")
    print(f"  Accuracy:  {m['accuracy']:.1f}%")
    print(f"  Precision: {m['precision']:.1f}%")
    print(f"  Recall:    {m['recall']:.1f}%")
    print(f"  F1:        {m['f1']:.1f}%")
    print(f"  TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']}")
    print(f"  Test points: {results['n_test_points']}, Skipped: {results['n_skipped']}")

    if ctx is not None:
        remove_gazelens_hooks(model)


if __name__ == "__main__":
    main()
