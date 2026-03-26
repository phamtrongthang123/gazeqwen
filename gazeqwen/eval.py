"""
GazeLens evaluation script.

Supports gazelens mode (V-JEPA + Coord-PE + Per-layer + LoRA) and no_gaze baseline.
All modes use eager attention for hook-based bias injection compatibility.
"""

import argparse
import gc
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader

from gazeqwen.data import GazeLensDataset, collate_fn
from gazeqwen.hooks import GazeLensContext, register_gazelens_hooks, remove_gazelens_hooks
from gazeqwen.model import GazeLens, _CONFIG_KEYS
from gazeqwen.train import (
    build_mcqa_prompt,
    compute_frame_times,
    create_video_clip,
    get_abcd_token_ids,
    pick_fps,
)
from gazeqwen.lora import apply_lora, load_lora_state_dict

logger = logging.getLogger(__name__)

SUPPORTED_MODES = ("no_gaze", "gazelens", "gazeqwen", "vjepa_coordpe_perlayer_lora")


# ---------------------------------------------------------------------------
# Per-sample forward pass (mode-aware)
# ---------------------------------------------------------------------------

def _forward_one(
    model,
    processor,
    mode: str,
    ctx: Optional[GazeLensContext],
    temp_video: str,
    scanpath: torch.Tensor,
    question_text: str,
    options: List[str],
    answer_idx: int,
    start_time: float,
    end_time: float,
    abcd_token_ids: List[int],
    device: torch.device,
    fps: float,
    max_frames: int,
    vjepa_extractor,
) -> Tuple[Optional[str], bool]:
    """Inner forward pass for a single sample. May raise on CUDA OOM."""
    from qwen_vl_utils import process_vision_info

    prompt = build_mcqa_prompt(question_text, options)

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

    # ---- Frame times (needed by hook modes) ----
    frame_times: List[float] = []
    video_grid_thw = inputs.get("video_grid_thw")
    if video_grid_thw is not None and len(video_grid_thw) > 0:
        T = int(video_grid_thw[0][0])
        frame_times = compute_frame_times(start_time, T, fps)

    scanpath_dev = scanpath.to(device)

    # ---- V-JEPA backbone features ----
    backbone_features = None
    if video_grid_thw is not None and len(video_grid_thw) > 0:
        T = int(video_grid_thw[0][0])
        llm_H_est = int(video_grid_thw[0][1]) // 2
        llm_W_est = int(video_grid_thw[0][2]) // 2
        if video_inputs is not None and len(video_inputs) > 0 and vjepa_extractor is not None:
            raw_frames = video_inputs[0]  # (N_frames, C, H, W) float [0,255]
            backbone_features = vjepa_extractor.extract_from_raw_frames(
                raw_frames, T, llm_H_est, llm_W_est,
            )

    # Free CPU/GPU memory no longer needed before forward pass
    del image_inputs, video_inputs
    gc.collect()

    # ---- Forward pass ----
    torch.cuda.empty_cache()
    with torch.no_grad():
        if mode != "no_gaze" and ctx is not None:
            with ctx.active(scanpath_dev, frame_times, backbone_features=backbone_features):
                output = model(**inputs)
        else:
            output = model(**inputs)

    # ---- Extract prediction from ABCD logits ----
    logits_at_ans = output.logits[0, -1, :]   # (vocab_size,)
    abcd_ids_t = torch.tensor(
        abcd_token_ids, device=logits_at_ans.device, dtype=torch.long
    )
    abcd_logits = logits_at_ans[abcd_ids_t]   # (4,)
    pred_idx = int(abcd_logits.argmax().item())
    pred_letter = chr(65 + pred_idx)
    is_correct = pred_idx == answer_idx

    return pred_letter, is_correct


def eval_one_sample(
    model,
    processor,
    mode: str,
    ctx: Optional[GazeLensContext],
    video_path: str,
    scanpath: torch.Tensor,
    question_text: str,
    options: List[str],
    answer_idx: int,
    start_time: float,
    end_time: float,
    abcd_token_ids: List[int],
    device: torch.device,
    *,
    max_frames: int = 8,
    vjepa_extractor=None,
    oom_cache: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[str], bool]:
    """Run one forward pass with OOM retry. Returns (pred_letter, is_correct).

    If *oom_cache* is provided (dict mapping video_path → last successful
    max_frames), the retry loop starts from the cached value instead of
    *max_frames*, avoiding redundant OOM retries for the same video.
    The cache is updated in-place after a successful forward pass.
    """
    duration = end_time - start_time
    fps = pick_fps(duration)

    temp_video = create_video_clip(video_path, start_time, end_time, target_fps=fps)

    if temp_video is None:
        return None, False

    try:
        # Use cached max_frames for this video if available
        if oom_cache is not None and video_path in oom_cache:
            current_frames = oom_cache[video_path]
        else:
            current_frames = max_frames
        while True:
            try:
                result = _forward_one(
                    model, processor, mode, ctx, temp_video, scanpath,
                    question_text, options, answer_idx, start_time, end_time,
                    abcd_token_ids, device, fps, current_frames, vjepa_extractor,
                )
                # Success — update cache so next QA for this video starts here
                if oom_cache is not None:
                    oom_cache[video_path] = current_frames
                return result
            except torch.cuda.OutOfMemoryError:
                reduced = max(1, current_frames // 2)
                if reduced >= current_frames:
                    # Already at minimum, give up
                    logger.warning(
                        "CUDA OOM for %s with max_frames=%d (minimum), skipping",
                        video_path, current_frames,
                    )
                    # Cache the failure so we skip retries next time too
                    if oom_cache is not None:
                        oom_cache[video_path] = 0
                    return None, False
                logger.warning(
                    "CUDA OOM for %s with max_frames=%d, retrying with %d",
                    video_path, current_frames, reduced,
                )
                gc.collect()
                torch.cuda.empty_cache()
                current_frames = reduced

    except Exception as exc:
        logger.warning("Exception in eval_one_sample for %s: %s", video_path, exc)
        return None, False

    finally:
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)


# ---------------------------------------------------------------------------
# Resumption helpers
# ---------------------------------------------------------------------------

def _resume_key(result_entry: Dict[str, Any]) -> tuple:
    """Build the skip-key from a saved result entry (video_path, task_type, question_text)."""
    return (
        result_entry.get("video_path", ""),
        result_entry.get("task_type", ""),
        result_entry.get("question_text", ""),
    )


def _build_result_dict(
    mode: str,
    all_results: List[Dict[str, Any]],
    task_correct: Dict[str, int],
    task_total: Dict[str, int],
    n_skipped: int = 0,
) -> Dict[str, Any]:
    """Assemble the standard eval result dict from accumulated counters."""
    accuracy = {
        k: round(task_correct[k] / task_total[k] * 100.0, 2) if task_total[k] > 0 else 0.0
        for k in task_total
    }
    return {
        "mode": mode,
        "accuracy": accuracy,
        "n_samples": dict(task_total),
        "n_skipped": n_skipped,
        "results": all_results,
    }


def _save_result_json(path: str, result_dict: Dict[str, Any]) -> None:
    """Atomically write result_dict to path (creates parent dirs if needed)."""
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result_dict, f, indent=2)


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model,
    processor,
    mode: str,
    ctx: Optional[GazeLensContext],
    dataset: GazeLensDataset,
    abcd_token_ids: List[int],
    device: torch.device,
    *,
    skip_keys: Optional[Set[tuple]] = None,
    prior_results: Optional[List[Dict[str, Any]]] = None,
    output_file: Optional[str] = None,
    save_every: int = 10,
    max_frames: int = 8,
    vjepa_extractor=None,
) -> dict:
    """Evaluate model over all samples. Returns dict with mode, accuracy, n_samples, results."""
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Seed from prior run if resuming
    all_results: List[Dict[str, Any]] = list(prior_results) if prior_results else []
    task_correct: Dict[str, int] = defaultdict(int)
    task_total: Dict[str, int] = defaultdict(int)
    if prior_results:
        for r in prior_results:
            t = r.get("task_type", "")
            task_total[t] += 1
            task_total["all"] += 1
            if r.get("is_correct"):
                task_correct[t] += 1
                task_correct["all"] += 1
        logger.info("Resuming: loaded %d prior results", len(prior_results))

    _skip = skip_keys or set()
    newly_processed = 0
    n_skipped = 0
    oom_cache: Dict[str, int] = {}  # video_path → last successful max_frames

    for batch in loader:
        video_path = batch["video_paths"][0]
        if not os.path.exists(video_path):
            logger.warning("Video not found, skipping: %s", video_path)
            continue

        task_type = batch["task_types"][0]
        question_text = batch["question_texts"][0]

        # Skip samples already processed in a prior partial run
        if _skip:
            key = (video_path, task_type, question_text)
            if key in _skip:
                continue

        # Skip videos that previously OOMed at minimum frames
        if oom_cache.get(video_path) == 0:
            n_skipped += 1
            logger.info(
                "Skipping %s [%s] — previously OOMed at minimum frames",
                video_path, task_type,
            )
            continue

        sp_len = int(batch["scanpath_lengths"][0])
        scanpath = batch["scanpaths"][0, :sp_len]   # (N, 4), possibly (0, 4)
        answer_idx = int(batch["answer_idxs"][0])
        correct_letter = batch["answers"][0]

        pred_letter, is_correct = eval_one_sample(
            model=model,
            processor=processor,
            mode=mode,
            ctx=ctx,
            video_path=video_path,
            scanpath=scanpath,
            question_text=question_text,
            options=batch["options"][0],
            answer_idx=answer_idx,
            start_time=float(batch["start_times"][0]),
            end_time=float(batch["end_times"][0]),
            abcd_token_ids=abcd_token_ids,
            device=device,
            max_frames=max_frames,
            vjepa_extractor=vjepa_extractor,
            oom_cache=oom_cache,
        )

        if pred_letter is None:
            n_skipped += 1
            logger.warning(
                "Inference failed (returned None) for %s [%s] -- skipping from accuracy denominator",
                video_path, task_type,
            )
            continue

        task_correct[task_type] += int(is_correct)
        task_total[task_type] += 1
        task_correct["all"] += int(is_correct)
        task_total["all"] += 1

        all_results.append({
            "video_path": video_path,
            "task_type": task_type,
            "question_text": question_text,
            "pred": pred_letter,
            "correct": correct_letter,
            "is_correct": is_correct,
        })

        newly_processed += 1
        if output_file and save_every > 0 and newly_processed % save_every == 0:
            _save_result_json(
                output_file,
                _build_result_dict(mode, all_results, task_correct, task_total, n_skipped),
            )
            logger.info(
                "Intermediate save: %d newly processed, %d total -> %s",
                newly_processed, len(all_results), output_file,
            )

    if n_skipped > 0:
        logger.warning(
            "run_eval complete: %d samples skipped due to inference failure "
            "(not counted in accuracy denominator -- check logs above for details)",
            n_skipped,
        )
    return _build_result_dict(mode, all_results, task_correct, task_total, n_skipped)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate GazeLens")
    p.add_argument(
        "--mode", required=True, choices=list(SUPPORTED_MODES),
        help="Evaluation mode.",
    )
    p.add_argument("--qa_files", nargs="+", required=True, help="QA JSON file paths")
    p.add_argument("--video_root", required=True, help="Root dir for video files")
    p.add_argument("--fixation_root", default=None, help="Root dir for fixation CSVs")
    p.add_argument("--output_file", required=True, help="Output JSON path")
    p.add_argument(
        "--checkpoint", default=None,
        help="f_theta state-dict .pt file (required for gaze modes)",
    )
    p.add_argument(
        "--vjepa_checkpoint", default=None,
        help="Path to V-JEPA 2.1 backbone checkpoint",
    )
    p.add_argument(
        "--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model path or local directory",
    )
    p.add_argument("--split_file", default=None, help="JSON with train/val/test video lists")
    p.add_argument(
        "--split", default=None, choices=["train", "val", "test"],
        help="Which split to evaluate (requires --split_file)",
    )
    p.add_argument("--max_frames", type=int, default=8, help="Max video frames per sample (default: 8)")
    p.add_argument(
        "--n_samples_max", type=int, default=None,
        help="Process at most this many samples (default: all).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted eval run from --output_file.",
    )
    p.add_argument(
        "--save_every", type=int, default=10,
        help="Write partial results every this many newly processed samples (default: 10).",
    )
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
        raise ValueError(f"--checkpoint is required for {mode} mode")

    # ---- Resumption: load prior results if --resume is set ----
    prior_results: Optional[List[Dict[str, Any]]] = None
    skip_keys: Optional[Set[tuple]] = None
    resume = getattr(args, "resume", False)
    if resume and os.path.exists(args.output_file):
        with open(args.output_file) as f:
            prev = json.load(f)
        prior = prev.get("results", [])
        if prior:
            prior_results = prior
            skip_keys = {_resume_key(r) for r in prior}
            logger.info(
                "Resume: found %d prior results in %s -- will skip those samples",
                len(prior), args.output_file,
            )
        else:
            logger.info("Resume: output file exists but has no prior results -- starting fresh")
    elif resume:
        logger.info("Resume: output file %s not found -- starting fresh", args.output_file)

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
    abcd_token_ids = get_abcd_token_ids(processor.tokenizer)
    logger.info("ABCD token IDs: %s", abcd_token_ids)

    # ---- Load LoRA adapter from checkpoint if present ----
    if args.checkpoint:
        _ckpt_peek = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(_ckpt_peek, dict) and "lora_state_dict" in _ckpt_peek:
            lora_cfg = _ckpt_peek.get("lora_config", {"rank": 8, "alpha": 16.0})
            apply_lora(model, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"])
            load_lora_state_dict(model, _ckpt_peek["lora_state_dict"])
            logger.info(
                "LoRA loaded from checkpoint: rank=%d alpha=%.1f",
                lora_cfg["rank"], lora_cfg["alpha"],
            )
        del _ckpt_peek

    # ---- Set up GazeLens context / hooks ----
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
        # Filter to valid constructor keys (old checkpoints may have extra keys)
        valid_keys = set(_CONFIG_KEYS)
        valid_keys.add("amplitude_init")
        ckpt_config = {k: v for k, v in ckpt_config.items() if k in valid_keys}

        gazelens_module = GazeLens(**ckpt_config).to(device)
        gazelens_module.load_state_dict(state)
        gazelens_module.eval()
        ctx = GazeLensContext(gazelens_module)
        register_gazelens_hooks(model, ctx)

        from gazeqwen.vjepa_features import VJEPAFeatureExtractor
        vjepa_ckpt = getattr(args, "vjepa_checkpoint", None)
        vjepa_extractor = VJEPAFeatureExtractor(device, checkpoint_path=vjepa_ckpt)

        logger.info(
            "GazeLens: checkpoint=%s  config=%s",
            args.checkpoint, gazelens_module.get_config(),
        )
        del raw

    # ---- Dataset ----
    dataset = GazeLensDataset(
        qa_files=args.qa_files,
        video_root=args.video_root,
        fixation_root=args.fixation_root,
        split=args.split,
        split_file=args.split_file,
    )
    n_samples_max = getattr(args, "n_samples_max", None)
    if n_samples_max is not None and len(dataset) > n_samples_max:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(n_samples_max))
        logger.info(
            "Dataset: %d samples (capped at --n_samples_max %d), mode=%s",
            n_samples_max, n_samples_max, mode,
        )
    else:
        logger.info("Dataset: %d samples, mode=%s", len(dataset), mode)

    # ---- Run evaluation ----
    results = run_eval(
        model=model,
        processor=processor,
        mode=mode,
        ctx=ctx,
        dataset=dataset,
        abcd_token_ids=abcd_token_ids,
        device=device,
        skip_keys=skip_keys,
        prior_results=prior_results,
        output_file=args.output_file,
        save_every=getattr(args, "save_every", 10),
        max_frames=getattr(args, "max_frames", 8),
        vjepa_extractor=vjepa_extractor,
    )

    # ---- Save final results ----
    _save_result_json(args.output_file, results)
    logger.info("Results saved to %s", args.output_file)

    # ---- Print accuracy table ----
    acc = results["accuracy"]
    n = results["n_samples"]
    print(f"\n=== {mode} ===")
    for task in sorted(acc):
        print(f"  {task:<35s}: {acc[task]:5.1f}%  (n={n.get(task, 0)})")

    if ctx is not None:
        remove_gazelens_hooks(model)


if __name__ == "__main__":
    main()
