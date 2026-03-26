"""
GazeLens training loop.

Trains f_theta (GazeLens module) while keeping Qwen2.5-VL-7B frozen.
CE loss is computed on the 4-way softmax over ABCD answer tokens.

Important: do NOT wrap the training forward pass in torch.no_grad().
The VLM parameters have requires_grad=False so their gradients are never
computed, but the computation graph must be built through the activations
so that gradients reach f_theta.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gazeqwen.config import (
    LR,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    GRAD_ACCUM_STEPS,
    MAX_EPOCHS,
    QWEN_TEMPORAL_PATCH_SIZE,
)
# Temporal task types for validation accuracy tracking
_TEMPORAL_TASKS = [
    "past_object_transition_prediction",
    "past_gaze_sequence_matching",
    "past_scene_recall",
    "past_non_fixated_object_identification",
    "present_future_action_prediction",
    "Scanpath_Next_After_Group",
    "Scanpath_Transition_Pattern",
    "Scanpath_Never_Gazed",
    "Scene_Reconstruction",
    "Future_Action_Prediction",
    "Object_Remind_Easy",
    "Object_Remind_Hard",
    "future_action_prediction",
    "FAP",
    "scene_recall",
    "scanpath_reasoning",
]
from gazeqwen.data import GazeLensDataset, collate_fn
from gazeqwen.hooks import GazeLensContext, register_gazelens_hooks, remove_gazelens_hooks
from gazeqwen.lora import apply_lora, get_lora_state_dict, load_lora_state_dict, get_lora_params, count_lora_params
from gazeqwen.model import GazeLens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping helper. patience=0 disables (always returns False)."""

    def __init__(self, patience: int = 0, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = -float("inf")
        self.counter: int = 0

    def step(self, val_acc: float) -> bool:
        """Record val accuracy; returns True when training should stop."""
        if val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.patience > 0 and self.counter >= self.patience

    @property
    def improved(self) -> bool:
        """True if the most recent .step() call produced a new best."""
        return self.counter == 0


# ---------------------------------------------------------------------------
# Pure utility functions (testable without the 7B model)
# ---------------------------------------------------------------------------

def pick_fps(duration: float) -> float:
    """Dynamic FPS selection: keeps sampled frames bounded regardless of clip length."""
    if duration <= 30:
        return 1.0
    elif duration <= 60:
        return 0.5
    elif duration <= 300:
        return 0.2
    else:
        return 0.1


def compute_frame_times(
    start_time: float,
    n_temporal_steps: int,
    fps: float,
    temporal_patch_size: int = QWEN_TEMPORAL_PATCH_SIZE,
) -> List[float]:
    """Compute wall-clock timestamp for each ViT temporal step."""
    dt = temporal_patch_size / fps
    return [start_time + i * dt for i in range(n_temporal_steps)]


def get_abcd_token_ids(tokenizer) -> List[int]:
    """Return token IDs for A, B, C, D answer options."""
    ids = []
    for letter in ("A", "B", "C", "D"):
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Tokenizer could not encode '{letter}'")
        ids.append(encoded[0])
    return ids


def build_mcqa_prompt(question_text: str, options: List[str]) -> str:
    """Build an MCQA prompt ending with 'The best option is:' for next-token prediction."""
    formatted = []
    for i, opt in enumerate(options):
        prefix = f"{chr(65 + i)}."
        if opt.strip().upper().startswith(prefix):
            formatted.append(opt.strip())
        else:
            formatted.append(f"{prefix} {opt.strip()}")
    opts_str = "\n".join(formatted)
    return (
        f"Question: {question_text}\n"
        f"Options:\n{opts_str}\n\n"
        "The best option is:"
    )


def create_video_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    target_fps: Optional[float] = None,
) -> Optional[str]:
    """Extract a video clip using FFmpeg. Returns temp .mp4 path or None on error."""
    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)
        duration = max(end_time - start_time, 0.1)

        # fps-limited extraction for long clips (> 60s) where pick_fps < 1.0
        if target_fps is not None and target_fps < 1.0 and duration > 60.0:
            cmd_fps = [
                "ffmpeg", "-y",
                "-vsync", "0",
                "-skip_frame", "noref",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(duration),
                "-vf", f"fps={target_fps:.4f}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-an",
                temp_path,
            ]
            try:
                subprocess.run(
                    cmd_fps,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=300,
                )
                return temp_path
            except Exception as exc:
                logger.warning(
                    "FPS-limited clip failed for %s (%s); falling back to stream copy",
                    video_path, exc,
                )
                open(temp_path, "wb").close()

        # Default: stream copy (fast, preserves original fps)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            temp_path,
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=120,
        )
        return temp_path
    except Exception as exc:
        logger.warning("FFmpeg clip failed for %s: %s", video_path, exc)
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return None


# ---------------------------------------------------------------------------
# AdamW parameter groups
# ---------------------------------------------------------------------------

def get_param_groups(model: GazeLens, weight_decay: float) -> list:
    """Separate params into decay / no-decay groups for AdamW."""
    decay: list = []
    no_decay: list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "layer_embed" in name
            or name == "pool_query"
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Scanpath augmentation (training only)
# ---------------------------------------------------------------------------

def augment_scanpath(
    scanpath: torch.Tensor,
    xy_noise_std: float = 0.02,
    drop_prob: float = 0.1,
) -> torch.Tensor:
    """Gaussian noise on x/y + random fixation drop; at least 1 fixation kept."""
    N = scanpath.shape[0]
    if N == 0:
        return scanpath
    noisy = scanpath.clone()
    noise = torch.randn(N, 2, device=scanpath.device) * xy_noise_std
    noisy[:, :2] = (noisy[:, :2] + noise).clamp(0.0, 1.0)

    keep_mask = torch.rand(N, device=scanpath.device) > drop_prob
    if not keep_mask.any():
        keep_idx = int(torch.randint(N, (1,)).item())
        keep_mask[keep_idx] = True
    return noisy[keep_mask]


# ---------------------------------------------------------------------------
# Core differentiable forward pass
# ---------------------------------------------------------------------------

def forward_sample(
    model,
    processor,
    ctx: GazeLensContext,
    video_path: str,
    scanpath: torch.Tensor,      # (N, 4) float tensor [x, y, t, dur]
    question_text: str,
    options: List[str],
    answer_idx: int,              # 0-3
    start_time: float,
    end_time: float,
    abcd_token_ids: List[int],
    device: torch.device,
    no_grad: bool = False,
    label_smoothing: float = 0.0,
    vjepa_extractor=None,
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    Single forward pass through the hooked VLM.

    Returns (loss, is_correct) where loss is a scalar CE tensor or None on error.
    When no_grad=False (training), the computation graph is preserved for backprop.
    """
    duration = end_time - start_time
    fps = pick_fps(duration)

    temp_video = create_video_clip(video_path, start_time, end_time, target_fps=fps)
    if temp_video is None:
        return None, False

    try:
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
                        "max_frames": 3,
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

        # Compute per-temporal-step timestamps from processor output
        frame_times: List[float] = []
        video_grid_thw = inputs.get("video_grid_thw")
        if video_grid_thw is not None and len(video_grid_thw) > 0:
            T = int(video_grid_thw[0][0])
            frame_times = compute_frame_times(start_time, T, fps)

        scanpath_dev = scanpath.to(device)

        # Extract V-JEPA backbone features
        backbone_features = None
        if video_grid_thw is not None and len(video_grid_thw) > 0:
            T = int(video_grid_thw[0][0])
            llm_H_est = int(video_grid_thw[0][1]) // 2  # spatial_merge_size=2
            llm_W_est = int(video_grid_thw[0][2]) // 2
            if video_inputs is not None and len(video_inputs) > 0 and vjepa_extractor is not None:
                raw_frames = video_inputs[0]  # (N_frames, C, H, W) float [0,255]
                backbone_features = vjepa_extractor.extract_from_raw_frames(
                    raw_frames, T, llm_H_est, llm_W_est,
                )

        # Forward pass through hooked VLM
        def _run():
            with ctx.active(scanpath_dev, frame_times, backbone_features=backbone_features):
                return model(**inputs)

        if no_grad:
            with torch.no_grad():
                output = _run()
        else:
            output = _run()

        # 4-way CE loss over ABCD tokens
        logits_at_ans = output.logits[0, -1, :]   # (vocab_size,)
        abcd_ids_t = torch.tensor(
            abcd_token_ids, device=logits_at_ans.device, dtype=torch.long
        )
        abcd_logits = logits_at_ans[abcd_ids_t]    # (4,)
        ans_t = torch.tensor(
            answer_idx, device=abcd_logits.device, dtype=torch.long
        )
        smoothing = label_smoothing if not no_grad else 0.0
        loss = F.cross_entropy(
            abcd_logits.unsqueeze(0), ans_t.unsqueeze(0), label_smoothing=smoothing
        )

        pred_idx = int(abcd_logits.argmax().item())
        is_correct = pred_idx == answer_idx

        return loss, is_correct

    except Exception as exc:
        logger.warning("Exception in forward_sample for %s: %s", video_path, exc)
        return None, False

    finally:
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def eval_epoch(
    model,
    processor,
    ctx: GazeLensContext,
    gazelens_module,
    dataloader: DataLoader,
    abcd_token_ids: List[int],
    device: torch.device,
    vjepa_extractor=None,
) -> Tuple[float, Dict[str, float]]:
    """Run one evaluation pass. Returns (overall_acc%, per_task_acc)."""
    if gazelens_module is not None:
        gazelens_module.eval()
    correct, total = 0, 0
    task_correct: Dict[str, int] = {}
    task_total: Dict[str, int] = {}

    for batch in dataloader:
        video_path = batch["video_paths"][0]
        if not os.path.exists(video_path):
            continue

        sp_len = int(batch["scanpath_lengths"][0])
        scanpath = batch["scanpaths"][0, :sp_len]
        task_type = batch["task_types"][0]

        sample_loss, is_correct = forward_sample(
            model=model, processor=processor, ctx=ctx,
            video_path=video_path, scanpath=scanpath,
            question_text=batch["question_texts"][0],
            options=batch["options"][0],
            answer_idx=int(batch["answer_idxs"][0]),
            start_time=float(batch["start_times"][0]),
            end_time=float(batch["end_times"][0]),
            abcd_token_ids=abcd_token_ids,
            device=device, no_grad=True,
            vjepa_extractor=vjepa_extractor,
        )

        if sample_loss is None:
            continue

        total += 1
        correct += int(is_correct)
        task_correct[task_type] = task_correct.get(task_type, 0) + int(is_correct)
        task_total[task_type] = task_total.get(task_type, 0) + 1

    if gazelens_module is not None:
        gazelens_module.train()
    overall_acc = (correct / total * 100.0) if total > 0 else 0.0
    per_task_acc = {
        t: task_correct[t] / task_total[t] * 100.0
        for t in task_total
    }
    return overall_acc, per_task_acc


# ---------------------------------------------------------------------------
# Val metric helpers
# ---------------------------------------------------------------------------

def compute_temporal_val_acc(per_task_acc: Dict[str, float]) -> Optional[float]:
    """Average accuracy over temporal task types present in per_task_acc."""
    vals = [per_task_acc[t] for t in _TEMPORAL_TASKS if t in per_task_acc]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args) -> Tuple[GazeLens, List[dict]]:
    """
    Main training entry point.

    Loads Qwen2.5-VL-7B (frozen, eager attention), sets up GazeLens f_theta
    and attention hooks, then trains for max_epochs with AdamW + cosine LR.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    # --- Load frozen VLM ---
    model_path = getattr(args, "model_path", None) or "Qwen/Qwen2.5-VL-7B-Instruct"
    logger.info("Loading %s (eager attention) ...", model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled (saves GPU memory, slower backward)")

    # --- LoRA on LLM Q/V projections (always enabled) ---
    lora_enabled = True
    lora_rank = getattr(args, "lora_rank", 8)
    lora_alpha = getattr(args, "lora_alpha", 16.0)
    if lora_enabled:
        lora_layers = apply_lora(model, rank=lora_rank, alpha=lora_alpha)
        logger.info(
            "LoRA enabled: rank=%d alpha=%.1f params=%d layers=%d",
            lora_rank, lora_alpha, count_lora_params(model), len(lora_layers),
        )

    # --- Two-stage training: freeze LoRA in stage 1 ---
    two_stage = getattr(args, "two_stage", False) and lora_enabled
    stage1_epochs = getattr(args, "stage1_epochs", 1)
    if two_stage:
        for p in get_lora_params(model):
            p.requires_grad = False
        logger.info(
            "Two-stage training: stage 1 = f_theta only (%d epoch(s)), "
            "stage 2 = f_theta + LoRA (remaining epochs)",
            stage1_epochs,
        )

    processor = AutoProcessor.from_pretrained(model_path)
    device = next(model.parameters()).device

    abcd_token_ids = get_abcd_token_ids(processor.tokenizer)
    logger.info("ABCD token IDs: %s", abcd_token_ids)

    # --- f_theta setup (GazeLens: V-JEPA + Coord-PE + Per-layer) ---
    from gazeqwen.vjepa_features import VJEPAFeatureExtractor
    gazelens_module = GazeLens().to(device)
    vjepa_ckpt = getattr(args, "vjepa_checkpoint", None)
    vjepa_extractor = VJEPAFeatureExtractor(device, checkpoint_path=vjepa_ckpt)

    gazelens_module.train()
    logger.info("GazeLens params: %d  config=%s",
                gazelens_module.count_params(), gazelens_module.get_config())

    pretrained_ftheta = getattr(args, "pretrained_ftheta", None)
    if pretrained_ftheta:
        pt_ckpt = torch.load(pretrained_ftheta, map_location=device, weights_only=False)
        pt_state = pt_ckpt.get("state_dict", pt_ckpt)
        gazelens_module.load_state_dict(pt_state)
        logger.info("Loaded pretrained f_theta from %s", pretrained_ftheta)
        del pt_ckpt

    ctx = GazeLensContext(gazelens_module)
    register_gazelens_hooks(model, ctx)

    # --- Datasets ---
    split_file = getattr(args, "split_file", None)

    def _make_ds(split):
        return GazeLensDataset(
            qa_files=args.qa_files,
            video_root=args.video_root,
            fixation_root=getattr(args, "fixation_root", None),
            split=split if split_file else None,
            split_file=split_file,
        )

    train_ds = _make_ds("train")
    val_ds = _make_ds("val")

    n_samples_max = getattr(args, "n_samples_max", None)
    if n_samples_max is not None:
        from torch.utils.data import Subset
        if len(train_ds) > n_samples_max:
            train_ds = Subset(train_ds, range(n_samples_max))
        if len(val_ds) > n_samples_max:
            val_ds = Subset(val_ds, range(n_samples_max))
        logger.info(
            "Train: %d samples (capped), Val: %d samples (capped) [--n_samples_max %d]",
            len(train_ds), len(val_ds), n_samples_max,
        )
    else:
        logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # --- Optimizer + cosine LR with warmup ---
    grad_accum = getattr(args, "grad_accum_steps", GRAD_ACCUM_STEPS)
    max_epochs = getattr(args, "max_epochs", MAX_EPOCHS)
    lr = getattr(args, "lr", LR)
    warmup = getattr(args, "warmup_steps", WARMUP_STEPS)
    wd = getattr(args, "weight_decay", WEIGHT_DECAY)

    param_groups = get_param_groups(gazelens_module, wd)
    lora_params = get_lora_params(model)
    if lora_params:
        param_groups.append({"params": lora_params, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / grad_accum))
    total_steps = max(steps_per_epoch * max_epochs, 1)
    warmup_frac = min(warmup / total_steps, 0.5)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_frac,
        anneal_strategy="cos",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    patience = getattr(args, "patience", 0)
    min_delta = getattr(args, "min_delta", 0.0)
    label_smoothing = getattr(args, "label_smoothing", 0.1)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    metrics_log: List[dict] = []

    # --- Optional: resume from a previous checkpoint ---
    start_epoch = 0
    resume_path = getattr(args, "resume", None)
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if lora_enabled:
            ckpt_preview = torch.load(resume_path, map_location=device, weights_only=False)
            if "lora_state_dict" in ckpt_preview:
                load_lora_state_dict(model, ckpt_preview["lora_state_dict"])
                logger.info("Restored LoRA weights from checkpoint")
            del ckpt_preview
        start_epoch = load_resume_checkpoint(
            path=resume_path,
            gazelens_module=gazelens_module,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            device=device,
        )
        logger.info(
            "Resumed from %s: start_epoch=%d  best_val_acc=%.1f%%",
            resume_path, start_epoch, early_stopping.best,
        )

    # All trainable params for gradient clipping
    all_trainable_params = list(gazelens_module.parameters()) if gazelens_module is not None else []
    if lora_enabled:
        all_trainable_params += get_lora_params(model)

    for epoch in range(start_epoch, max_epochs):
        # Two-stage transition: unfreeze LoRA at start of stage 2
        if two_stage and epoch == stage1_epochs:
            for p in get_lora_params(model):
                p.requires_grad = True
            logger.info(
                "Stage 2 begins (epoch %d): LoRA unfrozen, training f_theta + LoRA jointly",
                epoch + 1,
            )

        if gazelens_module is not None:
            gazelens_module.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad()
        accum_count = 0

        for batch in train_loader:
            video_path = batch["video_paths"][0]
            if not os.path.exists(video_path):
                logger.warning("Video not found, skipping: %s", video_path)
                continue

            sp_len = int(batch["scanpath_lengths"][0])
            if sp_len == 0:
                logger.debug("Skipping sample with empty scanpath: %s", video_path)
                continue
            scanpath = batch["scanpaths"][0, :sp_len]
            scanpath = augment_scanpath(scanpath)

            loss, is_correct = forward_sample(
                model=model, processor=processor, ctx=ctx,
                video_path=video_path, scanpath=scanpath,
                question_text=batch["question_texts"][0],
                options=batch["options"][0],
                answer_idx=int(batch["answer_idxs"][0]),
                start_time=float(batch["start_times"][0]),
                end_time=float(batch["end_times"][0]),
                abcd_token_ids=abcd_token_ids,
                device=device, no_grad=False,
                label_smoothing=label_smoothing,
                vjepa_extractor=vjepa_extractor,
            )

            if loss is None:
                continue

            (loss / grad_accum).backward()
            accum_count += 1

            train_loss_sum += loss.item()
            train_correct += int(is_correct)
            train_total += 1

            if accum_count >= grad_accum:
                torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0

        # Final partial accumulation
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_acc = (train_correct / train_total * 100.0) if train_total > 0 else 0.0
        avg_loss = train_loss_sum / train_total if train_total > 0 else 0.0

        val_acc = 0.0
        val_temporal_acc: Optional[float] = None
        per_task_acc: Dict[str, float] = {}
        if len(val_ds) > 0:
            val_acc, per_task_acc = eval_epoch(
                model=model, processor=processor, ctx=ctx,
                gazelens_module=gazelens_module, dataloader=val_loader,
                abcd_token_ids=abcd_token_ids, device=device,
                vjepa_extractor=vjepa_extractor,
            )
            val_temporal_acc = compute_temporal_val_acc(per_task_acc)

        val_metric_mode = getattr(args, "val_metric", "overall")
        if val_metric_mode == "temporal" and val_temporal_acc is not None:
            selection_acc = val_temporal_acc
        else:
            selection_acc = val_acc

        if val_temporal_acc is not None:
            logger.info(
                "Epoch %d/%d  loss=%.4f  train_acc=%.1f%%"
                "  val_acc=%.1f%%  val_temporal_acc=%.1f%%",
                epoch + 1, max_epochs, avg_loss, train_acc, val_acc, val_temporal_acc,
            )
        else:
            logger.info(
                "Epoch %d/%d  loss=%.4f  train_acc=%.1f%%  val_acc=%.1f%%",
                epoch + 1, max_epochs, avg_loss, train_acc, val_acc,
            )

        metrics_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_temporal_acc": val_temporal_acc,
            "per_task_acc": per_task_acc,
        })

        # Checkpoint every epoch
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt")
        ckpt_data = {
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stopping": {
                "best": early_stopping.best,
                "counter": early_stopping.counter,
            },
            "val_acc": val_acc,
        }
        if gazelens_module is not None:
            epoch_config = gazelens_module.get_config()
            epoch_config["version"] = gazelens_module.version
            ckpt_data["state_dict"] = gazelens_module.state_dict()
            ckpt_data["config"] = epoch_config
        if lora_enabled:
            ckpt_data["lora_state_dict"] = get_lora_state_dict(model)
            ckpt_data["lora_config"] = {"rank": lora_rank, "alpha": lora_alpha}
        torch.save(ckpt_data, ckpt_path)

        should_stop = early_stopping.step(selection_acc)

        if early_stopping.improved:
            best_path = os.path.join(args.output_dir, "best_model.pt")
            best_ckpt = {}
            if gazelens_module is not None:
                best_config = gazelens_module.get_config()
                best_config["version"] = gazelens_module.version
                best_ckpt["state_dict"] = gazelens_module.state_dict()
                best_ckpt["config"] = best_config
            if lora_enabled:
                best_ckpt["lora_state_dict"] = get_lora_state_dict(model)
                best_ckpt["lora_config"] = {"rank": lora_rank, "alpha": lora_alpha}
            torch.save(best_ckpt, best_path)
            logger.info(
                "New best %s=%.1f%%, saved to %s",
                val_metric_mode, selection_acc, best_path,
            )

        if should_stop:
            logger.info(
                "Early stopping: no improvement for %d epoch(s) (patience=%d).",
                early_stopping.counter, patience,
            )
            break

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    logger.info("Training complete. Best val_acc=%.1f%%", early_stopping.best)
    remove_gazelens_hooks(model)
    return gazelens_module, metrics_log


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

def load_resume_checkpoint(
    path: str,
    gazelens_module: GazeLens,
    optimizer: torch.optim.Optimizer,
    scheduler,
    early_stopping: EarlyStopping,
    device: torch.device,
) -> int:
    """Load checkpoint and restore training state. Returns start_epoch."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if gazelens_module is not None and "state_dict" in ckpt:
        gazelens_module.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    # scheduler state intentionally NOT restored: loading it overwrites total_steps
    # with the checkpoint's value, causing a crash when training is extended.
    es_state = ckpt.get("early_stopping", {})
    if "best" in es_state:
        early_stopping.best = es_state["best"]
    if "counter" in es_state:
        early_stopping.counter = es_state["counter"]
    return int(ckpt.get("epoch", 0))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train GazeLens f_theta")
    p.add_argument("--qa_files", nargs="+", required=True,
                   help="Paths to QA JSON files")
    p.add_argument("--video_root", required=True,
                   help="Root directory for resolving relative video paths")
    p.add_argument("--fixation_root", default=None,
                   help="Root directory for *_fixation_filtered.csv files")
    p.add_argument("--output_dir", required=True,
                   help="Directory for checkpoints and metrics")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint .pt to resume training from")
    p.add_argument("--split_file", default=None,
                   help="JSON with train/val/test video lists")
    p.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HuggingFace model path or local directory")
    p.add_argument("--vjepa_checkpoint", default=None,
                   help="Path to V-JEPA 2.1 backbone checkpoint")
    p.add_argument("--pretrained_ftheta", default=None,
                   help="Path to checkpoint .pt for warm-start f_theta weights")
    # LoRA (always enabled, but rank/alpha configurable)
    p.add_argument("--lora_rank", type=int, default=8,
                   help="LoRA rank (default: 8)")
    p.add_argument("--lora_alpha", type=float, default=16.0,
                   help="LoRA alpha scaling factor (default: 16.0)")
    # Two-stage training
    p.add_argument("--two_stage", action="store_true",
                   help="Stage 1: f_theta only; Stage 2: f_theta + LoRA. Requires --lora.")
    p.add_argument("--stage1_epochs", type=int, default=1,
                   help="Epochs for stage 1 (default: 1). Only with --two_stage.")
    # Training hyperparameters
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce GPU memory")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing for CE loss (default: 0.1)")
    # Early stopping
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="Min improvement to count as new best (default: 0.0)")
    p.add_argument("--n_samples_max", type=int, default=None,
                   help="Limit each split to N samples (for smoke-tests)")
    p.add_argument("--val_metric", choices=["overall", "temporal"], default="overall",
                   help="Metric for best-model selection: 'overall' or 'temporal'")
    return p.parse_args(argv)


def main():
    args = _parse_args()
    train(args)


if __name__ == "__main__":
    main()
