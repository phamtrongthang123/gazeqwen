"""
GazeQwen single-sample inference demo.

Usage:
    export PYTHONPATH=".:third_party/vjepa2"
    python demo.py \
        --video /path/to/video.mp4 \
        --fixation /path/to/video_fixation_filtered.csv \
        --checkpoint checkpoints/gazeqwen/best_model.pt \
        --question "What shape does this object have?" \
        --options "A. square" "B. circular" "C. triangular" "D. rectangular" \
        --time 127.0

If no arguments are given, runs a built-in example from the test set.
"""

import argparse
import gc
import logging
import os
import sys
import tempfile

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GazeQwen single-sample demo")
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    parser.add_argument("--fixation", type=str, default=None, help="Path to fixation CSV")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gazeqwen/best_model.pt")
    parser.add_argument("--vjepa_checkpoint", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--options", nargs="+", default=None)
    parser.add_argument("--time", type=float, default=None, help="Question timestamp (seconds)")
    parser.add_argument("--answer", type=str, default=None, help="Ground-truth answer letter (optional)")
    parser.add_argument("--max_frames", type=int, default=8, help="Max video frames for VLM (lower = less VRAM)")
    args = parser.parse_args()

    # ---- Built-in example if no args ----
    if args.video is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        demo_data = os.path.join(project_root, "demo_data")
        args.video = os.path.join(demo_data, "OP03-R01-PastaSalad.mp4")
        args.fixation = os.path.join(demo_data, "OP03-R01-PastaSalad_fixation_filtered.csv")
        args.checkpoint = os.path.join(project_root, "checkpoints", "gazeqwen", "best_model.pt")
        args.vjepa_checkpoint = os.path.join(project_root, "checkpoints", "vjepa2_1_vitb_dist_vitG_384.pt")
        args.question = "Which object is the user currently gazing at?"
        args.options = ["A. box", "B. spices", "C. knife", "D. jar"]
        args.time = 276.0  # 04:36
        args.answer = "C"
        logger.info("No arguments given — using built-in example")
        if not os.path.exists(args.video):
            logger.error(
                "Demo video not found: %s\n"
                "Download it to demo_data/OP03-R01-PastaSalad.mp4\n"
                "(EGTEA Gaze+ dataset: http://cbs.ic.gatech.edu/fpv/)",
                args.video,
            )
            sys.exit(1)

    assert os.path.exists(args.video), f"Video not found: {args.video}"
    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Load Qwen2.5-VL ----
    logger.info("Loading %s ...", args.model_path)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()

    # ---- 2. Load GazeQwen checkpoint ----
    logger.info("Loading GazeQwen checkpoint: %s", args.checkpoint)
    from gazeqwen.model import GazeLens
    from gazeqwen.hooks import GazeLensContext, register_gazelens_hooks
    from gazeqwen.lora import apply_lora, load_lora_state_dict
    from gazeqwen.vjepa_features import VJEPAFeatureExtractor
    from gazeqwen.train import build_mcqa_prompt, compute_frame_times, get_abcd_token_ids, pick_fps, create_video_clip
    from gazeqwen.data import ScanpathLoader

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    f_theta = GazeLens(**ckpt["config"])
    f_theta.load_state_dict(ckpt["state_dict"])
    f_theta = f_theta.to(device)
    f_theta.eval()
    logger.info("GazeQwen f_theta: %d params", f_theta.count_params())

    # ---- 3. Apply LoRA ----
    lora_cfg = ckpt["lora_config"]
    apply_lora(model, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"])
    load_lora_state_dict(model, ckpt["lora_state_dict"])
    logger.info("LoRA applied: rank=%d, alpha=%.1f", lora_cfg["rank"], lora_cfg["alpha"])

    # ---- 4. Register hooks ----
    ctx = GazeLensContext(f_theta)
    register_gazelens_hooks(model, ctx)

    # ---- 5. Load V-JEPA ----
    vjepa = VJEPAFeatureExtractor(device=device, checkpoint_path=args.vjepa_checkpoint)

    # ---- 6. Prepare input ----
    question_time = args.time
    start_time = max(0.0, question_time - 60.0)
    end_time = question_time
    duration = end_time - start_time
    fps = pick_fps(duration)

    # Extract clip
    logger.info("Extracting clip [%.1fs - %.1fs] at %.2f fps ...", start_time, end_time, fps)
    temp_video = create_video_clip(args.video, start_time, end_time, target_fps=fps)
    assert temp_video is not None, "FFmpeg clip extraction failed"

    # Load scanpath
    if args.fixation and os.path.exists(args.fixation):
        loader = ScanpathLoader(args.fixation)
        scanpath = loader.load_clipped(start_time, end_time).to(device)
        logger.info("Scanpath: %d fixations in [%.1f, %.1f]s", scanpath.shape[0], start_time, end_time)
    else:
        scanpath = torch.zeros(0, 4, device=device)
        logger.info("No fixation file — running without gaze")

    # Build prompt
    prompt = build_mcqa_prompt(args.question, args.options)

    # Process through Qwen
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{temp_video}", "fps": fps, "max_frames": args.max_frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    # Compute frame times and V-JEPA features
    video_grid_thw = inputs.get("video_grid_thw")
    T = int(video_grid_thw[0][0])
    llm_H = int(video_grid_thw[0][1]) // 2
    llm_W = int(video_grid_thw[0][2]) // 2
    frame_times = compute_frame_times(start_time, T, fps)

    raw_frames = video_inputs[0]
    backbone_features = vjepa.extract_from_raw_frames(raw_frames, T, llm_H, llm_W)
    del image_inputs, video_inputs
    gc.collect()

    # ---- 7. Forward pass ----
    logger.info("Running forward pass (T=%d, H=%d, W=%d) ...", T, llm_H, llm_W)
    abcd_ids = get_abcd_token_ids(processor.tokenizer)

    with torch.no_grad():
        with ctx.active(scanpath, frame_times, backbone_features=backbone_features):
            output = model(**inputs)

    # ---- 8. Extract prediction ----
    logits = output.logits[0, -1, :]
    abcd_logits = logits[torch.tensor(abcd_ids, device=logits.device)]
    probs = torch.softmax(abcd_logits.float(), dim=0)
    pred_idx = int(probs.argmax().item())
    pred_letter = chr(65 + pred_idx)

    # ---- 9. Print results ----
    print("\n" + "=" * 60)
    print(f"Video:    {os.path.basename(args.video)}")
    print(f"Clip:     [{start_time:.0f}s - {end_time:.0f}s]")
    print(f"Gaze:     {scanpath.shape[0]} fixations")
    print(f"Question: {args.question}")
    for opt in args.options:
        marker = " >>>" if opt.strip().startswith(f"{pred_letter}.") else "    "
        print(f"  {marker} {opt}")
    print(f"\nPrediction: {pred_letter} (confidence: {probs[pred_idx]:.1%})")
    print(f"Probabilities: " + " | ".join(
        f"{chr(65+i)}: {probs[i]:.1%}" for i in range(len(args.options))
    ))
    if args.answer:
        correct = pred_letter == args.answer.upper()
        print(f"Ground truth: {args.answer}  {'CORRECT' if correct else 'WRONG'}")
    print("=" * 60)

    # Cleanup
    if os.path.exists(temp_video):
        os.remove(temp_video)


if __name__ == "__main__":
    main()
