"""
GazeLens split file generator.

Creates a deterministic 70/15/15 train/val/test split **by video** (not by QA pair)
to prevent data leakage from the same video appearing in multiple splits.

The split file is consumed by GazeLensDataset via --split_file / split= arguments.

Usage:
    python -m gazelens.split \\
        --qa_files /path/to/otp.json /path/to/gsm.json ... \\
        --output_file splits/gazelens_split.json \\
        [--train_frac 0.70] [--val_frac 0.15] [--seed 42]

Output JSON format:
    {
      "train": ["video_a.mp4", "video_b.mp4", ...],
      "val":   ["video_c.mp4", ...],
      "test":  ["video_d.mp4", ...]
    }

Keys are video **basenames** (matching the basename used in GazeLensDataset).
Lists within each split are sorted alphabetically for reproducibility.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def collect_video_names(qa_files: List[str]) -> List[str]:
    """
    Collect all unique video basenames referenced across one or more QA JSON files.

    Each QA JSON may be a list of entries or a single entry dict.  Each entry
    is expected to have a "video_path" key; any entry missing that key is skipped.
    Files that cannot be read are silently skipped.

    Args:
        qa_files: paths to QA JSON files (OTP, GSM, SR, NFI, FAP, etc.)

    Returns:
        Sorted list of unique video basenames (e.g. ["video_001.mp4", ...]).
    """
    names: set = set()
    for path in qa_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            vp = entry.get("video_path", "")
            if vp:
                names.add(os.path.basename(vp))

    return sorted(names)


def make_split(
    video_names: List[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create a deterministic train/val/test split by video.

    Videos are shuffled with the given seed, then allocated in order:
      - train: first ``floor(n * train_frac)`` videos (min 1 if n >= 1)
      - val:   next  ``floor(n * val_frac)`` videos   (clamped so train+val <= n)
      - test:  remainder

    All three output lists are sorted alphabetically for reproducibility.

    Args:
        video_names: list of video basenames to split (need not be pre-sorted)
        train_frac:  fraction for training (default 0.70)
        val_frac:    fraction for validation (default 0.15); test gets the rest
        seed:        RNG seed for reproducible shuffling

    Returns:
        Dict with keys "train", "val", "test" mapping to sorted lists of basenames.

    Raises:
        ValueError: if train_frac + val_frac > 1.0 or either fraction < 0.
    """
    if train_frac < 0 or val_frac < 0:
        raise ValueError("train_frac and val_frac must be non-negative")
    if train_frac + val_frac > 1.0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) > 1.0"
        )

    if not video_names:
        return {"train": [], "val": [], "test": []}

    shuffled = list(video_names)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * train_frac))
    n_val = max(0, min(int(n * val_frac), n - n_train))
    # n_test is whatever remains (no explicit clamp needed)

    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val :])

    return {"train": train, "val": val, "test": test}


def save_split(split: Dict[str, List[str]], output_file: str) -> None:
    """
    Save a split dict to a JSON file.

    Parent directories are created automatically.

    Args:
        split:       dict with "train"/"val"/"test" keys
        output_file: destination file path
    """
    parent = os.path.dirname(os.path.abspath(output_file))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(split, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a by-video train/val/test split for GazeLens"
    )
    p.add_argument(
        "--qa_files", nargs="+", required=True,
        help="Paths to QA JSON files (any combination of OTP/GSM/SR/NFI/FAP/etc.)",
    )
    p.add_argument(
        "--output_file", required=True,
        help="Destination path for the split JSON (e.g. splits/gazelens_split.json)",
    )
    p.add_argument(
        "--train_frac", type=float, default=0.70,
        help="Fraction of videos for training (default 0.70)",
    )
    p.add_argument(
        "--val_frac", type=float, default=0.15,
        help="Fraction of videos for validation (default 0.15; test = remainder)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible shuffling (default 42)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    video_names = collect_video_names(args.qa_files)
    print(f"Found {len(video_names)} unique videos across {len(args.qa_files)} QA file(s)")

    split = make_split(
        video_names,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    n_train = len(split["train"])
    n_val = len(split["val"])
    n_test = len(split["test"])
    total = n_train + n_val + n_test
    print(
        f"Split: train={n_train} ({n_train/total*100:.1f}%),  "
        f"val={n_val} ({n_val/total*100:.1f}%),  "
        f"test={n_test} ({n_test/total*100:.1f}%)"
    )

    save_split(split, args.output_file)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
