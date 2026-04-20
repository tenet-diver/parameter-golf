#!/usr/bin/env python3
"""Cheap artifact budget check for local screening."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("code_path", nargs="?", default="train_gpt.py")
    parser.add_argument("--target-total-bytes", type=int, default=16_000_000)
    parser.add_argument("--model-bytes", type=int, default=0, help="Optional estimated model artifact bytes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_path = Path(args.code_path)
    if not code_path.exists():
        raise SystemExit(f"Missing code path: {code_path}")
    code_bytes = code_path.stat().st_size
    total_bytes = code_bytes + args.model_bytes
    remaining = args.target_total_bytes - total_bytes
    print(f"code_path={code_path}")
    print(f"code_bytes={code_bytes}")
    print(f"model_bytes_estimate={args.model_bytes}")
    print(f"target_total_bytes={args.target_total_bytes}")
    print(f"estimated_total_bytes={total_bytes}")
    print(f"remaining_budget_bytes={remaining}")
    if remaining < 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
