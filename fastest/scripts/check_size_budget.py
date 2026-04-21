#!/usr/bin/env python3
"""Cheap artifact budget check for local screening."""

from __future__ import annotations

import argparse
from pathlib import Path


def _bytes_from_params(param_count: int, bits: int) -> int:
    return (param_count * bits + 7) // 8


def estimate_component_bytes(
    *,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    quant_bits: int,
    mlp_multiplier: float,
) -> dict[str, int]:
    if n_heads <= 0 or d_model <= 0:
        raise ValueError("d_model and n_heads must be positive")
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if quant_bits <= 0:
        raise ValueError("quant_bits must be positive")
    if mlp_multiplier <= 0:
        raise ValueError("mlp_multiplier must be positive")

    mlp_hidden = int(d_model * mlp_multiplier)
    embedding_params = vocab_size * d_model
    # Attention projection weights (Q, K, V, and output) across all layers.
    head_params = n_layers * (4 * d_model * d_model)
    # MLP up/down projection plus two per-layer norms.
    block_params = n_layers * ((2 * d_model * mlp_hidden) + (2 * d_model))
    return {
        "embeddings": _bytes_from_params(embedding_params, quant_bits),
        "heads": _bytes_from_params(head_params, quant_bits),
        "blocks": _bytes_from_params(block_params, quant_bits),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("code_path", nargs="?", default="train_gpt.py")
    parser.add_argument("--target-total-bytes", type=int, default=16_000_000)
    parser.add_argument("--model-bytes", type=int, default=0, help="Optional estimated model artifact bytes")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--mlp-multiplier", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_path = Path(args.code_path)
    if not code_path.exists():
        raise SystemExit(f"Missing code path: {code_path}")
    code_bytes = code_path.stat().st_size
    architecture_values = [args.vocab_size, args.d_model, args.n_layers, args.n_heads]
    has_architecture_args = any(value is not None for value in architecture_values)
    if has_architecture_args and not all(value is not None for value in architecture_values):
        raise SystemExit("When using architecture estimation, provide --vocab-size --d-model --n-layers --n-heads")

    component_bytes = {"embeddings": 0, "heads": 0, "blocks": 0}
    model_bytes = args.model_bytes
    if has_architecture_args:
        component_bytes = estimate_component_bytes(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            quant_bits=args.quant_bits,
            mlp_multiplier=args.mlp_multiplier,
        )
        model_bytes = sum(component_bytes.values())

    total_bytes = code_bytes + model_bytes
    remaining = args.target_total_bytes - total_bytes
    print(f"code_path={code_path}")
    print(f"code_bytes={code_bytes}")
    print(f"model_bytes_estimate={model_bytes}")
    if has_architecture_args:
        print(f"quantization_bits={args.quant_bits}")
        print(f"component_embeddings_bytes={component_bytes['embeddings']}")
        print(f"component_blocks_bytes={component_bytes['blocks']}")
        print(f"component_heads_bytes={component_bytes['heads']}")
    print(f"target_total_bytes={args.target_total_bytes}")
    print(f"estimated_total_bytes={total_bytes}")
    print(f"remaining_budget_bytes={remaining}")
    if remaining < 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
