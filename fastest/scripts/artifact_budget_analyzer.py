from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LIMIT_BYTES = 16_000_000
DEFAULT_COMPRESSION_RATIO = 0.78
TENSOR_METADATA_BYTES = 64


@dataclass(frozen=True)
class ArtifactComponentEstimate:
    name: str
    parameter_count: int
    raw_quantized_bytes: int
    estimated_bytes: int
    tensor_count: int


@dataclass(frozen=True)
class ArtifactBudgetEstimate:
    limit_bytes: int
    total_estimated_bytes: int
    model_estimated_bytes: int
    code_estimated_bytes: int
    over_budget: bool
    headroom_bytes: int
    quantization_scheme: str
    compression_ratio: float
    components: tuple[ArtifactComponentEstimate, ...]

    def to_log_lines(self) -> list[str]:
        state = "over-budget" if self.over_budget else "within-budget"
        lines = [
            (
                "artifact_budget:"
                f"state:{state} total_bytes:{self.total_estimated_bytes} "
                f"limit_bytes:{self.limit_bytes} headroom_bytes:{self.headroom_bytes} "
                f"quantization:{self.quantization_scheme} "
                f"compression_ratio:{self.compression_ratio:.2f}"
            )
        ]
        for component in self.components:
            lines.append(
                "artifact_budget_component:"
                f"name:{component.name} params:{component.parameter_count} "
                f"raw_quantized_bytes:{component.raw_quantized_bytes} "
                f"estimated_bytes:{component.estimated_bytes}"
            )
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "limitBytes": self.limit_bytes,
            "totalEstimatedBytes": self.total_estimated_bytes,
            "modelEstimatedBytes": self.model_estimated_bytes,
            "codeEstimatedBytes": self.code_estimated_bytes,
            "overBudget": self.over_budget,
            "headroomBytes": self.headroom_bytes,
            "quantizationScheme": self.quantization_scheme,
            "compressionRatio": self.compression_ratio,
            "components": [
                {
                    "name": component.name,
                    "parameterCount": component.parameter_count,
                    "rawQuantizedBytes": component.raw_quantized_bytes,
                    "estimatedBytes": component.estimated_bytes,
                    "tensorCount": component.tensor_count,
                }
                for component in self.components
            ],
        }


def _attr(config: Any, name: str, default: Any | None = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _int_attr(config: Any, name: str, default: int) -> int:
    value = _attr(config, name, default)
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_attr(config: Any, name: str, default: bool) -> bool:
    value = _attr(config, name, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no"}
    if value is None:
        return default
    return bool(value)


def _compressed_bytes(raw_quantized_bytes: int, compression_ratio: float) -> int:
    return int(math.ceil(raw_quantized_bytes * compression_ratio))


def _quantized_payload_bytes(parameter_count: int, quantization_bits: int) -> int:
    return int(math.ceil(parameter_count * quantization_bits / 8))


def _matrix_quantized_payload_bytes(rows: int, cols: int, quantization_bits: int) -> int:
    return _quantized_payload_bytes(rows * cols, quantization_bits)


def _matrix_quantization_overhead_bytes(rows: int) -> int:
    return rows * 4 + TENSOR_METADATA_BYTES


def _vector_fp16_bytes(size: int) -> int:
    return size * 2 + TENSOR_METADATA_BYTES


def _component(
    name: str,
    *,
    parameter_count: int,
    raw_quantized_bytes: int,
    tensor_count: int,
    compression_ratio: float,
) -> ArtifactComponentEstimate:
    return ArtifactComponentEstimate(
        name=name,
        parameter_count=parameter_count,
        raw_quantized_bytes=raw_quantized_bytes,
        estimated_bytes=_compressed_bytes(raw_quantized_bytes, compression_ratio),
        tensor_count=tensor_count,
    )


def estimate_artifact_budget(
    config: Any,
    *,
    limit_bytes: int = DEFAULT_LIMIT_BYTES,
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    code_path: str | Path | None = None,
) -> ArtifactBudgetEstimate:
    vocab_size = _int_attr(config, "vocab_size", 1024)
    num_layers = _int_attr(config, "num_layers", 9)
    model_dim = _int_attr(config, "model_dim", _int_attr(config, "dim", 512))
    num_heads = _int_attr(config, "num_heads", 8)
    num_kv_heads = _int_attr(config, "num_kv_heads", 4)
    mlp_mult = _int_attr(config, "mlp_mult", 2)
    tie_embeddings = _bool_attr(config, "tie_embeddings", True)
    quantization_bits = _int_attr(config, "quantization_bits", 8)
    quantization_scheme = str(
        _attr(config, "quantization_scheme", "int8-per-row-zlib-projection")
    )

    if model_dim <= 0 or vocab_size <= 0 or num_layers <= 0:
        raise ValueError("vocab_size, model_dim, and num_layers must be positive")
    if num_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_heads and num_kv_heads must be positive")
    if quantization_bits <= 0:
        raise ValueError("quantization_bits must be positive")
    if model_dim % num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    hidden_dim = model_dim * mlp_mult

    embedding_params = vocab_size * model_dim
    embedding_raw = _matrix_quantized_payload_bytes(
        vocab_size,
        model_dim,
        quantization_bits,
    )
    quantization_overhead_raw = _matrix_quantization_overhead_bytes(vocab_size)
    output_head_params = 0
    output_head_raw = 0
    if not tie_embeddings:
        output_head_params = vocab_size * model_dim
        output_head_raw = _matrix_quantized_payload_bytes(
            vocab_size,
            model_dim,
            quantization_bits,
        )
        quantization_overhead_raw += _matrix_quantization_overhead_bytes(vocab_size)

    attention_params_per_layer = model_dim * model_dim * 2 + model_dim * kv_dim * 2
    attention_raw_per_layer = (
        _matrix_quantized_payload_bytes(model_dim, model_dim, quantization_bits)
        + _matrix_quantized_payload_bytes(kv_dim, model_dim, quantization_bits)
        + _matrix_quantized_payload_bytes(kv_dim, model_dim, quantization_bits)
        + _matrix_quantized_payload_bytes(model_dim, model_dim, quantization_bits)
    )
    quantization_overhead_raw += num_layers * (
        _matrix_quantization_overhead_bytes(model_dim)
        + _matrix_quantization_overhead_bytes(kv_dim)
        + _matrix_quantization_overhead_bytes(kv_dim)
        + _matrix_quantization_overhead_bytes(model_dim)
    )
    mlp_params_per_layer = model_dim * hidden_dim * 2
    mlp_raw_per_layer = _matrix_quantized_payload_bytes(
        hidden_dim,
        model_dim,
        quantization_bits,
    ) + _matrix_quantized_payload_bytes(
        model_dim,
        hidden_dim,
        quantization_bits,
    )
    quantization_overhead_raw += num_layers * (
        _matrix_quantization_overhead_bytes(hidden_dim)
        + _matrix_quantization_overhead_bytes(model_dim)
    )
    control_params_per_layer = model_dim * 2 + 1
    control_raw_per_layer = _vector_fp16_bytes(model_dim) * 2 + _vector_fp16_bytes(1)

    components = [
        _component(
            "embeddings",
            parameter_count=embedding_params,
            raw_quantized_bytes=embedding_raw,
            tensor_count=1,
            compression_ratio=compression_ratio,
        ),
        _component(
            "attention_blocks",
            parameter_count=attention_params_per_layer * num_layers,
            raw_quantized_bytes=attention_raw_per_layer * num_layers,
            tensor_count=4 * num_layers,
            compression_ratio=compression_ratio,
        ),
        _component(
            "mlp_blocks",
            parameter_count=mlp_params_per_layer * num_layers,
            raw_quantized_bytes=mlp_raw_per_layer * num_layers,
            tensor_count=2 * num_layers,
            compression_ratio=compression_ratio,
        ),
        _component(
            "control_tensors",
            parameter_count=control_params_per_layer * num_layers,
            raw_quantized_bytes=control_raw_per_layer * num_layers,
            tensor_count=3 * num_layers,
            compression_ratio=1.0,
        ),
        _component(
            "quantization_overhead",
            parameter_count=0,
            raw_quantized_bytes=quantization_overhead_raw,
            tensor_count=7 * num_layers + (2 if not tie_embeddings else 1),
            compression_ratio=1.0,
        ),
    ]
    if output_head_params:
        components.append(
            _component(
                "output_head",
                parameter_count=output_head_params,
                raw_quantized_bytes=output_head_raw,
                tensor_count=1,
                compression_ratio=compression_ratio,
            )
        )

    model_estimated_bytes = sum(component.estimated_bytes for component in components)
    code_estimated_bytes = 0
    if code_path is not None:
        path = Path(code_path)
        if path.exists():
            code_estimated_bytes = len(zlib.compress(path.read_bytes(), level=9))
    total_estimated_bytes = model_estimated_bytes + code_estimated_bytes

    return ArtifactBudgetEstimate(
        limit_bytes=limit_bytes,
        total_estimated_bytes=total_estimated_bytes,
        model_estimated_bytes=model_estimated_bytes,
        code_estimated_bytes=code_estimated_bytes,
        over_budget=total_estimated_bytes > limit_bytes,
        headroom_bytes=limit_bytes - total_estimated_bytes,
        quantization_scheme=quantization_scheme,
        compression_ratio=compression_ratio,
        components=tuple(components),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate packed model artifact bytes before training or final packing."
    )
    parser.add_argument("--config", type=Path, required=True, help="JSON candidate/model config path.")
    parser.add_argument(
        "--code-path",
        type=Path,
        default=None,
        help="Optional training or submission script to include as compressed code bytes.",
    )
    parser.add_argument("--limit-bytes", type=int, default=DEFAULT_LIMIT_BYTES)
    parser.add_argument("--compression-ratio", type=float, default=DEFAULT_COMPRESSION_RATIO)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config JSON must contain an object")

    estimate = estimate_artifact_budget(
        config,
        limit_bytes=args.limit_bytes,
        compression_ratio=args.compression_ratio,
        code_path=args.code_path,
    )
    if args.format == "json":
        print(json.dumps(estimate.to_dict(), indent=2, sort_keys=True))
    else:
        for line in estimate.to_log_lines():
            print(line)
    return 2 if estimate.over_budget else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
