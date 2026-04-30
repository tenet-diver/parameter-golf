from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = REPO_ROOT / "planning/parameter_tuning_space.json"
DEFAULT_QUEUE_PATH = REPO_ROOT / "planning/gpu_experiment_queue.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "planning/generated_parameter_tuning_gpu_queue.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be an object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def slugify(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-._").lower()
    return slug or "variant"


def stringify_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    raise ValueError(f"environment values must be scalar strings/numbers/bools, got {value!r}")


def build_axis_variants(
    *,
    base_env: dict[str, str],
    axes: dict[str, list[Any]],
    strategy: str,
    max_variants: int | None,
) -> list[tuple[str, dict[str, str], list[str]]]:
    normalized_axes = {
        str(key): [stringify_env_value(value) for value in values]
        for key, values in axes.items()
        if isinstance(values, list) and values
    }
    variants: list[tuple[str, dict[str, str], list[str]]] = []
    if strategy == "one-factor":
        variants.append(("base", dict(base_env), []))
        for key in sorted(normalized_axes):
            base_value = base_env.get(key)
            for value in normalized_axes[key]:
                if value == base_value:
                    continue
                env = dict(base_env)
                env[key] = value
                variants.append((f"{slugify(key)}-{slugify(value)}", env, [key]))
    elif strategy == "grid":
        keys = sorted(normalized_axes)
        for values in itertools.product(*(normalized_axes[key] for key in keys)):
            env = dict(base_env)
            changed: list[str] = []
            label_parts: list[str] = []
            for key, value in zip(keys, values, strict=True):
                env[key] = value
                label_parts.append(f"{slugify(key)}-{slugify(value)}")
                if value != base_env.get(key):
                    changed.append(key)
            variants.append(("-".join(label_parts), env, changed))
    else:
        raise ValueError(f"unsupported tuning strategy {strategy!r}")
    if max_variants is not None:
        variants = variants[:max_variants]
    return variants


def build_tuning_experiments(spec: dict[str, Any]) -> list[dict[str, Any]]:
    spaces = spec.get("searchSpaces")
    if not isinstance(spaces, list) or not spaces:
        raise ValueError("tuning spec must contain non-empty searchSpaces")
    generated: list[dict[str, Any]] = []
    for space in spaces:
        if not isinstance(space, dict):
            raise ValueError("search space entries must be objects")
        space_id = str(space.get("id") or "").strip()
        if not space_id:
            raise ValueError("search space id is required")
        base_env_raw = space.get("baseEnv")
        axes_raw = space.get("axes")
        if not isinstance(base_env_raw, dict) or not isinstance(axes_raw, dict):
            raise ValueError(f"search space {space_id!r} must define baseEnv and axes objects")
        base_env = {str(key): stringify_env_value(value) for key, value in base_env_raw.items()}
        axes = {str(key): value for key, value in axes_raw.items()}
        strategy = str(space.get("strategy") or spec.get("defaultStrategy") or "one-factor")
        max_variants_raw = space.get("maxVariants", spec.get("maxVariantsPerSpace"))
        max_variants = int(max_variants_raw) if isinstance(max_variants_raw, int) else None
        priority_start = int(space.get("priorityStart", spec.get("priorityStart", 100)))
        lane = str(space.get("lane") or "h100-1x-parameter-tuning")
        status = str(space.get("status") or "ready")
        implementation = str(space.get("implementation") or "autoregressive_gpt")
        variants = build_axis_variants(
            base_env=base_env,
            axes=axes,
            strategy=strategy,
            max_variants=max_variants,
        )
        for index, (variant_slug, env, changed_keys) in enumerate(variants):
            experiment_id = f"gpuq_tune_{slugify(space_id)}_{variant_slug}"
            expected_decision = str(
                space.get("expectedDecision")
                or "Promote only if the tuned value improves early bpb, throughput, or artifact headroom without legality risk."
            )
            stop_conditions = space.get("stopConditions")
            if not isinstance(stop_conditions, list):
                stop_conditions = [
                    "Retire if artifact budget exceeds 16MB",
                    "Retire if early validation regresses without throughput or memory-headroom gain",
                ]
            generated.append(
                {
                    "id": experiment_id,
                    "lane": lane,
                    "status": status,
                    "priority": priority_start + index,
                    "expectedSignal": str(space.get("expectedSignal") or expected_decision),
                    "argumentation": {
                        "acceptedArgumentIds": list(space.get("acceptedArgumentIds") or []),
                        "evidenceIds": list(space.get("evidenceIds") or []),
                        "expectedDecision": expected_decision,
                        "stopConditions": stop_conditions,
                    },
                    "gpuPlan": {
                        "firstPass": str(space.get("firstPass") or "1xH100 bounded tuning rehearsal"),
                        "promotionGate": str(space.get("promotionGate") or expected_decision),
                        "eightH100PromotionGate": str(
                            space.get("eightH100PromotionGate")
                            or "Only promote to 8xH100 after one-H100 tuning evidence survives legality and artifact checks."
                        ),
                        "retirementCriteria": str(
                            space.get("retirementCriteria")
                            or "Retire values that are dominated on bpb, throughput, artifact budget, or stability."
                        ),
                    },
                    "candidate": {
                        "id": experiment_id,
                        "family": str(space.get("family") or "parameter_tuning"),
                        "hypothesis": str(space.get("hypothesis") or f"{space_id}_parameter_tuning"),
                        "hypothesisTags": list(space.get("hypothesisTags") or ["parameter_tuning"]),
                        "quantization": "int8-zlib-artifact",
                        "description": (
                            f"Generated tuning variant for {space_id}; "
                            f"changed={','.join(changed_keys) if changed_keys else 'baseline'}."
                        ),
                        "implementation": implementation,
                        "env": env,
                        "tuningMetadata": {
                            "spaceId": space_id,
                            "variant": variant_slug,
                            "changedKeys": changed_keys,
                            "legality": "No pretrained weights or validation-derived state are carried into the timed run; only static hyperparameters are selected.",
                        },
                    },
                }
            )
    return generated


def build_generated_queue(spec: dict[str, Any], experiments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "kind": "parameter-golf-generated-parameter-tuning-gpu-queue",
        "generatedAt": utc_now_iso(),
        "sourceSpec": str(DEFAULT_SPEC_PATH.relative_to(REPO_ROOT)),
        "runCommand": "python fastest/scripts/run_gpu_experiment_queue.py --queue planning/generated_parameter_tuning_gpu_queue.json --auto-tune-batch",
        "legalityPolicy": {
            "allowed": "Search may choose static hyperparameters before a timed run.",
            "forbidden": "Do not carry pretrained weights, validation-derived state, or external data into leaderboard-timed artifacts.",
        },
        "experiments": experiments,
    }


def update_gpu_queue(queue: dict[str, Any], generated: list[dict[str, Any]]) -> dict[str, Any]:
    updated = deepcopy(queue)
    existing = updated.get("experiments")
    if not isinstance(existing, list):
        raise ValueError("GPU queue must contain experiments list")
    generated_by_id = {str(experiment["id"]): experiment for experiment in generated}
    retained = [
        experiment
        for experiment in existing
        if not (isinstance(experiment, dict) and str(experiment.get("id")) in generated_by_id)
    ]
    updated["experiments"] = retained + generated
    updated["generatedAt"] = utc_now_iso()
    return updated


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPU queue packets from bounded parameter tuning spaces.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--update-queue", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    spec = load_json(args.spec)
    experiments = build_tuning_experiments(spec)
    generated_queue = build_generated_queue(spec, experiments)
    if args.dry_run:
        print(json.dumps({"generatedExperimentIds": [experiment["id"] for experiment in experiments]}, indent=2))
        return 0
    write_json(args.output, generated_queue)
    if args.update_queue is not None:
        queue = load_json(args.update_queue)
        write_json(args.update_queue, update_gpu_queue(queue, experiments))
    print(f"generated {len(experiments)} tuning experiment(s)")
    print(f"output={args.output}")
    if args.update_queue is not None:
        print(f"updated_queue={args.update_queue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
