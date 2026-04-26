from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable


ModelFactory = Callable[[Any], Any]


@dataclass(frozen=True)
class CandidateSpec:
    id: str
    family: str
    hypothesis: str
    hypothesis_tags: tuple[str, ...]
    quantization: str
    description: str
    implementation: str
    env: dict[str, str] = field(default_factory=dict)
    runnable: bool = True

    def to_runner_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "hypothesis": self.hypothesis,
            "hypothesisTags": list(self.hypothesis_tags),
            "quantization": self.quantization,
            "description": self.description,
            "implementation": self.implementation,
            "runnable": self.runnable,
            "env": dict(self.env),
        }


def _load_module_specs(module_name: str) -> list[CandidateSpec]:
    module = importlib.import_module(module_name)
    specs = module.candidate_specs()
    if not isinstance(specs, list):
        raise TypeError(f"{module_name}.candidate_specs() must return a list")
    return specs


def default_candidates() -> list[CandidateSpec]:
    candidates = []
    for module_name in (
        "candidates.autoregressive.candidates",
        "candidates.moe.candidates",
    ):
        candidates.extend(_load_module_specs(module_name))
    return [candidate for candidate in candidates if candidate.runnable]


def default_candidate_dicts() -> list[dict[str, Any]]:
    return [candidate.to_runner_dict() for candidate in default_candidates()]


def candidate_name_from_env() -> str:
    return os.environ.get("CANDIDATE_IMPL", "autoregressive_gpt").strip() or "autoregressive_gpt"


def get_model_factory(name: str) -> ModelFactory:
    factories: dict[str, str] = {
        "autoregressive_gpt": "candidates.autoregressive.model:create_model",
        "routed_moe_gpt": "candidates.moe.model:create_model",
    }
    target = factories[name]
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    return factory


def build_model(args: Any) -> Any:
    return get_model_factory(candidate_name_from_env())(args)
