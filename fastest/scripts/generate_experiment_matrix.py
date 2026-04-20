#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = ROOT / "fastest" / "generated"
LEADERBOARD_PATH = GENERATED_DIR / "leaderboard_snapshot.json"
OUTPUT_PATH = GENERATED_DIR / "experiment_matrix.json"


def build_experiments() -> list[dict[str, object]]:
    return [
        {
            "id": "control-baseline-qk150",
            "stage": "local-screen",
            "summary": "Control run on the shipped baseline without parallel residuals.",
            "env": {
                "QK_GAIN_INIT": "1.5",
                "PARALLEL_RESIDUAL": "0",
            },
            "expected_signal": "Reconfirm local harness behavior and establish a no-change control.",
        },
        {
            "id": "qk500-no-parallel",
            "stage": "local-screen",
            "summary": "Isolate the public QK gain motif without changing residual topology.",
            "env": {
                "QK_GAIN_INIT": "5.0",
                "PARALLEL_RESIDUAL": "0",
            },
            "expected_signal": "Measure whether higher QK gain alone is directionally useful in the current code path.",
        },
        {
            "id": "qk525-no-parallel",
            "stage": "local-screen",
            "summary": "Probe the current winning public QK gain range without parallel residuals.",
            "env": {
                "QK_GAIN_INIT": "5.25",
                "PARALLEL_RESIDUAL": "0",
            },
            "expected_signal": "Check whether the latest winning QK gain transfers before composing additional changes.",
        },
        {
            "id": "qk500-parallel-residual",
            "stage": "local-screen",
            "summary": "Compose higher QK gain with the newly added parallel residual path.",
            "env": {
                "QK_GAIN_INIT": "5.0",
                "PARALLEL_RESIDUAL": "1",
            },
            "expected_signal": "Detect whether parallel residuals improve stability or early validation direction locally.",
        },
        {
            "id": "qk525-parallel-residual",
            "stage": "local-screen",
            "summary": "Public-winning QK gain composed with parallel residuals.",
            "env": {
                "QK_GAIN_INIT": "5.25",
                "PARALLEL_RESIDUAL": "1",
            },
            "expected_signal": "Confirm that the higher-gain parallel path remains stable before adding recurrence.",
        },
        {
            "id": "qk525-parallel-mini-recurrence",
            "stage": "local-screen",
            "summary": "Add a lightweight mid-stack recurrence loop on top of the current best local composition.",
            "env": {
                "QK_GAIN_INIT": "5.25",
                "PARALLEL_RESIDUAL": "1",
                "ENCODER_LAYER_ORDER": "0,1,2,3,2",
                "DECODER_LAYER_ORDER": "3,2,4,5,6,7,8",
            },
            "expected_signal": "Check whether a compact recurrence schedule improves directionally without obvious local instability.",
        },
        {
            "id": "qk525-parallel-deeper-recurrence",
            "stage": "remote-candidate",
            "summary": "Escalate the recurrence schedule once the mini loop is non-regressive.",
            "env": {
                "QK_GAIN_INIT": "5.25",
                "PARALLEL_RESIDUAL": "1",
                "ENCODER_LAYER_ORDER": "0,1,2,3,2,3",
                "DECODER_LAYER_ORDER": "3,2,3,4,5,6,7,8",
            },
            "expected_signal": "Best first remote candidate if local screens do not show obvious regressions.",
        },
    ]


def main() -> None:
    leaderboard = {}
    if LEADERBOARD_PATH.exists():
        leaderboard = json.loads(LEADERBOARD_PATH.read_text())
    payload = {
        "objective": "Translate public-winning motifs into concrete env sweeps for the local-to-remote loop.",
        "leaderboard_reference": leaderboard.get("best_entry"),
        "experiments": build_experiments(),
    }
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with {len(payload['experiments'])} experiments")


if __name__ == "__main__":
    main()
