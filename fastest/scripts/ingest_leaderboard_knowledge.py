from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
RECORDS_DIR = REPO_ROOT / "records"
OUTPUT_PATH = REPO_ROOT / "fastest" / "source" / "leaderboard_knowledge.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"submission root must be object: {path}")
    return payload


def _extract_score(text: str) -> float | None:
    match = re.search(r"([0-9]+\.[0-9]+)", text)
    return float(match.group(1)) if match else None


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_readme_leaderboard(readme_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_track = "10min_16mb"
    for line in readme_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#### Unlimited Compute"):
            current_track = "non_record_16mb"
        if not stripped.startswith("|") or stripped.startswith("|-----"):
            continue
        cells = _split_table_row(stripped)
        if len(cells) < 6 or cells[0] == "Run":
            continue
        score = _extract_score(cells[1])
        if score is None:
            continue
        rows.append(
            {
                "name": cells[0],
                "score": score,
                "author": cells[2],
                "summary": cells[3],
                "date": cells[4],
                "track": current_track,
                "source": "README.md",
            }
        )
    return rows


def _submission_paths(records_dir: Path) -> list[Path]:
    return sorted(records_dir.glob("**/submission*.json"))


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def parse_submission_records(records_dir: Path = RECORDS_DIR) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _submission_paths(records_dir):
        payload = _load_json(path)
        score = payload.get("val_bpb")
        if not isinstance(score, (int, float)):
            continue
        rel_path = _display_path(path, records_dir.parent)
        records.append(
            {
                "name": str(payload.get("name") or path.parent.name),
                "score": float(score),
                "author": str(payload.get("author") or payload.get("github_id") or "unknown"),
                "date": str(payload.get("date") or ""),
                "track": str(payload.get("track") or path.parts[-3]),
                "artifactBytes": payload.get("bytes_total"),
                "trainTimeSeconds": payload.get("train_time_seconds"),
                "techniqueSummary": str(payload.get("technique_summary") or ""),
                "compliance": payload.get("compliance") if isinstance(payload.get("compliance"), dict) else {},
                "source": rel_path,
            }
        )
    return sorted(records, key=lambda record: (record["score"], record["date"], record["name"]))


def _motif_terms(text: str) -> list[str]:
    terms = {
        "sp8192": "SP8192",
        "depth_recurrence": "recurrence",
        "parallel_residuals": "parallel residual",
        "legal_ttt": "ttt",
        "gptq": "gptq",
        "sdclip": "sdclip",
        "qk_gain": "qk",
        "int6": "int6",
        "ema": "ema",
        "swa": "swa",
    }
    lower = text.lower()
    return [name for name, needle in terms.items() if needle in lower]


def summarize_motifs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    motif_scores: dict[str, list[float]] = {}
    for record in records:
        text = " ".join(
            [
                str(record.get("name", "")),
                str(record.get("summary", "")),
                str(record.get("techniqueSummary", "")),
            ]
        )
        for motif in _motif_terms(text):
            motif_scores.setdefault(motif, []).append(float(record["score"]))
    return [
        {
            "motif": motif,
            "recordCount": len(scores),
            "bestScore": min(scores),
        }
        for motif, scores in sorted(
            motif_scores.items(),
            key=lambda item: (min(item[1]), item[0]),
        )
    ]


def build_leaderboard_knowledge(
    *,
    readme_path: Path = README_PATH,
    records_dir: Path = RECORDS_DIR,
) -> dict[str, Any]:
    readme_records = parse_readme_leaderboard(readme_path.read_text())
    submission_records = parse_submission_records(records_dir)
    all_records = sorted(
        [*submission_records, *readme_records],
        key=lambda record: (float(record["score"]), str(record.get("date", ""))),
    )
    best = all_records[0] if all_records else None
    return {
        "schemaVersion": 1,
        "kind": "parameter-golf-leaderboard-knowledge",
        "sources": {
            "readme": _display_path(readme_path, readme_path.parent),
            "recordsDir": _display_path(records_dir, records_dir.parent),
            "submissionRecordCount": len(submission_records),
            "readmeRecordCount": len(readme_records),
        },
        "summary": {
            "totalRecords": len(all_records),
            "bestScore": best["score"] if best else None,
            "bestRecord": best["name"] if best else None,
            "bestRecordSource": best["source"] if best else None,
        },
        "records": all_records,
        "motifs": summarize_motifs(all_records),
    }


def write_leaderboard_knowledge(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    knowledge = build_leaderboard_knowledge()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(knowledge, indent=2)}\n")
    return knowledge


def main() -> None:
    knowledge = write_leaderboard_knowledge()
    print(
        "leaderboard_knowledge:"
        f"records:{knowledge['summary']['totalRecords']} "
        f"best:{knowledge['summary']['bestScore']} "
        f"output:{OUTPUT_PATH.relative_to(REPO_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
