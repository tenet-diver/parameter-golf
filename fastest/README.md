# Fastest Autonomy For Parameter Golf

This directory is the local autonomy layer for running `parameter-golf` as a Fastest-driven project.

The goal is not to replace the upstream challenge workflow. The goal is to:

- mine the existing competition record history into machine-usable backlog inputs
- keep a project charter and decision boundary close to the code
- rank new experiments by expected information gain, not novelty alone
- record learnings so the next experiment wave is better than the last
- support cheap local screening on a non-H100 machine before expensive remote runs

## Files

- `project-charter.md`: project objective, constraints, and merge posture
- `generated/`: mined leaderboard snapshots, motif summaries, and candidate backlog
- `logs/experiment-log.jsonl`: append-only experiment and learning ledger
- `scripts/mine_records.py`: converts `records/**/submission.json` into a ranked local backlog
- `scripts/log_experiment.py`: appends a structured experiment result to the learning log
- `scripts/check_size_budget.py`: cheap code-size and artifact-budget check

## Local Workflow

1. Refresh the mined knowledge:

```bash
python3 fastest/scripts/mine_records.py
```

2. Inspect the generated backlog:

```bash
python3 - <<'PY'
import json
from pathlib import Path
path = Path('fastest/generated/candidate_backlog.json')
data = json.loads(path.read_text())
for item in data['tasks'][:10]:
    print(f"{item['priority']:>2} | {item['category']:<12} | {item['title']}")
PY
```

3. Record each completed experiment or rejected idea:

```bash
python3 fastest/scripts/log_experiment.py \
  --title "Example recurrence ablation" \
  --category experiment \
  --status failed \
  --summary "No useful local signal; deferred to remote GPU runs." \
  --idea recurrence \
  --idea parallel_residuals \
  --next-step "Keep recurrence fixed and test QK gain variants instead."
```

4. Check cheap artifact budget constraints before expensive work:

```bash
python3 fastest/scripts/check_size_budget.py train_gpt.py --target-total-bytes 16000000
```

## Operating Principle

Machine-generated backlog is allowed.

Machine-generated merges are not.

The system should freely create:

- analysis tasks
- experiment tasks
- validation tasks
- harness and reproducibility tasks

But promotion into the main competitive path still requires:

- reproducible evidence
- legal or clearly marked non-record classification
- artifact-size compliance
- a written learning summary
