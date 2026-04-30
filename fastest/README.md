# Fastest Competition Workspace

This directory holds Fastest project context for the competition target at `/workspaces/parameter-golf`.

Purpose:
- keep competition charter close to the target repo
- preserve operator intent and hard constraints
- leave a stable place for generated evidence and experiment notes

Bootstrap outputs:
- `project-charter.md`: canonical competition brief and merge policy
- `generated/`: machine-generated evidence and ranked idea outputs
- `logs/`: experiment and learning notes

Core rule reminders:
- 16MB artifact limit
- leaderboard submissions target 10 minutes on 8xH100s
- FineWeb validation bits-per-byte decides score
- current bootstrap deadline: 2026-04-30

Suggested next commands:
```bash
npm run env:local:start -- --project-dir "/workspaces/parameter-golf"
npm run fast -- --project-dir "/workspaces/parameter-golf" operator-status
npm run fast -- --project-dir "/workspaces/parameter-golf" task list --format text
```
