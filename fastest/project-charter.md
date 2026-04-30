# Parameter Golf Project Charter

Target repo: /workspaces/parameter-golf
Deadline: 2026-04-30

Goal statement: Win or place competitively in OpenAI Parameter Golf by discovering, validating, and shipping model, training, and compression improvements faster than manual iteration alone.
Target outcome: Produce at least 1 reproducible submission run that stays within the 16MB artifact limit, preserves a credible path toward 10 minutes on 8xH100, and records reusable learnings before 2026-04-30.

Constraints:
- Treat upstream competition rules as hard constraints.
- Preserve a reproducible baseline before speculative changes.
- Keep every merged change tied to benchmark evidence, artifact size evidence, and a written learning summary.
- Prefer experiments that can be screened cheaply before expensive training runs.
- Separate leaderboard-legal runs from non-record exploratory runs.
- Keep backlog generation evidence-driven instead of free-form brainstorming.

Non-goals:
- Do not optimize unrelated product polish or generic infrastructure work.
- Do not merge changes that improve one metric while violating artifact size or reproducibility constraints.
- Do not let speculative high-cost ideas starve cheap, high-information experiments.
- Do not treat raw novelty as success without benchmark evidence.

Merge posture:
- merge measurement and reproducibility improvements first
- merge speculative model changes only after benchmark and size evidence
- isolate non-record exploration until promoted deliberately
