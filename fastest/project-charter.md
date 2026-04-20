# Parameter Golf Project Charter

## Objective

Build a repeatable research-and-execution loop that improves this fork's competitive `val_bpb` on OpenAI Parameter Golf while staying aligned with the challenge rules.

## Success Condition

The project is succeeding when it produces:

- a trustworthy local knowledge base of prior ideas and results
- a continuously refreshed ranked backlog of candidate experiments
- reproducible submission-path improvements
- durable learning records that reduce repeated dead-end work

## Hard Constraints

- leaderboard submissions must fit in a 16,000,000 byte total artifact
- leaderboard submissions must train in under 10 minutes on 8xH100s
- evaluation must remain within the spirit and letter of the challenge rules
- non-record exploratory runs are allowed, but they must be labeled explicitly

## Local Machine Policy

This machine is an early-experiment and control-plane host, not the final submission machine.

Use it for:

- record mining
- config comparison
- artifact-budget checks
- dry-run submission assembly
- local smoke tests
- cheap screening tasks

Do not mistake weak local hardware for signal that an idea is bad.
Use local runs to reject obviously broken ideas, not to overfit on CPU throughput.

## Merge Policy

Merge freely:

- record-mining automation
- reproducibility fixes
- size-budget tooling
- legality and benchmark harness improvements

Merge cautiously:

- architectural or optimizer changes without strong evidence
- changes that create complexity faster than they create information

Keep isolated:

- non-record exploration
- high-cost speculative combinations
- rule-uncertain evaluation paths

## Preferred Idea Sources

- motifs that recur across the top leaderboard entries
- improvements that repeatedly compose with other wins
- underexplored pairings near the current best stack
- local harness gaps that block rapid iteration
- failure analyses from prior non-record experiments
