# FAST-46 Public PR Triage Evidence Note

Date: 2026-04-24
Campaign lane: `target-repo-campaign`
Triage scope: open public PRs claiming `val_bpb < 1.0785`

## Repro commands
1. `web.open https://github.com/openai/parameter-golf/pull/1698`
2. `web.open https://github.com/openai/parameter-golf/pull/1722`
3. `web.open https://github.com/openai/parameter-golf/pull/1795`
4. `web.open https://github.com/openai/parameter-golf/pull/1797`
5. `web.open https://github.com/openai/parameter-golf/pull/1801`
6. `web.open https://github.com/openai/parameter-golf/pull/1807`
7. `web.open https://github.com/openai/parameter-golf/pulls`

## Observed results
- PR [#1698](https://github.com/openai/parameter-golf/pull/1698): claim `1.00995`; blocked by posted artifact-size cap issue and canonical byte-accounting dispute in thread.
- PR [#1722](https://github.com/openai/parameter-golf/pull/1722): claim `0.65802`; method title includes `SLOT v3 + Pre-Quant TTT`, treated as illegal for current winning-path gate without explicit organizer clearance.
- PR [#1795](https://github.com/openai/parameter-golf/pull/1795): claim `1.01252`; author marks legality as pending organizer ruling for byte-level PPM online mixture.
- PR [#1797](https://github.com/openai/parameter-golf/pull/1797): claim `1.06157`; strongest inspected legal/runnable path with complete 3-seed and under-cap artifact reporting.
- PR [#1801](https://github.com/openai/parameter-golf/pull/1801): claim `1.06287`; legal backup candidate, slightly weaker than #1797.
- PR [#1807](https://github.com/openai/parameter-golf/pull/1807): claim `1.07037`; lower score and pre-quant TTT interpretation risk.

## Promotion outcome
Decision: advance PR #1797.

Rationale: #1797 is the best candidate that currently clears legal and runnable gates with complete evidence. Lower-claimed entries (#1698, #1722, #1795) remain blocked/illegal/unverified for immediate winning-path promotion.

## Artifact paths
- `planning/fast46_public_pr_triage_execution_packet.json`
- `fastest/source/fast-46-public-pr-triage-evidence.md`
- `tests/test_fast46_public_pr_triage_packet.py`
