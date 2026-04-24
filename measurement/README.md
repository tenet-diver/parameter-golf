# Control Measurement Contract

This directory implements the FAST-1 control measurement pipeline for `parameter-golf`.

## Boundaries
- Controller boundary: call `submitControlRun(spec_ref, campaign_tick_id, evidence_store)`.
- Measurement runner boundary: call `publishEvidenceBundle(submission, seed_metrics, log_files)`.
- Trust evaluator boundary: call `evaluateControlTrust(bundle_path, spec_ref)` and gate promotions from its result only.
- Source-of-truth boundary: evidence bundles in the evidence store are authoritative; generated campaign status files are derived read models.

## Versioned Reproducibility Inputs
- Canonical spec: `measurement/control_measurement.yaml`.
- Spec includes fixed seed set, dataset/tokenizer IDs, metric tolerance, command template, and required environment constraints.
- Spec fingerprint (`spec_hash`) is calculated from canonical JSON.

## Immutable Evidence Outputs
Each bundle contains:
- `bundle_id`, `spec_hash`, `campaign_tick_id`.
- Environment manifest and environment incompatibility findings.
- Per-seed metrics and aggregate `primary_metric`.
- Artifact checksums for spec/log inputs.

## Trust Decisions
`evaluateControlTrust` returns `trusted` only when all checks pass:
- Schema version compatibility.
- Spec hash and dataset/tokenizer fingerprint match.
- Artifact checksums are present and valid.
- Seed metrics are finite and within drift tolerance.
- Aggregate metric is finite.

## Recovery + Idempotency
- Duplicate submissions with the same `(campaign_tick_id, spec_hash)` reuse one `bundle_id`.
- Reused submissions can finalize incomplete evidence bundles by publishing missing metrics/checksums to the same bundle ID.
