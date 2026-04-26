import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fastest.scripts.run_cpu_subset_experiment as cpu_subset_runner
from fastest.scripts.run_cpu_subset_experiment import (
    append_cpu_subset_evidence,
    build_cpu_subset_env,
    parse_final_val_bpb,
    run_cpu_subset_experiment,
    sanitize_candidate_id,
)


def trust_receipt(subject: str, trusted_root: str, signature_subject: str | None = None) -> dict[str, str]:
    resolved_signature_subject = signature_subject if signature_subject is not None else subject
    return {
        "schema": "parameter-golf-trust-receipt/v1",
        "issuedBy": "parameter-golf-model-factory-trust-authority",
        "subject": subject,
        "trustedRoot": trusted_root,
        "signature": f"local-signed-bundle:{resolved_signature_subject}",
    }


class CpuSubsetExperimentRunnerTest(unittest.TestCase):
    def test_parses_final_roundtrip_metric_from_train_output(self) -> None:
        val_loss, val_bpb = parse_final_val_bpb(
            "step:1/1 val_loss:4.1 val_bpb:4.2\n"
            "final_int8_zlib_roundtrip_exact val_loss:3.98765432 val_bpb:4.01234567\n"
        )

        self.assertEqual(val_loss, 3.98765432)
        self.assertEqual(val_bpb, 4.01234567)

    def test_builds_cpu_bounded_default_environment(self) -> None:
        env = build_cpu_subset_env(
            {
                "candidateId": "tiny",
                "seed": 7,
                "env": {
                    "ITERATIONS": 2,
                    "MODEL_DIM": "64",
                },
            },
            "cpu_subset_tiny",
        )

        self.assertEqual(env["RUN_ID"], "cpu_subset_tiny")
        self.assertEqual(env["SEED"], "7")
        self.assertEqual(env["ITERATIONS"], "2")
        self.assertEqual(env["MODEL_DIM"], "64")
        self.assertEqual(env["VAL_TOKEN_LIMIT"], "8192")
        self.assertEqual(env["MAX_WALLCLOCK_SECONDS"], "600")

    def test_ignores_candidate_override_for_hard_runtime_cap(self) -> None:
        env = build_cpu_subset_env(
            {
                "candidateId": "tiny",
                "seed": 7,
                "env": {
                    "MAX_WALLCLOCK_SECONDS": "1200",
                },
            },
            "cpu_subset_tiny",
        )

        self.assertEqual(env["MAX_WALLCLOCK_SECONDS"], "600")

    def test_accepts_muon_and_adamw_exemption_sweep_factors(self) -> None:
        env = build_cpu_subset_env(
            {
                "candidateId": "muon-sweep",
                "seed": 11,
                "env": {
                    "MATRIX_LR": "0.03",
                    "MUON_BACKEND_STEPS": 7,
                    "MUON_MOMENTUM": "0.93",
                    "MUON_MOMENTUM_WARMUP_START": "0.8",
                    "MUON_MOMENTUM_WARMUP_STEPS": 120,
                    "CONTROL_TENSOR_NAME_PATTERNS": "attn_scale,q_gain,mlp_scale",
                },
            },
            "cpu_subset_muon_sweep",
        )

        self.assertEqual(env["MATRIX_LR"], "0.03")
        self.assertEqual(env["MUON_BACKEND_STEPS"], "7")
        self.assertEqual(env["MUON_MOMENTUM"], "0.93")
        self.assertEqual(env["MUON_MOMENTUM_WARMUP_START"], "0.8")
        self.assertEqual(env["MUON_MOMENTUM_WARMUP_STEPS"], "120")
        self.assertEqual(env["CONTROL_TENSOR_NAME_PATTERNS"], "attn_scale,q_gain,mlp_scale")

    def test_accepts_attention_norm_mode_env_overrides(self) -> None:
        env = build_cpu_subset_env(
            {
                "candidateId": "attn-norm",
                "seed": 17,
                "env": {
                    "ATTN_NORM_MODE": "qk_rmsnorm",
                    "ATTN_NORM_EPS": "1e-5",
                },
            },
            "cpu_subset_attn_norm",
        )

        self.assertEqual(env["ATTN_NORM_MODE"], "qk_rmsnorm")
        self.assertEqual(env["ATTN_NORM_EPS"], "1e-5")

    def test_rejects_non_finite_swiglu_clamp_config_before_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            status_path = root / "campaign_status.json"
            state_path = root / "campaign_state.json"
            run_root = root / "runs"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": [],
                            "trustedControlState": "established",
                            "totalExperiments": 0,
                            "acceptedExperiments": 0,
                            "mostRecentEvidenceId": None,
                        },
                        "recentCompletedExperiments": [],
                        "experimentRecords": [],
                    },
                )
            )
            status_path.write_text(
                json.dumps(
                    {
                        "generatedAt": "2026-04-25T00:00:00Z",
                        "trustedControlState": "established",
                        "latestVerificationClass": None,
                        "missingEvidence": [],
                        "integrityFlags": [],
                        "policyViolations": [],
                        "adjudication": {"decision": "hold", "reason": "seed"},
                    },
                )
            )
            state_path.write_text(
                json.dumps(
                    {
                        "generatedAt": "2026-04-25T00:00:00Z",
                        "candidate": None,
                        "history": [],
                    },
                )
            )

            with (
                patch("fastest.scripts.run_cpu_subset_experiment.subprocess.Popen") as mock_popen,
                patch("fastest.scripts.run_cpu_subset_experiment.regenerate_views"),
            ):
                result = run_cpu_subset_experiment(
                    candidate={
                        "taskId": "FAST-54",
                        "candidateId": "swiglu-invalid",
                        "activationMode": "swiglu",
                        "swigluClampEnabled": True,
                        "swigluLinearClampMin": "nan",
                        "swigluLinearClampMax": 10.0,
                        "swigluGateClampMax": 10.0,
                    },
                    evidence_path=evidence_path,
                    status_path=status_path,
                    state_path=state_path,
                    run_root=run_root,
                )

            self.assertEqual(result["status"], "integrity_error")
            self.assertEqual(result["reasonCode"], "cpu-subset-preflight-validation-failed")
            self.assertIn("non-finite-swigluLinearClampMin", result["validationErrors"])
            mock_popen.assert_not_called()

    def test_rejects_invalid_muon_numeric_config_before_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            status_path = root / "campaign_status.json"
            state_path = root / "campaign_state.json"
            run_root = root / "runs"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": [],
                            "trustedControlState": "established",
                            "totalExperiments": 0,
                            "acceptedExperiments": 0,
                            "mostRecentEvidenceId": None,
                        },
                        "recentCompletedExperiments": [],
                        "experimentRecords": [],
                    },
                )
            )
            status_path.write_text(
                json.dumps(
                    {
                        "generatedAt": "2026-04-25T00:00:00Z",
                        "trustedControlState": "established",
                        "latestVerificationClass": None,
                        "missingEvidence": [],
                        "integrityFlags": [],
                        "policyViolations": [],
                        "adjudication": {"decision": "hold", "reason": "seed"},
                    },
                )
            )
            state_path.write_text(
                json.dumps(
                    {
                        "generatedAt": "2026-04-25T00:00:00Z",
                        "candidate": None,
                        "history": [],
                    },
                )
            )

            with (
                patch("fastest.scripts.run_cpu_subset_experiment.subprocess.Popen") as mock_popen,
                patch("fastest.scripts.run_cpu_subset_experiment.regenerate_views"),
            ):
                result = run_cpu_subset_experiment(
                    candidate={
                        "taskId": "FAST-55",
                        "candidateId": "muon-invalid",
                        "env": {
                            "MATRIX_LR": "nan",
                            "MUON_BACKEND_STEPS": 0,
                            "MUON_MOMENTUM": "1.2",
                            "MUON_MOMENTUM_WARMUP_START": "-0.1",
                            "MUON_MOMENTUM_WARMUP_STEPS": 9,
                            "CONTROL_TENSOR_NAME_PATTERNS": "unknown_tensor",
                        },
                    },
                    evidence_path=evidence_path,
                    status_path=status_path,
                    state_path=state_path,
                    run_root=run_root,
                )

            self.assertEqual(result["status"], "integrity_error")
            self.assertEqual(result["reasonCode"], "cpu-subset-preflight-validation-failed")
            self.assertIn("non-finite-matrixLr", result["validationErrors"])
            self.assertIn("invalid-muonBackendSteps", result["validationErrors"])
            self.assertIn("invalid-muonMomentum", result["validationErrors"])
            self.assertIn("invalid-muonMomentumWarmupStart", result["validationErrors"])
            self.assertIn("invalid-muonMomentumWarmupSchedule", result["validationErrors"])
            self.assertIn("invalid-controlTensorNamePatterns", result["validationErrors"])
            mock_popen.assert_not_called()

    def test_rejects_boolean_muon_numeric_values_before_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            status_path = root / "campaign_status.json"
            state_path = root / "campaign_state.json"
            run_root = root / "runs"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": [],
                            "trustedControlState": "established",
                            "totalExperiments": 0,
                            "acceptedExperiments": 0,
                            "mostRecentEvidenceId": None,
                        },
                        "recentCompletedExperiments": [],
                        "experimentRecords": [],
                    },
                )
            )
            status_path.write_text(
                json.dumps(
                    {
                        "generatedAt": "2026-04-25T00:00:00Z",
                        "trustedControlState": "established",
                        "latestVerificationClass": None,
                        "missingEvidence": [],
                        "integrityFlags": [],
                        "policyViolations": [],
                        "adjudication": {"decision": "hold", "reason": "seed"},
                    },
                )
            )
            state_path.write_text(
                json.dumps(
                    {
                        "generatedAt": "2026-04-25T00:00:00Z",
                        "candidate": None,
                        "history": [],
                    },
                )
            )

            with (
                patch("fastest.scripts.run_cpu_subset_experiment.subprocess.Popen") as mock_popen,
                patch("fastest.scripts.run_cpu_subset_experiment.regenerate_views"),
            ):
                result = run_cpu_subset_experiment(
                    candidate={
                        "taskId": "FAST-55",
                        "candidateId": "muon-bool-invalid",
                        "env": {
                            "MATRIX_LR": True,
                            "MUON_BACKEND_STEPS": 5,
                            "MUON_MOMENTUM": "0.95",
                            "MUON_MOMENTUM_WARMUP_START": "0.85",
                            "MUON_MOMENTUM_WARMUP_STEPS": 2,
                            "CONTROL_TENSOR_NAME_PATTERNS": "attn_scale,q_gain",
                        },
                    },
                    evidence_path=evidence_path,
                    status_path=status_path,
                    state_path=state_path,
                    run_root=run_root,
                )

            self.assertEqual(result["status"], "integrity_error")
            self.assertEqual(result["reasonCode"], "cpu-subset-preflight-validation-failed")
            self.assertIn("invalid-matrixLr", result["validationErrors"])
            mock_popen.assert_not_called()

    def test_sanitizes_candidate_id_for_run_directory(self) -> None:
        self.assertEqual(sanitize_candidate_id("../muon:trial/../../bad"), "muon-trial-bad")
        self.assertEqual(sanitize_candidate_id(""), "candidate")

    def test_timeout_enforcement_uses_process_group_kill_and_fixed_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            status_path = root / "campaign_status.json"
            state_path = root / "campaign_state.json"
            run_root = root / "runs"
            evidence_path.write_text("{}\n")
            status_path.write_text("{}\n")
            state_path.write_text("{}\n")
            spawned: dict[str, object] = {}

            class FakeTimeoutProcess:
                def __init__(self, *args, **kwargs) -> None:
                    spawned["args"] = args
                    spawned["kwargs"] = kwargs
                    self.pid = 4321
                    self.returncode = None

                def communicate(self, timeout: float | None = None):
                    raise subprocess.TimeoutExpired(
                        cmd=spawned["args"][0],
                        timeout=timeout or 0,
                        output="partial-stdout",
                        stderr="partial-stderr",
                    )

                def wait(self, timeout: float | None = None) -> int:
                    self.returncode = -9
                    return self.returncode

            with (
                patch("fastest.scripts.run_cpu_subset_experiment.subprocess.Popen", FakeTimeoutProcess),
                patch("fastest.scripts.run_cpu_subset_experiment.os.getpgid", return_value=4321),
                patch("fastest.scripts.run_cpu_subset_experiment.os.killpg") as mock_killpg,
            ):
                result = run_cpu_subset_experiment(
                    candidate={
                        "taskId": "FAST-51",
                        "candidateId": "../muon:trial",
                        "seed": 7,
                        "env": {"MAX_WALLCLOCK_SECONDS": "1200"},
                    },
                    evidence_path=evidence_path,
                    status_path=status_path,
                    state_path=state_path,
                    run_root=run_root,
                )

            self.assertEqual(result["status"], "timeout_enforced")
            self.assertEqual(result["reasonCode"], "cpu-subset-timeout-enforced")
            self.assertEqual(result["effectiveRuntimeCapSec"], 600)
            self.assertEqual(result["capPolicySource"], "runner-fixed-fast-51")
            self.assertEqual(result["timeoutContainmentMode"], "host-level")
            self.assertEqual(result["processTerminationScope"], "process-group")
            self.assertEqual(result["capOverrideActor"], "none")
            self.assertEqual(result["terminationReason"], "hard-timeout")

            env = spawned["kwargs"]["env"]
            self.assertEqual(env["MAX_WALLCLOCK_SECONDS"], "600")
            self.assertTrue(spawned["kwargs"]["start_new_session"])
            mock_killpg.assert_called_once_with(4321, signal.SIGKILL)

    def test_appends_cpu_subset_evidence_without_benchmark_verified_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": [],
                            "trustedControlState": "established",
                            "totalExperiments": 0,
                            "acceptedExperiments": 0,
                            "mostRecentEvidenceId": None,
                        },
                        "recentCompletedExperiments": [],
                        "experimentRecords": [],
                    },
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_test"
            run_dir.mkdir(parents=True)

            record = append_cpu_subset_evidence(
                candidate={"taskId": "FAST-CPU", "candidateId": "tiny", "seed": 7},
                run_id="cpu_subset_tiny",
                completed_at="2026-04-25T01:00:00Z",
                val_loss=4.1,
                val_bpb=4.2,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=build_cpu_subset_env({"seed": 7}, "cpu_subset_tiny"),
                evidence_path=evidence_path,
            )

            self.assertEqual(record["objectiveMetricName"], "val_bpb")
            self.assertEqual(record["resultClass"], "cpu-subset")
            self.assertEqual(record["verificationClass"], "cpu-subset")
            self.assertNotEqual(record["verificationClass"], "benchmark-verified")
            self.assertEqual(record["promotionRationale"]["decision"], "hold")

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["summary"]["totalExperiments"], 1)
            self.assertEqual(evidence["summary"]["acceptedExperiments"], 1)
            self.assertEqual(evidence["recentCompletedExperiments"], ["exp-tiny"])

    def test_appends_model_factory_lineage_ranking_and_follow_up_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            trusted_registry_path = root / "trusted_parent_lineage_registry.json"
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            trusted_operator_acceptance_registry_path = (
                root / "trusted_operator_risk_acceptance_registry.json"
            )
            acceptance_ref = "risk-accept-fast52-cpu-subset"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                                "runConfig": {
                                    "seed": "1337",
                                    "iterations": 4,
                                    "trainSeqLen": 128,
                                    "trainBatchTokens": 2048,
                                    "valTokenLimit": 8192,
                                    "modelDim": 96,
                                    "numLayers": 1,
                                },
                            }
                        ],
                    },
                )
            )
            trusted_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:lineage-prod-root"],
                            "residualRiskAcceptance": {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-52/notes/operator-risk-acceptance"
                                ),
                                "acceptanceRef": acceptance_ref,
                                "expiresAt": "2099-01-01T00:00:00Z",
                                "monitoringOwner": "model-factory-oncall",
                            },
                            "authorityBinding": {
                                "authorityBoundary": "registry://lineage-prod",
                                "registryRef": "trusted-parent-lineage-registry",
                                "validatedAt": "2026-04-26T00:00:00Z",
                                "trustedRoots": ["registry:lineage-prod-root"],
                            },
                        },
                        "parentLineage": {
                            "exp-parent-cpu": {
                                "parentFrontierId": "deepseek-v4-muon-tuning",
                                "provenanceVerified": True,
                                "receipt": {
                                    "schema": "parameter-golf-trust-receipt/v1",
                                    "issuedBy": "parameter-golf-model-factory-trust-authority",
                                    "subject": "exp-parent-cpu",
                                    "trustedRoot": "registry:lineage-prod-root",
                                    "signature": "cryptographic-signed-bundle:exp-parent-cpu",
                                },
                            }
                        },
                    }
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)
            (run_dir / "deterministic_rerun.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "deterministic-rerun-cpu_subset_child",
                    }
                )
            )
            (run_dir / "seed_variance.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "seed-variance-cpu_subset_child",
                    }
                )
            )
            train_env = build_cpu_subset_env(
                {"seed": 7, "env": {"MODEL_DIM": "128", "VAL_TOKEN_LIMIT": "16384"}},
                "cpu_subset_child",
            )
            runner_verification = cpu_subset_runner._mint_runner_verification(
                run_id="cpu_subset_child",
                completed_at="2026-04-25T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=train_env,
            )
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                            "residualRiskAcceptance": {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-52/notes/operator-risk-acceptance"
                                ),
                                "acceptanceRef": acceptance_ref,
                                "expiresAt": "2099-01-01T00:00:00Z",
                                "monitoringOwner": "model-factory-oncall",
                            },
                            "authorityBinding": {
                                "authorityBoundary": "registry://attestation-prod",
                                "registryRef": "trusted-runner-attestation-registry",
                                "validatedAt": "2026-04-26T00:00:00Z",
                                "trustedRoots": ["registry:attestation-prod-root"],
                            },
                        },
                        "runnerAttestations": {
                            runner_verification["attestationRef"]: {
                                "runId": "cpu_subset_child",
                                "attestationRef": runner_verification["attestationRef"],
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": {
                                    "schema": "parameter-golf-trust-receipt/v1",
                                    "issuedBy": "parameter-golf-model-factory-trust-authority",
                                    "subject": runner_verification["attestationRef"],
                                    "trustedRoot": "registry:attestation-prod-root",
                                    "signature": (
                                        "cryptographic-signed-bundle:"
                                        f"{runner_verification['attestationRef']}"
                                    ),
                                },
                            }
                        },
                    }
                )
            )
            trusted_operator_acceptance_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "authorityBoundary": (
                            "fastest-task-store://parameter-golf/"
                            "tasks/FAST-52/trust/acceptance-registry"
                        ),
                        "acceptedResidualRisks": {
                            acceptance_ref: {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-52/notes/operator-risk-acceptance"
                                ),
                                "monitoringOwner": "model-factory-oncall",
                                "expiresAt": "2099-01-01T00:00:00Z",
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_PARENT_REGISTRY_PATH",
                trusted_registry_path,
            ), patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH",
                trusted_attestation_registry_path,
            ), patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_OPERATOR_RISK_ACCEPTANCE_REGISTRY_PATH",
                trusted_operator_acceptance_registry_path,
            ):
                record = append_cpu_subset_evidence(
                    candidate={
                        "taskId": "FAST-49",
                        "candidateId": "child-cpu",
                        "parentExperimentId": "exp-parent-cpu",
                        "parentFrontierId": "forged-frontier-id",
                        "seed": 7,
                        "env": {"MODEL_DIM": "128", "VAL_TOKEN_LIMIT": "16384"},
                    },
                    run_id="cpu_subset_child",
                    completed_at="2026-04-25T01:00:00Z",
                    val_loss=4.0,
                    val_bpb=4.1,
                    command=["python", "train_gpt.py"],
                    run_dir=run_dir,
                    train_env=train_env,
                    evidence_path=evidence_path,
                )

            self.assertEqual(record["lineage"]["parentExperimentId"], "exp-parent-cpu")
            self.assertEqual(record["lineage"]["parentFrontierId"], "deepseek-v4-muon-tuning")
            self.assertEqual(record["lineageResolution"]["status"], "resolved")
            self.assertEqual(record["attestationResolution"]["status"], "resolved")
            changed_factors = record["lineage"]["changedFactors"]
            self.assertIn(
                {"factor": "MODEL_DIM", "parentValue": "96", "candidateValue": "128"},
                changed_factors,
            )
            self.assertIn(
                {"factor": "VAL_TOKEN_LIMIT", "parentValue": "8192", "candidateValue": "16384"},
                changed_factors,
            )
            self.assertEqual(record["runtimeValidationCaps"]["maxRuntimeSeconds"], 600)
            self.assertEqual(record["runtimeValidationCaps"]["valTokenLimit"], 16384)
            self.assertEqual(record["runtimeValidationCaps"]["validationClass"], "cpu-subset")
            self.assertEqual(record["rankingTable"]["candidateRank"], 1)
            self.assertEqual(record["rankingTable"]["totalCandidates"], 2)
            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "propose-follow-up")
            self.assertEqual(record["modelFactoryDecision"]["retirementDecision"], "retain")
            self.assertEqual(record["followUpTaskProposal"]["lane"], "cheap-screen")
            self.assertEqual(record["followUpTaskProposal"]["sourceLane"], "cpu-subset")
            self.assertTrue(record["followUpTaskProposal"]["taskId"].startswith("FAST-49-"))
            self.assertEqual(record["followUpTaskProposal"]["measuredEvidenceRef"], "evidence-exp-child-cpu")
            self.assertEqual(record["resultClass"], "cpu-subset")
            self.assertEqual(record["verificationClass"], "cpu-subset")

    def test_fails_closed_when_local_receipts_have_no_operator_risk_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            trusted_registry_path = root / "trusted_parent_lineage_registry.json"
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                                "runConfig": {
                                    "seed": "1337",
                                    "iterations": 4,
                                    "trainSeqLen": 128,
                                    "trainBatchTokens": 2048,
                                    "valTokenLimit": 8192,
                                    "modelDim": 96,
                                    "numLayers": 1,
                                },
                            }
                        ],
                    },
                )
            )
            trusted_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:lineage-prod-root"],
                        },
                        "parentLineage": {
                            "exp-parent-cpu": {
                                "parentFrontierId": "deepseek-v4-muon-tuning",
                                "provenanceVerified": True,
                                "receipt": trust_receipt(
                                    "exp-parent-cpu",
                                    "registry:lineage-prod-root",
                                ),
                            }
                        },
                    }
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)
            (run_dir / "deterministic_rerun.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "deterministic-rerun-cpu_subset_child",
                    }
                )
            )
            (run_dir / "seed_variance.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "seed-variance-cpu_subset_child",
                    }
                )
            )
            train_env = build_cpu_subset_env(
                {"seed": 7, "env": {"MODEL_DIM": "128", "VAL_TOKEN_LIMIT": "16384"}},
                "cpu_subset_child",
            )
            runner_verification = cpu_subset_runner._mint_runner_verification(
                run_id="cpu_subset_child",
                completed_at="2026-04-25T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=train_env,
            )
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                        },
                        "runnerAttestations": {
                            runner_verification["attestationRef"]: {
                                "runId": "cpu_subset_child",
                                "attestationRef": runner_verification["attestationRef"],
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    runner_verification["attestationRef"],
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_PARENT_REGISTRY_PATH",
                trusted_registry_path,
            ), patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH",
                trusted_attestation_registry_path,
            ):
                record = append_cpu_subset_evidence(
                    candidate={
                        "taskId": "FAST-52",
                        "candidateId": "child-cpu",
                        "parentExperimentId": "exp-parent-cpu",
                        "seed": 7,
                        "env": {"MODEL_DIM": "128", "VAL_TOKEN_LIMIT": "16384"},
                    },
                    run_id="cpu_subset_child",
                    completed_at="2026-04-25T01:00:00Z",
                    val_loss=4.0,
                    val_bpb=4.1,
                    command=["python", "train_gpt.py"],
                    run_dir=run_dir,
                    train_env=train_env,
                    evidence_path=evidence_path,
                )

            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "hold")
            self.assertEqual(record["verificationStatus"], "verification_failed_trust")
            self.assertIsNone(record["followUpTaskProposal"])
            self.assertEqual(
                record["attestationResolution"]["reasonCode"],
                "attestation-receipt-local-signature-disallowed",
            )

    def test_fails_closed_when_local_receipt_fallback_is_requested_with_operator_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            trusted_operator_acceptance_registry_path = (
                root / "trusted_operator_risk_acceptance_registry.json"
            )
            attestation_ref = "attestation-fast55-local-fallback"
            acceptance_ref = "risk-accept-fast55-cpu-subset"
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                            "residualRiskAcceptance": {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-55/notes/operator-risk-acceptance"
                                ),
                                "acceptanceRef": acceptance_ref,
                                "expiresAt": "2099-01-01T00:00:00Z",
                                "monitoringOwner": "model-factory-oncall",
                            },
                            "authorityBinding": {
                                "authorityBoundary": "registry://attestation-prod",
                                "registryRef": "trusted-runner-attestation-registry",
                                "validatedAt": "2026-04-26T00:00:00Z",
                                "trustedRoots": ["registry:attestation-prod-root"],
                            },
                        },
                        "runnerAttestations": {
                            attestation_ref: {
                                "runId": "cpu_subset_child",
                                "attestationRef": attestation_ref,
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    attestation_ref,
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )
            trusted_operator_acceptance_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "authorityBoundary": (
                            "fastest-task-store://parameter-golf/"
                            "tasks/FAST-55/trust/acceptance-registry"
                        ),
                        "acceptedResidualRisks": {
                            acceptance_ref: {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-55/notes/operator-risk-acceptance"
                                ),
                                "monitoringOwner": "model-factory-oncall",
                                "expiresAt": "2099-01-01T00:00:00Z",
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_OPERATOR_RISK_ACCEPTANCE_REGISTRY_PATH",
                trusted_operator_acceptance_registry_path,
            ):
                resolution = cpu_subset_runner._resolve_runner_attestation(
                    attestation_ref=attestation_ref,
                    run_id="cpu_subset_child",
                    registry_path=trusted_attestation_registry_path,
                    trust_gate_mode="strict",
                )

            self.assertEqual(resolution["status"], "unresolved")
            self.assertEqual(
                resolution["reasonCode"],
                "attestation-receipt-local-signature-disallowed",
            )

    def test_fails_closed_when_operator_fallback_mode_tries_local_signature_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            trusted_operator_acceptance_registry_path = (
                root / "trusted_operator_risk_acceptance_registry.json"
            )
            attestation_ref = "attestation-fast55-local-fallback-operator-mode"
            acceptance_ref = "risk-accept-fast55-cpu-subset"
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                            "residualRiskAcceptance": {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-55/notes/operator-risk-acceptance"
                                ),
                                "acceptanceRef": acceptance_ref,
                                "expiresAt": "2099-01-01T00:00:00Z",
                                "monitoringOwner": "model-factory-oncall",
                            },
                            "authorityBinding": {
                                "authorityBoundary": "registry://attestation-prod",
                                "registryRef": "trusted-runner-attestation-registry",
                                "validatedAt": "2026-04-26T00:00:00Z",
                                "trustedRoots": ["registry:attestation-prod-root"],
                            },
                        },
                        "runnerAttestations": {
                            attestation_ref: {
                                "runId": "cpu_subset_child",
                                "attestationRef": attestation_ref,
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    attestation_ref,
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )
            trusted_operator_acceptance_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "authorityBoundary": (
                            "fastest-task-store://parameter-golf/"
                            "tasks/FAST-55/trust/acceptance-registry"
                        ),
                        "acceptedResidualRisks": {
                            acceptance_ref: {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-55/notes/operator-risk-acceptance"
                                ),
                                "monitoringOwner": "model-factory-oncall",
                                "expiresAt": "2099-01-01T00:00:00Z",
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_OPERATOR_RISK_ACCEPTANCE_REGISTRY_PATH",
                trusted_operator_acceptance_registry_path,
            ):
                resolution = cpu_subset_runner._resolve_runner_attestation(
                    attestation_ref=attestation_ref,
                    run_id="cpu_subset_child",
                    registry_path=trusted_attestation_registry_path,
                    trust_gate_mode="operator-approved-fallback",
                )

            self.assertEqual(resolution["status"], "unresolved")
            self.assertEqual(
                resolution["reasonCode"],
                "attestation-receipt-local-signature-disallowed",
            )

    def test_fails_closed_for_cryptographic_receipt_without_external_proof_or_risk_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            attestation_ref = "attestation-fast52-crypto"
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                        },
                        "runnerAttestations": {
                            attestation_ref: {
                                "runId": "cpu_subset_child",
                                "attestationRef": attestation_ref,
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": {
                                    "schema": "parameter-golf-trust-receipt/v1",
                                    "issuedBy": "parameter-golf-model-factory-trust-authority",
                                    "subject": attestation_ref,
                                    "trustedRoot": "registry:attestation-prod-root",
                                    "signature": f"cryptographic-signed-bundle:{attestation_ref}",
                                },
                            }
                        },
                    }
                )
            )

            resolution = cpu_subset_runner._resolve_runner_attestation(
                attestation_ref=attestation_ref,
                run_id="cpu_subset_child",
                registry_path=trusted_attestation_registry_path,
            )

            self.assertEqual(resolution["status"], "unresolved")
            self.assertEqual(
                resolution["reasonCode"],
                "attestation-receipt-residual-risk-acceptance-missing",
            )

    def test_fails_closed_when_local_receipt_has_no_runtime_authority_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            attestation_ref = "attestation-fast52-local-no-authority-binding"
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                            "residualRiskAcceptance": {
                                "scope": "cpu-subset",
                                "acceptedBy": "operator",
                                "expiresAt": "2099-01-01T00:00:00Z",
                                "monitoringOwner": "model-factory-oncall",
                            },
                        },
                        "runnerAttestations": {
                            attestation_ref: {
                                "runId": "cpu_subset_child",
                                "attestationRef": attestation_ref,
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    attestation_ref,
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )

            resolution = cpu_subset_runner._resolve_runner_attestation(
                attestation_ref=attestation_ref,
                run_id="cpu_subset_child",
                registry_path=trusted_attestation_registry_path,
            )

            self.assertEqual(resolution["status"], "unresolved")
            self.assertEqual(
                resolution["reasonCode"],
                "attestation-receipt-local-signature-disallowed",
            )

    def test_fails_closed_when_local_receipt_uses_non_authoritative_operator_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            attestation_ref = "attestation-fast52-non-authoritative-acceptance"
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                            "residualRiskAcceptance": {
                                "scope": "cpu-subset",
                                "acceptedBy": "operator",
                                "acceptanceRef": "risk-accept-fast52-cpu-subset",
                                "expiresAt": "2099-01-01T00:00:00Z",
                                "monitoringOwner": "model-factory-oncall",
                            },
                            "authorityBinding": {
                                "authorityBoundary": "registry://attestation-prod",
                                "registryRef": "trusted-runner-attestation-registry",
                                "validatedAt": "2026-04-26T00:00:00Z",
                                "trustedRoots": ["registry:attestation-prod-root"],
                            },
                        },
                        "runnerAttestations": {
                            attestation_ref: {
                                "runId": "cpu_subset_child",
                                "attestationRef": attestation_ref,
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    attestation_ref,
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )

            resolution = cpu_subset_runner._resolve_runner_attestation(
                attestation_ref=attestation_ref,
                run_id="cpu_subset_child",
                registry_path=trusted_attestation_registry_path,
            )

            self.assertEqual(resolution["status"], "unresolved")
            self.assertEqual(
                resolution["reasonCode"],
                "attestation-receipt-local-signature-disallowed",
            )

    def test_validates_runtime_operator_acceptance_from_trust_boundary_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            trusted_operator_acceptance_registry_path = (
                root / "trusted_operator_risk_acceptance_registry.json"
            )
            attestation_ref = "attestation-fast52-runtime-acceptance-registry"
            acceptance_ref = "risk-accept-fast52-cpu-subset"
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                            "residualRiskAcceptance": {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-52/notes/operator-risk-acceptance"
                                ),
                                "acceptanceRef": acceptance_ref,
                                "expiresAt": "2099-01-01T00:00:00Z",
                                "monitoringOwner": "model-factory-oncall",
                            },
                            "authorityBinding": {
                                "authorityBoundary": "registry://attestation-prod",
                                "registryRef": "trusted-runner-attestation-registry",
                                "validatedAt": "2026-04-26T00:00:00Z",
                                "trustedRoots": ["registry:attestation-prod-root"],
                            },
                        },
                        "runnerAttestations": {
                            attestation_ref: {
                                "runId": "cpu_subset_child",
                                "attestationRef": attestation_ref,
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    attestation_ref,
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )
            trusted_operator_acceptance_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "authorityBoundary": (
                            "fastest-task-store://parameter-golf/"
                            "tasks/FAST-52/trust/acceptance-registry"
                        ),
                        "acceptedResidualRisks": {},
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_OPERATOR_RISK_ACCEPTANCE_REGISTRY_PATH",
                trusted_operator_acceptance_registry_path,
            ):
                unresolved = cpu_subset_runner._resolve_runner_attestation(
                    attestation_ref=attestation_ref,
                    run_id="cpu_subset_child",
                    registry_path=trusted_attestation_registry_path,
                )

                self.assertEqual(unresolved["status"], "unresolved")
                self.assertEqual(
                    unresolved["reasonCode"],
                    "attestation-receipt-local-signature-disallowed",
                )

                trusted_operator_acceptance_registry_path.write_text(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "authorityBoundary": (
                                "fastest-task-store://parameter-golf/"
                                "tasks/FAST-52/trust/acceptance-registry"
                            ),
                            "acceptedResidualRisks": {
                                acceptance_ref: {
                                    "scope": "cpu-subset",
                                    "acceptedBy": (
                                        "fastest-task-store://parameter-golf/"
                                        "tasks/FAST-52/notes/operator-risk-acceptance"
                                    ),
                                    "monitoringOwner": "model-factory-oncall",
                                    "expiresAt": "2099-01-01T00:00:00Z",
                                }
                            },
                        }
                    )
                )

                resolved = cpu_subset_runner._resolve_runner_attestation(
                    attestation_ref=attestation_ref,
                    run_id="cpu_subset_child",
                    registry_path=trusted_attestation_registry_path,
                )

            self.assertEqual(resolved["status"], "unresolved")
            self.assertEqual(
                resolved["reasonCode"],
                "attestation-receipt-local-signature-disallowed",
            )

    def test_fails_closed_when_runtime_operator_acceptance_registry_boundary_is_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            trusted_operator_acceptance_registry_path = (
                root / "trusted_operator_risk_acceptance_registry.json"
            )
            attestation_ref = "attestation-fast52-runtime-acceptance-local-boundary"
            acceptance_ref = "risk-accept-fast52-cpu-subset"
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                            "residualRiskAcceptance": {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-52/notes/operator-risk-acceptance"
                                ),
                                "acceptanceRef": acceptance_ref,
                                "expiresAt": "2099-01-01T00:00:00Z",
                                "monitoringOwner": "model-factory-oncall",
                            },
                            "authorityBinding": {
                                "authorityBoundary": "registry://attestation-prod",
                                "registryRef": "trusted-runner-attestation-registry",
                                "validatedAt": "2026-04-26T00:00:00Z",
                                "trustedRoots": ["registry:attestation-prod-root"],
                            },
                        },
                        "runnerAttestations": {
                            attestation_ref: {
                                "runId": "cpu_subset_child",
                                "attestationRef": attestation_ref,
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    attestation_ref,
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )
            trusted_operator_acceptance_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "authorityBoundary": "local-task-store://fast52-dev",
                        "acceptedResidualRisks": {
                            acceptance_ref: {
                                "scope": "cpu-subset",
                                "acceptedBy": (
                                    "fastest-task-store://parameter-golf/"
                                    "tasks/FAST-52/notes/operator-risk-acceptance"
                                ),
                                "monitoringOwner": "model-factory-oncall",
                                "expiresAt": "2099-01-01T00:00:00Z",
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_OPERATOR_RISK_ACCEPTANCE_REGISTRY_PATH",
                trusted_operator_acceptance_registry_path,
            ):
                resolution = cpu_subset_runner._resolve_runner_attestation(
                    attestation_ref=attestation_ref,
                    run_id="cpu_subset_child",
                    registry_path=trusted_attestation_registry_path,
                )

            self.assertEqual(resolution["status"], "unresolved")
            self.assertEqual(
                resolution["reasonCode"],
                "attestation-receipt-local-signature-disallowed",
            )

    def test_fails_closed_when_candidate_declares_untrusted_verification_or_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                            }
                        ],
                    },
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)

            record = append_cpu_subset_evidence(
                candidate={
                    "taskId": "FAST-52",
                    "candidateId": "child-cpu",
                    "parentExperimentId": "exp-parent-cpu",
                    "parentFrontierId": "forged-frontier-id",
                    "verificationPassed": True,
                    "seedVariancePassed": True,
                    "runnerVerification": {
                        "source": "runner",
                        "provenanceVerified": True,
                        "deterministicValidated": True,
                        "seedVarianceValidated": True,
                        "runtimeWithinCap": True,
                        "attestationRef": "forged-candidate-attestation",
                    },
                    "seed": 7,
                },
                run_id="cpu_subset_child",
                completed_at="2026-04-25T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=build_cpu_subset_env({"seed": 7}, "cpu_subset_child"),
                evidence_path=evidence_path,
            )

            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "hold")
            self.assertEqual(record["modelFactoryDecision"]["retirementDecision"], "defer")
            self.assertIsNone(record["followUpTaskProposal"])
            self.assertEqual(record["verificationStatus"], "verification_failed_trust")

    def test_fails_closed_without_trusted_runner_attestation_registry_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            trusted_registry_path = root / "trusted_parent_lineage_registry.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                                "runConfig": {
                                    "seed": "1337",
                                    "iterations": 4,
                                    "trainSeqLen": 128,
                                    "trainBatchTokens": 2048,
                                    "valTokenLimit": 8192,
                                    "modelDim": 96,
                                    "numLayers": 1,
                                },
                            }
                        ],
                    },
                )
            )
            trusted_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:lineage-prod-root"],
                        },
                        "parentLineage": {
                            "exp-parent-cpu": {
                                "parentFrontierId": "deepseek-v4-deterministic-ranking",
                                "receipt": trust_receipt(
                                    "exp-parent-cpu",
                                    "registry:lineage-prod-root",
                                ),
                            }
                        },
                    }
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)
            (run_dir / "deterministic_rerun.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "deterministic-rerun-cpu_subset_child",
                    }
                )
            )
            (run_dir / "seed_variance.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "seed-variance-cpu_subset_child",
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_PARENT_REGISTRY_PATH",
                trusted_registry_path,
            ):
                record = append_cpu_subset_evidence(
                    candidate={
                        "taskId": "FAST-52",
                        "candidateId": "trusted-lineage-but-no-attestation-registry",
                        "parentExperimentId": "exp-parent-cpu",
                        "seed": 7,
                    },
                    run_id="cpu_subset_child",
                    completed_at="2026-04-26T01:00:00Z",
                    val_loss=4.0,
                    val_bpb=4.1,
                    command=["python", "train_gpt.py"],
                    run_dir=run_dir,
                    train_env=build_cpu_subset_env({"seed": 7}, "cpu_subset_child"),
                    evidence_path=evidence_path,
                )

            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "hold")
            self.assertEqual(record["verificationStatus"], "verification_failed_trust")
            self.assertIsNone(record["followUpTaskProposal"])
            self.assertIn("provenance", record["runnerVerificationGate"]["reasonCode"])

    def test_fails_closed_when_attestation_registry_uses_run_id_fallback_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            trusted_registry_path = root / "trusted_parent_lineage_registry.json"
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                            }
                        ],
                    }
                )
            )
            trusted_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:lineage-prod-root"],
                        },
                        "parentLineage": {
                            "exp-parent-cpu": {
                                "parentFrontierId": "deepseek-v4-deterministic-ranking",
                                "receipt": trust_receipt(
                                    "exp-parent-cpu",
                                    "registry:lineage-prod-root",
                                ),
                            }
                        },
                    }
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)
            (run_dir / "deterministic_rerun.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "deterministic-rerun-cpu_subset_child",
                    }
                )
            )
            (run_dir / "seed_variance.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "seed-variance-cpu_subset_child",
                    }
                )
            )
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                        },
                        "runnerAttestations": {
                            "cpu_subset_child": {
                                "runId": "cpu_subset_child",
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_PARENT_REGISTRY_PATH",
                trusted_registry_path,
            ), patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH",
                trusted_attestation_registry_path,
            ):
                record = append_cpu_subset_evidence(
                    candidate={
                        "taskId": "FAST-52",
                        "candidateId": "run-id-fallback-attempt",
                        "parentExperimentId": "exp-parent-cpu",
                        "seed": 7,
                    },
                    run_id="cpu_subset_child",
                    completed_at="2026-04-26T01:00:00Z",
                    val_loss=4.0,
                    val_bpb=4.1,
                    command=["python", "train_gpt.py"],
                    run_dir=run_dir,
                    train_env=build_cpu_subset_env({"seed": 7}, "cpu_subset_child"),
                    evidence_path=evidence_path,
                )

            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "hold")
            self.assertEqual(record["verificationStatus"], "verification_failed_trust")
            self.assertIsNone(record["followUpTaskProposal"])
            self.assertEqual(record["attestationResolution"]["reasonCode"], "attestation-unresolved")

    def test_fails_closed_when_attestation_trust_root_policy_allows_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            trusted_registry_path = root / "trusted_parent_lineage_registry.json"
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                            }
                        ],
                    }
                )
            )
            trusted_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:lineage-prod-root"],
                        },
                        "parentLineage": {
                            "exp-parent-cpu": {
                                "parentFrontierId": "deepseek-v4-deterministic-ranking",
                                "receipt": trust_receipt(
                                    "exp-parent-cpu",
                                    "registry:lineage-prod-root",
                                ),
                            }
                        },
                    }
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)
            (run_dir / "deterministic_rerun.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "deterministic-rerun-cpu_subset_child",
                    }
                )
            )
            (run_dir / "seed_variance.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "seed-variance-cpu_subset_child",
                    }
                )
            )
            train_env = build_cpu_subset_env({"seed": 7}, "cpu_subset_child")
            runner_verification = cpu_subset_runner._mint_runner_verification(
                run_id="cpu_subset_child",
                completed_at="2026-04-26T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=train_env,
            )
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": True,
                            "trustedRoots": ["registry:attestation-prod-root"],
                        },
                        "runnerAttestations": {
                            runner_verification["attestationRef"]: {
                                "runId": "cpu_subset_child",
                                "attestationRef": runner_verification["attestationRef"],
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    runner_verification["attestationRef"],
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_PARENT_REGISTRY_PATH",
                trusted_registry_path,
            ), patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH",
                trusted_attestation_registry_path,
            ):
                record = append_cpu_subset_evidence(
                    candidate={
                        "taskId": "FAST-52",
                        "candidateId": "trust-root-bypass-attempt",
                        "parentExperimentId": "exp-parent-cpu",
                        "seed": 7,
                    },
                    run_id="cpu_subset_child",
                    completed_at="2026-04-26T01:00:00Z",
                    val_loss=4.0,
                    val_bpb=4.1,
                    command=["python", "train_gpt.py"],
                    run_dir=run_dir,
                    train_env=train_env,
                    evidence_path=evidence_path,
                )

            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "hold")
            self.assertEqual(record["verificationStatus"], "verification_failed_trust")
            self.assertIsNone(record["followUpTaskProposal"])
            self.assertEqual(
                record["attestationResolution"]["reasonCode"],
                "trusted-attestation-registry-invalid-trust-root-policy-bypass-disallowed",
            )

    def test_fails_closed_when_attestation_registry_entry_is_not_cpu_subset_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            trusted_registry_path = root / "trusted_parent_lineage_registry.json"
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                            }
                        ],
                    }
                )
            )
            trusted_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:lineage-prod-root"],
                        },
                        "parentLineage": {
                            "exp-parent-cpu": {
                                "parentFrontierId": "deepseek-v4-deterministic-ranking",
                                "receipt": trust_receipt(
                                    "exp-parent-cpu",
                                    "registry:lineage-prod-root",
                                ),
                            }
                        },
                    }
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)
            (run_dir / "deterministic_rerun.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "deterministic-rerun-cpu_subset_child",
                    }
                )
            )
            (run_dir / "seed_variance.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "seed-variance-cpu_subset_child",
                    }
                )
            )
            train_env = build_cpu_subset_env({"seed": 7}, "cpu_subset_child")
            runner_verification = cpu_subset_runner._mint_runner_verification(
                run_id="cpu_subset_child",
                completed_at="2026-04-26T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=train_env,
            )
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                        },
                        "runnerAttestations": {
                            runner_verification["attestationRef"]: {
                                "runId": "cpu_subset_child",
                                "attestationRef": runner_verification["attestationRef"],
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "benchmark-verified",
                                "verificationClass": "benchmark-verified",
                                "receipt": trust_receipt(
                                    runner_verification["attestationRef"],
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_PARENT_REGISTRY_PATH",
                trusted_registry_path,
            ), patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH",
                trusted_attestation_registry_path,
            ):
                record = append_cpu_subset_evidence(
                    candidate={
                        "taskId": "FAST-52",
                        "candidateId": "attestation-wrong-class",
                        "parentExperimentId": "exp-parent-cpu",
                        "seed": 7,
                    },
                    run_id="cpu_subset_child",
                    completed_at="2026-04-26T01:00:00Z",
                    val_loss=4.0,
                    val_bpb=4.1,
                    command=["python", "train_gpt.py"],
                    run_dir=run_dir,
                    train_env=train_env,
                    evidence_path=evidence_path,
                )

            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "hold")
            self.assertEqual(record["verificationStatus"], "verification_failed_trust")
            self.assertIsNone(record["followUpTaskProposal"])
            self.assertEqual(
                record["attestationResolution"]["reasonCode"],
                "attestation-result-class-invalid",
            )

    def test_fails_closed_when_parent_lineage_receipt_uses_untrusted_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            trusted_registry_path = root / "trusted_parent_lineage_registry.json"
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                            }
                        ],
                    }
                )
            )
            trusted_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:lineage-prod-root"],
                        },
                        "parentLineage": {
                            "exp-parent-cpu": {
                                "parentFrontierId": "deepseek-v4-deterministic-ranking",
                                "provenanceVerified": True,
                                "receipt": trust_receipt(
                                    "exp-parent-cpu",
                                    "registry:lineage-rogue-root",
                                ),
                            }
                        },
                    }
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)
            (run_dir / "deterministic_rerun.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "deterministic-rerun-cpu_subset_child",
                    }
                )
            )
            (run_dir / "seed_variance.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "seed-variance-cpu_subset_child",
                    }
                )
            )
            train_env = build_cpu_subset_env({"seed": 7}, "cpu_subset_child")
            runner_verification = cpu_subset_runner._mint_runner_verification(
                run_id="cpu_subset_child",
                completed_at="2026-04-26T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=train_env,
            )
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                        },
                        "runnerAttestations": {
                            runner_verification["attestationRef"]: {
                                "runId": "cpu_subset_child",
                                "attestationRef": runner_verification["attestationRef"],
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    runner_verification["attestationRef"],
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_PARENT_REGISTRY_PATH",
                trusted_registry_path,
            ), patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH",
                trusted_attestation_registry_path,
            ):
                record = append_cpu_subset_evidence(
                    candidate={
                        "taskId": "FAST-52",
                        "candidateId": "lineage-untrusted-root",
                        "parentExperimentId": "exp-parent-cpu",
                        "seed": 7,
                    },
                    run_id="cpu_subset_child",
                    completed_at="2026-04-26T01:00:00Z",
                    val_loss=4.0,
                    val_bpb=4.1,
                    command=["python", "train_gpt.py"],
                    run_dir=run_dir,
                    train_env=train_env,
                    evidence_path=evidence_path,
                )

            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "hold")
            self.assertEqual(record["verificationStatus"], "verification_failed_trust")
            self.assertEqual(
                record["lineageResolution"]["reasonCode"],
                "lineage-receipt-trusted-root-untrusted",
            )

    def test_fails_closed_when_parent_lineage_receipt_signature_subject_is_forged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            trusted_registry_path = root / "trusted_parent_lineage_registry.json"
            trusted_attestation_registry_path = root / "trusted_runner_attestation_registry.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                            }
                        ],
                    }
                )
            )
            trusted_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:lineage-prod-root"],
                        },
                        "parentLineage": {
                            "exp-parent-cpu": {
                                "parentFrontierId": "deepseek-v4-deterministic-ranking",
                                "provenanceVerified": True,
                                "receipt": trust_receipt(
                                    "exp-parent-cpu",
                                    "registry:lineage-prod-root",
                                    signature_subject="forged-parent",
                                ),
                            }
                        },
                    }
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)
            (run_dir / "deterministic_rerun.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "deterministic-rerun-cpu_subset_child",
                    }
                )
            )
            (run_dir / "seed_variance.json").write_text(
                json.dumps(
                    {
                        "source": "cpu-subset-runner",
                        "status": "passed",
                        "evidenceRef": "seed-variance-cpu_subset_child",
                    }
                )
            )
            train_env = build_cpu_subset_env({"seed": 7}, "cpu_subset_child")
            runner_verification = cpu_subset_runner._mint_runner_verification(
                run_id="cpu_subset_child",
                completed_at="2026-04-26T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=train_env,
            )
            trusted_attestation_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:attestation-prod-root"],
                        },
                        "runnerAttestations": {
                            runner_verification["attestationRef"]: {
                                "runId": "cpu_subset_child",
                                "attestationRef": runner_verification["attestationRef"],
                                "source": "cpu-subset-runner",
                                "provenanceVerified": True,
                                "resultClass": "cpu-subset",
                                "verificationClass": "cpu-subset",
                                "receipt": trust_receipt(
                                    runner_verification["attestationRef"],
                                    "registry:attestation-prod-root",
                                ),
                            }
                        },
                    }
                )
            )

            with patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_PARENT_REGISTRY_PATH",
                trusted_registry_path,
            ), patch.object(
                cpu_subset_runner,
                "DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH",
                trusted_attestation_registry_path,
            ):
                record = append_cpu_subset_evidence(
                    candidate={
                        "taskId": "FAST-52",
                        "candidateId": "lineage-signature-forged",
                        "parentExperimentId": "exp-parent-cpu",
                        "seed": 7,
                    },
                    run_id="cpu_subset_child",
                    completed_at="2026-04-26T01:00:00Z",
                    val_loss=4.0,
                    val_bpb=4.1,
                    command=["python", "train_gpt.py"],
                    run_dir=run_dir,
                    train_env=train_env,
                    evidence_path=evidence_path,
                )

            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "hold")
            self.assertEqual(record["verificationStatus"], "verification_failed_trust")
            self.assertEqual(
                record["lineageResolution"]["reasonCode"],
                "lineage-receipt-signature-subject-mismatch",
            )

    def test_fails_closed_when_parent_lineage_entry_provenance_is_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trusted_registry_path = root / "trusted_parent_lineage_registry.json"
            trusted_registry_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustRootPolicy": {
                            "environment": "production",
                            "allowBypass": False,
                            "trustedRoots": ["registry:lineage-prod-root"],
                        },
                        "parentLineage": {
                            "exp-parent-cpu": {
                                "parentFrontierId": "deepseek-v4-deterministic-ranking",
                                "provenanceVerified": False,
                                "receipt": trust_receipt(
                                    "exp-parent-cpu",
                                    "registry:lineage-prod-root",
                                ),
                            }
                        },
                    }
                )
            )

            lineage_resolution = cpu_subset_runner._resolve_parent_lineage(
                parent_experiment_id="exp-parent-cpu",
                registry_path=trusted_registry_path,
            )

            self.assertEqual(lineage_resolution["status"], "unresolved")
            self.assertEqual(lineage_resolution["reasonCode"], "lineage-provenance-unverified")
            self.assertIsNone(lineage_resolution["parentFrontierId"])

    def test_excludes_sensitive_env_keys_from_model_factor_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                                "modelFactory": {
                                    "owner": "cpu-subset-runner",
                                    "normalizedFactors": {
                                        "MODEL_DIM": "96",
                                        "OPENAI_API_KEY": "parent-secret",
                                    },
                                },
                            }
                        ],
                    },
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)

            candidate = {
                "taskId": "FAST-49",
                "candidateId": "child-cpu",
                "parentExperimentId": "exp-parent-cpu",
                "seed": 7,
                "env": {
                    "MODEL_DIM": "128",
                    "OPENAI_API_KEY": "child-secret",
                    "AWS_SECRET_ACCESS_KEY": "child-aws-secret",
                },
            }
            train_env = build_cpu_subset_env(candidate, "cpu_subset_child")
            record = append_cpu_subset_evidence(
                candidate=candidate,
                run_id="cpu_subset_child",
                completed_at="2026-04-25T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=train_env,
                evidence_path=evidence_path,
            )

            serialized_factors = record["modelFactory"]["normalizedFactors"]
            self.assertNotIn("OPENAI_API_KEY", serialized_factors)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", serialized_factors)
            self.assertEqual(serialized_factors["MODEL_DIM"], "128")

            changed_factor_names = {entry["factor"] for entry in record["lineage"]["changedFactors"]}
            self.assertNotIn("OPENAI_API_KEY", changed_factor_names)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", changed_factor_names)

    def test_serializes_muon_and_adamw_exemption_factors_into_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                                "modelFactory": {
                                    "owner": "cpu-subset-runner",
                                    "normalizedFactors": {
                                        "MATRIX_LR": "0.04",
                                        "MUON_BACKEND_STEPS": "5",
                                        "MUON_MOMENTUM": "0.95",
                                        "MUON_MOMENTUM_WARMUP_START": "0.85",
                                        "MUON_MOMENTUM_WARMUP_STEPS": "500",
                                        "CONTROL_TENSOR_NAME_PATTERNS": "attn_scale,q_gain",
                                    },
                                },
                            }
                        ],
                    },
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)

            candidate = {
                "taskId": "FAST-51",
                "candidateId": "muon-cpu",
                "parentExperimentId": "exp-parent-cpu",
                "seed": 13,
                "env": {
                    "MATRIX_LR": "0.03",
                    "MUON_BACKEND_STEPS": "7",
                    "MUON_MOMENTUM": "0.93",
                    "MUON_MOMENTUM_WARMUP_START": "0.80",
                    "MUON_MOMENTUM_WARMUP_STEPS": "120",
                    "CONTROL_TENSOR_NAME_PATTERNS": "attn_scale,mlp_scale,q_gain",
                },
            }
            train_env = build_cpu_subset_env(candidate, "cpu_subset_muon_cpu")
            record = append_cpu_subset_evidence(
                candidate=candidate,
                run_id="cpu_subset_muon_cpu",
                completed_at="2026-04-25T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=train_env,
                evidence_path=evidence_path,
            )

            serialized_factors = record["modelFactory"]["normalizedFactors"]
            self.assertEqual(serialized_factors["MATRIX_LR"], "0.03")
            self.assertEqual(serialized_factors["MUON_BACKEND_STEPS"], "7")
            self.assertEqual(serialized_factors["MUON_MOMENTUM"], "0.93")
            self.assertEqual(serialized_factors["MUON_MOMENTUM_WARMUP_START"], "0.80")
            self.assertEqual(serialized_factors["MUON_MOMENTUM_WARMUP_STEPS"], "120")
            self.assertEqual(
                serialized_factors["CONTROL_TENSOR_NAME_PATTERNS"],
                "attn_scale,mlp_scale,q_gain",
            )

            changed_factors = record["lineage"]["changedFactors"]
            self.assertIn(
                {"factor": "MATRIX_LR", "parentValue": "0.04", "candidateValue": "0.03"},
                changed_factors,
            )
            self.assertIn(
                {"factor": "MUON_BACKEND_STEPS", "parentValue": "5", "candidateValue": "7"},
                changed_factors,
            )

    def test_serializes_attention_norm_factors_into_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                                "modelFactory": {
                                    "owner": "cpu-subset-runner",
                                    "normalizedFactors": {
                                        "ATTN_NORM_MODE": "baseline",
                                        "ATTN_NORM_EPS": "1e-06",
                                    },
                                },
                            }
                        ],
                    },
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_attn_norm"
            run_dir.mkdir(parents=True)

            candidate = {
                "taskId": "FAST-53",
                "candidateId": "attn-norm-qk-rmsnorm",
                "parentExperimentId": "exp-parent-cpu",
                "seed": 13,
                "env": {
                    "ATTN_NORM_MODE": "qk_rmsnorm",
                    "ATTN_NORM_EPS": "1e-5",
                },
            }
            train_env = build_cpu_subset_env(candidate, "cpu_subset_attn_norm")
            record = append_cpu_subset_evidence(
                candidate=candidate,
                run_id="cpu_subset_attn_norm",
                completed_at="2026-04-25T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=train_env,
                evidence_path=evidence_path,
            )

            serialized_factors = record["modelFactory"]["normalizedFactors"]
            self.assertEqual(serialized_factors["ATTN_NORM_MODE"], "qk_rmsnorm")
            self.assertEqual(serialized_factors["ATTN_NORM_EPS"], "1e-5")
            self.assertIn(
                {
                    "factor": "ATTN_NORM_MODE",
                    "parentValue": "baseline",
                    "candidateValue": "qk_rmsnorm",
                },
                record["lineage"]["changedFactors"],
            )


if __name__ == "__main__":
    unittest.main()
