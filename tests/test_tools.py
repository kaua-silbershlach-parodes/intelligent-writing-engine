"""Tests for the tools/ utilities: bounded execution, the evidence
gate, the disaster risk matrix, checkpoints, and the strict report
consumers."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import authorization_consumer as consumer
import bounded_capture
import bounded_subprocess
import checkpoint
import disaster_risk_matrix as drm
import matrix_consumer_audit
import operational_evidence_gate as evidence_gate

PY = sys.executable


class BoundedSubprocessTests(unittest.TestCase):
    def test_echo(self):
        r = bounded_subprocess.run_bounded([PY, "-c", "print('hello')"])
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip(), b"hello")
        self.assertIsNone(r.reason)

    def test_timeout(self):
        r = bounded_subprocess.run_bounded(
            [PY, "-c", "import time; time.sleep(5)"], timeout_s=0.5)
        self.assertEqual(r.returncode, 124)
        self.assertEqual(r.reason, "timeout")

    def test_output_limit(self):
        r = bounded_subprocess.run_bounded(
            [PY, "-c", "print('x' * 100000)"], max_output=1024)
        self.assertEqual(r.returncode, 125)
        self.assertEqual(r.reason, "output_limit")

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            bounded_subprocess.run_bounded([PY], timeout_s=0)
        with self.assertRaises(ValueError):
            bounded_subprocess.run_bounded([], timeout_s=1)


class BoundedCaptureTests(unittest.TestCase):
    def test_capture_stdout_and_stderr_count(self):
        c = bounded_capture.capture(
            [PY, "-c", "import sys; print('out'); print('err', file=sys.stderr)"])
        self.assertEqual(c.returncode, 0)
        self.assertEqual(c.stdout.strip(), b"out")
        self.assertEqual(c.stderr_bytes, 4)
        self.assertIsNone(c.failure)

    def test_stdout_limit(self):
        c = bounded_capture.capture(
            [PY, "-c", "print('x' * 100000)"], stdout_limit=1024)
        self.assertEqual(c.returncode, 125)
        self.assertEqual(c.failure, "output_limit")


class DisasterRiskTests(unittest.TestCase):
    def test_scales(self):
        self.assertEqual(drm.resolve_probability("likely"), 0.6)
        self.assertEqual(drm.resolve_probability(0.6), 0.6)
        self.assertEqual(drm.resolve_impact("severe"), 0.8)
        with self.assertRaises(drm.MatrixError):
            drm.resolve_probability("certain-ish")
        with self.assertRaises(drm.MatrixError):
            drm.resolve_probability(0.3)
        with self.assertRaises(drm.MatrixError):
            drm.resolve_probability(True)

    def test_rank_and_gate(self):
        register = drm.parse_register({"risks": [
            {"theme": "Security", "risk": "System compromise",
             "probability": "likely", "impact": "moderate",
             "response_plan": "isolate and rebuild"},
            {"theme": "Environmental", "risk": "Flood"},
        ]})
        result = drm.assess(register)
        self.assertEqual(len(result.ranked), 1)
        self.assertAlmostEqual(result.ranked[0][1], 0.3)
        self.assertEqual(len(result.unassessed), 1)
        gated = drm.gate(register)
        self.assertFalse(gated.passed)  # the flood risk is unassessed
        register2 = drm.parse_register({"risks": [
            {"theme": "Security", "risk": "System compromise",
             "probability": "likely", "impact": "moderate",
             "response_plan": "isolate and rebuild"}]})
        self.assertTrue(drm.gate(register2).passed)

    def test_high_rank_without_plan_blocks(self):
        register = drm.parse_register({"risks": [
            {"theme": "Security", "risk": "DDoS attack",
             "probability": "inevitable", "impact": "critical"}]})
        gated = drm.gate(register)
        self.assertFalse(gated.passed)
        self.assertIn("no response plan", gated.blockers[0][1])

    def test_sample_register_is_an_unassessed_template(self):
        register = drm.parse_register(json.loads(drm.sample_register_json()))
        self.assertEqual(len(register), 17)
        self.assertFalse(drm.gate(register).passed)


def _evidence_package(root: Path) -> dict:
    """A complete, valid evidence manifest inside `root`."""
    def attach(name: str, content: bytes) -> dict:
        (root / name).write_bytes(content)
        return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}

    reviewers = []
    for i, role in enumerate(evidence_gate.ROLES):
        reviewers.append({
            "role": role, "reviewer_id": f"reviewer-{i}",
            "decision": "approve",
            "evidence": [attach(f"review-{i}.txt", f"review {i}".encode())],
        })
    gates = {g: {"decision": "pass",
                 "evidence": [attach(f"gate-{g}.txt", f"gate {g}".encode())]}
             for g in evidence_gate.GATES}
    return {
        "schema_version": 1,
        "release_commit": "a" * 40,
        "environment": "authorized_preproduction",
        "scope_authorized": True,
        "reviewers": reviewers,
        "gates": gates,
        "measured_recovery": {
            "rto_target_seconds": 3600, "rto_measured_seconds": 1800,
            "rpo_target_seconds": 900, "rpo_measured_seconds": 300,
        },
    }


class EvidenceGateTests(unittest.TestCase):
    def test_complete_package_is_ready_but_never_a_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = evidence_gate.assess(_evidence_package(root), root)
            self.assertEqual(result["evidence_structure"],
                             "READY_FOR_EXTERNAL_REVIEW")
            self.assertEqual(result["production_decision"],
                             "NO_GO_UNTIL_INDEPENDENT_AUTHORIZATION")
            self.assertEqual(result["evidence_attachments_hashed"], 12)

    def test_tampered_attachment_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _evidence_package(root)
            (root / "gate-G0.txt").write_bytes(b"tampered")
            with self.assertRaises(evidence_gate.ValidationProblem):
                evidence_gate.assess(manifest, root)

    def test_missing_reviewer_role_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _evidence_package(root)
            manifest["reviewers"] = manifest["reviewers"][:3]
            with self.assertRaises(evidence_gate.ValidationProblem):
                evidence_gate.assess(manifest, root)


class CheckpointTests(unittest.TestCase):
    def test_inventory_counts_the_project(self):
        per_area, files, lines = checkpoint.inventory()
        self.assertGreater(per_area["engine"]["files"], 10)
        self.assertGreater(files, 30)
        self.assertGreater(lines, 1000)

    def test_count_tests(self):
        counts = checkpoint.count_tests()
        self.assertGreaterEqual(counts["files"], 4)
        self.assertGreater(counts["methods"], 50)

    def test_next_seq_starts_at_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = checkpoint.CHECKPOINTS
            checkpoint.CHECKPOINTS = Path(tmp)
            try:
                self.assertEqual(checkpoint.next_seq(), 1)
            finally:
                checkpoint.CHECKPOINTS = original


class AuthorizationConsumerTests(unittest.TestCase):
    def test_gate_go_accepted_never_production(self):
        raw = json.dumps({"decision": "go", "file_names_logged": False,
                          "secret_values_logged": False}).encode()
        c = consumer.parse_response(raw, 0, channel="publication",
                                    intent="operation")
        self.assertEqual(c.disposition, "ACCEPT_GATE_REPORT")
        self.assertTrue(c.preview_accepted)
        self.assertFalse(c.production_authorized)

    def test_gate_no_go_rejected(self):
        raw = json.dumps({"decision": "no_go"}).encode()
        c = consumer.parse_response(raw, 0, channel="publication",
                                    intent="operation")
        self.assertEqual(c.disposition, "REJECT_DENIED")

    def test_strict_parsing(self):
        c = consumer.parse_response(b'{"decision": "go", "decision": "go"}', 0,
                                    channel="publication", intent="operation")
        self.assertEqual(c.disposition, "REJECT_INVALID_ENVELOPE")
        c = consumer.parse_response(b'{"decision": "go"}', 1,
                                    channel="publication", intent="operation")
        self.assertEqual(c.disposition, "REJECT_PROCESS_FAILURE")
        c = consumer.parse_response(b'{"decision": "go", "file_names_logged": true}',
                                    0, channel="publication", intent="operation")
        self.assertEqual(c.disposition, "REJECT_EFFECTS")

    def test_assurance_channel(self):
        raw = json.dumps({"ok": True, "checks": {"a": True, "b": True}}).encode()
        c = consumer.parse_response(raw, 0, channel="assurance", intent="operation")
        self.assertEqual(c.disposition, "ACCEPT_ASSURANCE_REPORT")
        raw = json.dumps({"ok": True, "checks": {"a": True, "b": False}}).encode()
        c = consumer.parse_response(raw, 0, channel="assurance", intent="operation")
        self.assertEqual(c.disposition, "REJECT_INCONSISTENT")
        raw = json.dumps({"ok": False, "checks": {"a": False}}).encode()
        c = consumer.parse_response(raw, 0, channel="assurance", intent="operation")
        self.assertEqual(c.disposition, "REJECT_DENIED")

    def test_scenario_recorded(self):
        raw = json.dumps({"ok": True, "decision": "no_go",
                          "scenario": "staged-secret"}).encode()
        c = consumer.parse_response(raw, 0, channel="publication",
                                    intent="scenario",
                                    expected_scenario="staged-secret")
        self.assertEqual(c.disposition, "SCENARIO_RECORDED")

    def test_invoke_pinned_gate(self):
        gate = Path(__file__).resolve().parents[1] / "tools" / "git_publication_gate.py"
        digest = hashlib.sha256(gate.read_bytes()).hexdigest()
        c = consumer.invoke_report(gate, ["--staged"], channel="publication",
                                   intent="operation", sha256=digest)
        self.assertEqual(c.disposition, "ACCEPT_GATE_REPORT")
        c = consumer.invoke_report(gate, ["--staged"], channel="publication",
                                   intent="operation", sha256="0" * 64)
        self.assertEqual(c.disposition, "REJECT_ARTIFACT_MISMATCH")
        c = consumer.invoke_report(gate, ["--bogus"], channel="publication",
                                   intent="operation", sha256=digest)
        self.assertEqual(c.disposition, "REJECT_ROUTE_MISMATCH")


class MatrixConsumerTests(unittest.TestCase):
    def test_matrix_is_go_with_no_unsafe_acceptance(self):
        report = matrix_consumer_audit.audit_engine()
        self.assertTrue(report["security_gate_go"])
        self.assertEqual(report["unsafe_acceptances"], 0)
        self.assertEqual(report["valid_preview_rejected"], 0)
        self.assertEqual(report["cases"], 9)
        self.assertFalse(report["production_authorized"])


if __name__ == "__main__":
    unittest.main()
