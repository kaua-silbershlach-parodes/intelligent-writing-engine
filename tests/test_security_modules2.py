"""Tests for the lifecycle, agent, change, AI-assurance, infrastructure,
vector-security and rate-limit modules.

Every scenario in every module must be blocked at exactly the gate it
names — the same completeness contract the fixtures encode.
"""
import random
import unittest

from engine import agentsec, aiassurance, change, infrastructure
from engine import lifecycle, ratelimit, vectorsec


class LifecycleTests(unittest.TestCase):
    def test_baseline_advances(self):
        self.assertTrue(lifecycle.baseline().allowed)

    def test_every_scenario_blocked_at_its_gate(self):
        for scenario_id, gate in lifecycle.SCENARIOS:
            with self.subTest(scenario=scenario_id):
                d = lifecycle.simulate(scenario_id)
                self.assertFalse(d.allowed)
                self.assertEqual(d.failed_gate, gate)

    def test_full_chain_of_transitions(self):
        evidence = {
            ("scoping", "threat_modeling"): "scope_verified",
            ("threat_modeling", "adversarial_preview"): "threats_reviewed",
            ("adversarial_preview", "analysis_reporting"): "tests_reproduced",
            ("analysis_reporting", "remediation"): "findings_correlated",
            ("remediation", "independent_retest"): "fix_reviewed",
            ("independent_retest", "closed"): "retest_passed",
        }
        for (cur, nxt), ev in evidence.items():
            d = lifecycle.check("reviewer", "same_tenant", cur, nxt, ev,
                                "approved_read_only", "redacted_trace")
            self.assertTrue(d.allowed, (cur, nxt))

    def test_unknown_scenario(self):
        self.assertEqual(lifecycle.simulate("nope").failed_gate, "unknown")


class AgentSecTests(unittest.TestCase):
    def test_untrusted_sources_read_fixtures_only(self):
        for source in ("document", "web_page", "tool_output", "user_input"):
            d = agentsec.authorize(source, "read_fixture", "viewer",
                                   "same_session", "none")
            self.assertTrue(d.allowed, source)
            d = agentsec.authorize(source, "execute_tool", "reviewer",
                                   "same_session", "approved")
            self.assertFalse(d.allowed, source)
            self.assertEqual(d.reason, "untrusted_data_cannot_authorize_actions")

    def test_sensitive_egress_always_disabled(self):
        for action in ("reveal_secret", "external_egress"):
            d = agentsec.authorize("trusted_application", action, "reviewer",
                                   "same_session", "approved")
            self.assertEqual(d.reason, "sensitive_egress_disabled")

    def test_session_boundary(self):
        d = agentsec.authorize("trusted_application", "read_fixture",
                               "reviewer", "cross_session", "approved")
        self.assertEqual(d.reason, "session_boundary")

    def test_memory_write_requires_review(self):
        d = agentsec.authorize("trusted_application", "write_shared_memory",
                               "operator", "same_session", "approved")
        self.assertEqual(d.reason, "memory_write_requires_review")
        d = agentsec.authorize("trusted_application", "write_shared_memory",
                               "reviewer", "same_session", "approved")
        self.assertTrue(d.allowed)

    def test_tool_execution_role_and_approval(self):
        d = agentsec.authorize("trusted_application", "execute_tool",
                               "viewer", "same_session", "approved")
        self.assertEqual(d.reason, "insufficient_role")
        d = agentsec.authorize("trusted_application", "execute_tool",
                               "operator", "same_session", "none")
        self.assertEqual(d.reason, "approval_required")
        d = agentsec.authorize("trusted_application", "execute_tool",
                               "operator", "same_session", "approved")
        self.assertTrue(d.allowed)

    def test_invalid_input(self):
        d = agentsec.authorize("carrier_pigeon", "read_fixture", "viewer",
                               "same_session", "none")
        self.assertEqual(d.reason, "invalid_policy_input")


class ChangeTests(unittest.TestCase):
    def test_baseline_allowed(self):
        self.assertTrue(change.check(change.baseline_fixture()).allowed)

    def test_every_scenario_blocked_at_its_gate(self):
        for scenario_id, _domain, gate in change.SCENARIOS:
            with self.subTest(scenario=scenario_id):
                d = change.simulate(scenario_id)
                self.assertFalse(d.allowed)
                self.assertEqual(d.failed_gate, gate)

    def test_shape_and_offline(self):
        from engine.change import Evidence
        self.assertEqual(change.check(Evidence(gates=())).failed_gate, "shape")
        self.assertEqual(change.check(change.bad_fixture("offline")).failed_gate,
                         "offline")

    def test_impact_requires_documentation(self):
        ok = change.impact("reviewer", "documented", "documented", "bounded",
                           "unknown_not_invented", "reviewed", "documented")
        self.assertTrue(ok.allowed)
        bad = change.impact("reviewer", "documented", "unknown", "bounded",
                            "unknown_not_invented", "reviewed", "documented")
        self.assertFalse(bad.allowed)


class AiAssuranceTests(unittest.TestCase):
    def test_baseline_allowed(self):
        self.assertTrue(aiassurance.check(aiassurance.baseline_fixture()).allowed)

    def test_every_scenario_blocked_at_its_gate(self):
        self.assertEqual(len(aiassurance.SCENARIOS), 24)
        for row in aiassurance.SCENARIOS:
            with self.subTest(scenario=row[0]):
                d = aiassurance.simulate(row[0])
                self.assertFalse(d.allowed)
                self.assertEqual(d.failed_gate, row[4])

    def test_no_live_access_fixture(self):
        d = aiassurance.check(aiassurance.bad_fixture("no_live_access"))
        self.assertEqual(d.failed_gate, "no_live_access")

    def test_decide_and_impact(self):
        self.assertTrue(aiassurance.decide(
            "reviewer", "approved_fixture", "corroborated",
            "approved_read_only", "redacted_trace").allowed)
        self.assertFalse(aiassurance.decide(
            "reviewer", "approved_fixture", "assumed",
            "approved_read_only", "redacted_trace").allowed)
        self.assertTrue(aiassurance.impact(
            "reviewer", "approved_fixture", "bounded", "documented",
            "unknown_not_invented").allowed)
        self.assertFalse(aiassurance.impact(
            "reviewer", "approved_fixture", "bounded", "documented",
            "invented").allowed)


class InfrastructureTests(unittest.TestCase):
    def test_default_evidence_allowed(self):
        d = infrastructure.check_policy(infrastructure.PolicyEvidence())
        self.assertTrue(d.allowed)

    def test_every_scenario_blocked_at_its_gate(self):
        self.assertEqual(len(infrastructure.SCENARIOS), 28)
        for row in infrastructure.SCENARIOS:
            with self.subTest(scenario=row[0]):
                d = infrastructure.simulate(row[0])
                self.assertFalse(d.allowed)
                self.assertEqual(d.failed_gate, row[4])


class VectorSecTests(unittest.TestCase):
    def test_default_evidence_accepted(self):
        self.assertTrue(vectorsec.verify(vectorsec.RetrievalEvidence()).accepted)

    def test_every_scenario_blocked_at_its_gate(self):
        self.assertEqual(len(vectorsec.SCENARIOS), 22)
        for row in vectorsec.SCENARIOS:
            with self.subTest(scenario=row[0]):
                d = vectorsec.simulate(row[0])
                self.assertFalse(d.accepted)
                self.assertEqual(d.failed_gate, row[4])

    def test_internal_classification_needs_reviewer(self):
        from dataclasses import replace
        e = replace(vectorsec.RetrievalEvidence(),
                    classification="internal", role="operator")
        self.assertEqual(vectorsec.verify(e).failed_gate, "role")
        e = replace(vectorsec.RetrievalEvidence(), classification="internal")
        self.assertTrue(vectorsec.verify(e).accepted)

    def test_bounds(self):
        from dataclasses import replace
        base = vectorsec.RetrievalEvidence()
        self.assertEqual(vectorsec.verify(replace(base, top_k=6)).failed_gate, "top_k")
        self.assertEqual(vectorsec.verify(replace(base, top_k=0)).failed_gate, "top_k")
        self.assertEqual(
            vectorsec.verify(replace(base, daily_query_count=101)).failed_gate,
            "query_budget")


class RateLimitTests(unittest.TestCase):
    def test_bucket_drains_and_refills(self):
        clock = [100.0]
        bucket = ratelimit.TokenBucket(2, 1.0, now=lambda: clock[0])
        self.assertTrue(bucket.take())
        self.assertTrue(bucket.take())
        self.assertFalse(bucket.take())
        clock[0] += 1.0
        self.assertTrue(bucket.take())

    def test_bucket_validation(self):
        with self.assertRaises(ratelimit.RateLimitError):
            ratelimit.TokenBucket(0, 1.0)
        bucket = ratelimit.TokenBucket(1, 1.0)
        with self.assertRaises(ratelimit.RateLimitError):
            bucket.take(0)

    def test_retry_backoff_doubles_and_caps(self):
        policy = ratelimit.RetryPolicy(1.0, 8.0, 0.0)
        self.assertEqual([policy.delay(n) for n in (1, 2, 3, 4, 5)],
                         [1.0, 2.0, 4.0, 8.0, 8.0])
        with self.assertRaises(ratelimit.RateLimitError):
            policy.delay(0)

    def test_retry_jitter_bounded_and_deterministic(self):
        policy = ratelimit.RetryPolicy(2.0, 60.0, 0.5)
        d = policy.delay(3, rng=random.Random(42))
        self.assertTrue(4.0 <= d <= 12.0)
        self.assertEqual(d, policy.delay(3, rng=random.Random(42)))

    def test_priority_gate_protects_control(self):
        clock = [0.0]
        gate = ratelimit.PriorityGate(2, 100, now=lambda: clock[0])
        self.assertEqual(gate.decide("a", cls="batch").verdict, "allow")
        self.assertEqual(gate.decide("a", cls="batch").verdict, "allow")
        # Window full: batch sheds, interactive degrades, control passes.
        self.assertEqual(gate.decide("a", cls="batch").verdict, "shed")
        self.assertEqual(gate.decide("a", cls="interactive").verdict, "degrade")
        self.assertEqual(gate.decide("a", cls="control").verdict, "allow")
        self.assertEqual(gate.decide("a", tool="monitoring.health").verdict, "allow")

    def test_per_client_isolation(self):
        clock = [0.0]
        gate = ratelimit.PriorityGate(1000, 1, now=lambda: clock[0])
        self.assertEqual(gate.decide("greedy", cls="batch").verdict, "allow")
        self.assertEqual(gate.decide("greedy", cls="batch").reason,
                         "per_client_rate_exceeded")
        self.assertEqual(gate.decide("other", cls="batch").verdict, "allow")


if __name__ == "__main__":
    unittest.main()
