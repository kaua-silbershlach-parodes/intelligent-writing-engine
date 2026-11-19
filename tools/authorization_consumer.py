#!/usr/bin/env python3
"""Strict consumer for this project's gate and assurance reports.

This is NOT a production authorization service. It never interprets
process success as acceptance: it accepts only an explicitly validated
report — a publication gate `go`, or an assurance run whose every
check passed — parsed strictly (no duplicate keys, no non-finite
numbers, bounded size), with the logging contract intact (no file
names, no secret values in the report). The artifact that produced
the report is pinned by SHA-256 before it runs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import re
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounded_capture import capture

MAX_REPLY = 65536
MAX_STDERR = 8192
MAX_ARGS = 12
CHANNELS = ("publication", "assurance")


class DuplicateKey(ValueError):
    pass


def no_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in items:
        if k in result:
            raise DuplicateKey("duplicate_json_key")
        result[k] = v
    return result


def invalid_constant(value: str) -> None:
    raise ValueError("nonfinite_json")


@dataclass(frozen=True)
class Classification:
    policy_version: str
    intent: str
    disposition: str
    preview_accepted: bool
    production_authorized: bool
    rationale: str


def classification(intent: str, disposition: str, rationale: str) -> Classification:
    return Classification(
        policy_version="engine-report-consumer-v1", intent=intent,
        disposition=disposition,
        preview_accepted=disposition in ("ACCEPT_GATE_REPORT", "ACCEPT_ASSURANCE_REPORT"),
        production_authorized=False, rationale=rationale,
    )


def parse_response(raw: bytes, exit_code: int, *, channel: str,
                   intent: str, expected_scenario: str = "") -> Classification:
    """Interpret only the declared intent; never infer it from fields.

    `operation` means a gate/assurance run whose report is being
    accepted or rejected. `scenario` means a negative case: a report
    that records a denial for the expected scenario.
    """
    if channel not in CHANNELS or intent not in ("operation", "scenario"):
        return classification(intent, "REJECT_UNSUPPORTED_ROUTE", "invalid_channel_or_intent")
    if not isinstance(raw, bytes) or len(raw) > MAX_REPLY:
        return classification(intent, "REJECT_INVALID_ENVELOPE", "unbounded_output")
    if not isinstance(exit_code, int) or exit_code != 0:
        return classification(intent, "REJECT_PROCESS_FAILURE", "command_not_completed")
    try:
        data = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=no_duplicates,
                          parse_constant=invalid_constant)
    except (UnicodeError, ValueError, TypeError):
        return classification(intent, "REJECT_INVALID_ENVELOPE", "malformed_or_duplicated_json")
    if not isinstance(data, dict):
        return classification(intent, "REJECT_INVALID_ENVELOPE", "non_object_reply")
    # The reports of this project never log file names or secret
    # values; a report that does is not the contract.
    for field in ("file_names_logged", "secret_values_logged"):
        if field in data and data[field] is not False:
            return classification(intent, "REJECT_EFFECTS", "logging_contract_contradiction")

    if intent == "scenario":
        if not expected_scenario or data.get("scenario") != expected_scenario:
            return classification(intent, "REJECT_SCENARIO_ID", "scenario_identity_unconfirmed")
        if data.get("ok") is True and data.get("decision") in ("deny", "no_go", "hold"):
            return classification(intent, "SCENARIO_RECORDED", "negative_case_completed_not_access")
        return classification(intent, "REJECT_SCENARIO_RESULT", "unexpected_scenario_output")

    if channel == "publication":
        decision = data.get("decision")
        if decision == "go":
            return classification(intent, "ACCEPT_GATE_REPORT", "gate_go_verified")
        if decision == "no_go":
            return classification(intent, "REJECT_DENIED", "explicit_gate_no_go")
        return classification(intent, "REJECT_UNKNOWN_DECISION", "no_explicit_gate_decision")

    # channel == "assurance"
    ok = data.get("ok")
    if not isinstance(ok, bool):
        return classification(intent, "REJECT_INVALID_ENVELOPE", "non_boolean_ok")
    checks = data.get("checks")
    if not isinstance(checks, dict) or not checks:
        return classification(intent, "REJECT_INVALID_ENVELOPE", "checks_missing")
    if not ok:
        return classification(intent, "REJECT_DENIED", "assurance_not_ok")
    if not all(v is True for v in checks.values()):
        return classification(intent, "REJECT_INCONSISTENT", "ok_true_with_failed_checks")
    return classification(intent, "ACCEPT_ASSURANCE_REPORT", "every_check_measured_true")


def _route_matches(argv: list[str], channel: str, intent: str) -> bool:
    if channel == "publication":
        return argv in (["--staged"], ["--pre-push"])
    if channel == "assurance":
        return argv == []
    return False


def invoke_report(artifact: Path, argv: list[str], *, channel: str, intent: str,
                  expected_scenario: str = "", sha256: str = "") -> Classification:
    """Run only the pinned artifact whose route matches the intent."""
    if not artifact.is_file() or len(argv) > MAX_ARGS:
        return classification(intent, "REJECT_INVALID_INVOCATION", "missing_artifact_or_args")
    if any(not isinstance(a, str) or len(a) > 256 or "\0" in a for a in argv):
        return classification(intent, "REJECT_INVALID_INVOCATION", "unbounded_input")
    expected_name = {"publication": "git_publication_gate.py",
                     "assurance": "assurance.py"}.get(channel, "")
    if artifact.name != expected_name or not _route_matches(argv, channel, intent):
        return classification(intent, "REJECT_ROUTE_MISMATCH", "channel_intent_argv_mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        return classification(intent, "REJECT_UNPINNED_ARTIFACT", "artifact_digest_required")
    try:
        if artifact.is_symlink() or artifact.stat().st_size > 50_000_000:
            return classification(intent, "REJECT_UNPINNED_ARTIFACT", "untrusted_artifact_file")
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            return classification(intent, "REJECT_ARTIFACT_MISMATCH", "artifact_digest_does_not_match")
    except OSError:
        return classification(intent, "REJECT_UNPINNED_ARTIFACT", "unavailable_artifact")
    if channel == "publication":
        command = [sys.executable, str(artifact), *argv]
    else:
        # The module runs from the repository root (the consumer's
        # documented working directory); anywhere else it fails closed.
        command = [sys.executable, "-m", "engine.assurance"]
    try:
        result = capture(command, timeout_s=30, stdout_limit=MAX_REPLY,
                         stderr_limit=MAX_STDERR)
    except (OSError, ValueError):
        return classification(intent, "REJECT_PROCESS_FAILURE", "execution_failed")
    if result.failure:
        return classification(intent, "REJECT_CAPTURE_FAILURE", result.failure)
    return parse_response(result.stdout, result.returncode, channel=channel,
                          intent=intent, expected_scenario=expected_scenario)


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict report consumer: no production authority")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="exact SHA-256 of the artifact")
    parser.add_argument("--channel", choices=list(CHANNELS), required=True)
    parser.add_argument("--intent", choices=["operation", "scenario"], required=True)
    parser.add_argument("--expected-scenario", default="")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    result = invoke_report(args.artifact, args.args, channel=args.channel,
                           intent=args.intent,
                           expected_scenario=args.expected_scenario,
                           sha256=args.sha256)
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if result.disposition in ("ACCEPT_GATE_REPORT", "ACCEPT_ASSURANCE_REPORT",
                                       "SCENARIO_RECORDED") else 2


if __name__ == "__main__":
    sys.exit(main())
