"""Tests for the publication gates in tools/ and the .githooks shims.

The end-to-end tests build a throwaway git repository: a clean staged
file passes the pre-commit gate; a staged credential shape fails it,
and the pre-push gate catches the same shape in outgoing history.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import publication_guard as guard  # noqa: E402

FAKE_TOKEN = "github_pat_" + "A" * 30


def git(cwd, *args, stdin=""):
    return subprocess.run(["git", "-C", str(cwd), *args], input=stdin,
                          capture_output=True, text=True, timeout=30)


class CheckEntryTests(unittest.TestCase):
    def test_clean_entry(self):
        self.assertEqual(guard.check_entry("engine/bm25.py", b"print('ok')\n"), set())

    def test_private_path(self):
        self.assertIn("private_path", guard.check_entry("private/notes.md", b"x"))

    def test_environment_file(self):
        self.assertIn("environment_file", guard.check_entry(".env", b"x"))
        self.assertEqual(guard.check_entry(".env.example", b"x"), set())

    def test_blocked_extension(self):
        self.assertIn("proprietary_or_sensitive_file_type",
                      guard.check_entry("dump.pdf", b"x"))

    def test_credential_shape(self):
        problems = guard.check_entry("notes.md", f"token {FAKE_TOKEN}".encode())
        self.assertIn("github_pat", problems)

    def test_openai_key_shape(self):
        problems = guard.check_entry("notes.md", ("key " + "sk-" + "B" * 24).encode())
        self.assertIn("openai_key", problems)

    def test_repo_tracked_files_pass(self):
        report = guard.scan_tracked(ROOT)
        self.assertEqual(report["decision"], "go")


class GateEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gate-test-"))
        git(self.tmp, "init", "-q")
        git(self.tmp, "config", "user.email", "test@example.invalid")
        git(self.tmp, "config", "user.name", "Gate Test")
        tools = self.tmp / "tools"
        tools.mkdir()
        for name in ("publication_guard.py", "git_publication_gate.py"):
            (tools / name).write_bytes((ROOT / "tools" / name).read_bytes())

    def run_gate(self, *args, stdin=""):
        return subprocess.run(
            [sys.executable, str(self.tmp / "tools" / "git_publication_gate.py"),
             "--root", str(self.tmp), *args],
            input=stdin, capture_output=True, text=True, timeout=60)

    def test_staged_clean_passes_and_dirty_fails(self):
        (self.tmp / "ok.md").write_text("clean\n")
        git(self.tmp, "add", "ok.md")
        clean = self.run_gate("--staged")
        self.assertEqual(clean.returncode, 0, clean.stdout)
        self.assertEqual(json.loads(clean.stdout)["decision"], "go")

        (self.tmp / "leak.md").write_text(f"token {FAKE_TOKEN}\n")
        git(self.tmp, "add", "leak.md")
        dirty = self.run_gate("--staged")
        self.assertEqual(dirty.returncode, 2)
        report = json.loads(dirty.stdout)
        self.assertEqual(report["decision"], "no_go")
        self.assertFalse(report["file_names_logged"])
        self.assertNotIn("leak.md", dirty.stdout)

    def test_pre_push_scans_outgoing_history(self):
        (self.tmp / "leak.md").write_text(f"token {FAKE_TOKEN}\n")
        git(self.tmp, "add", "leak.md")
        git(self.tmp, "commit", "-q", "-m", "x")
        sha = git(self.tmp, "rev-parse", "HEAD").stdout.strip()
        zero = "0" * 40
        refs = f"refs/heads/main {sha} refs/heads/main {zero}\n"
        result = self.run_gate("--pre-push", stdin=refs)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["decision"], "no_go")


if __name__ == "__main__":
    unittest.main()
