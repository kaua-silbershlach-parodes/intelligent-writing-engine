"""Supply-chain and change control: a SHA-256 manifest over the corpus
files and the engine sources, plus a standard-library-only audit.

The manifest is the checkpoint of the corpus state: any change to a
tracked file after the manifest was taken fails verification and names
the file. The import audit parses every engine module and rejects any
import outside the standard library.
"""

import ast
import hashlib
import sys
from pathlib import Path

from . import policy

ROOT = Path(__file__).resolve().parent.parent
TRACKED_DIRS = ("techniques",)
TRACKED_FILES = ("README.md",)
ENGINE_DIR = Path(__file__).resolve().parent


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest():
    """{relative_path: sha256} for every tracked file. Deterministic."""
    entries = {}
    for name in TRACKED_FILES:
        p = ROOT / name
        if p.exists():
            entries[name] = _sha256(p)
    for dirname in TRACKED_DIRS:
        d = ROOT / dirname
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    entries[str(p.relative_to(ROOT))] = _sha256(p)
    for p in sorted(ENGINE_DIR.glob("*.py")):
        entries[str(p.relative_to(ROOT))] = _sha256(p)
    return entries


def verify_manifest(manifest):
    """Decision: allowed, or failed_gate naming the first problem."""
    current = build_manifest()
    for rel, digest in manifest.items():
        if rel not in current:
            return policy.Decision(False, f"missing:{rel}")
        if current[rel] != digest:
            return policy.Decision(False, f"changed:{rel}")
    for rel in current:
        if rel not in manifest:
            return policy.Decision(False, f"untracked:{rel}")
    return policy.Decision(True, "")


def stdlib_audit():
    """Decision: every engine import must be standard library."""
    stdlib = sys.stdlib_module_names
    for p in sorted(ENGINE_DIR.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import: this package
                    continue
                names = [(node.module or "").split(".")[0]]
            for name in names:
                if name and name not in stdlib and name != "engine":
                    return policy.Decision(False, f"non_stdlib:{p.name}:{name}")
    return policy.Decision(True, "")
