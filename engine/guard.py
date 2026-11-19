"""Publication guard: scan files for credential-shaped strings before
anything is published. A hit fails the gate and names file and pattern
kind — never the value itself.
"""

import re
from pathlib import Path

from . import policy

PATTERNS = {
    "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "bearer_value": re.compile(r"Bearer\s+[A-Za-z0-9._-]{24,}"),
}


def scan_text(text):
    return [name for name, pat in PATTERNS.items() if pat.search(text or "")]


def scan_file(path):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ["unreadable"]
    return scan_text(text)


TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".cfg", ".ini",
                 ".yml", ".yaml", ".rst"}


def scan_tree(root):
    """{relative_path: [pattern kinds]} for text files with hits.

    Only text files are scanned: bytecode and other binaries can contain
    byte runs that merely look like a credential shape.
    """
    root = Path(root)
    hits = {}
    for p in sorted(root.rglob("*")):
        if (p.is_file() and ".git" not in p.parts
                and "__pycache__" not in p.parts
                and p.suffix.lower() in TEXT_SUFFIXES):
            found = scan_file(p)
            if found:
                hits[str(p.relative_to(root))] = found
    return hits


def publication_decision(root):
    hits = scan_tree(root)
    if hits:
        first = sorted(hits)[0]
        return policy.Decision(False, f"credential_shape:{first}:{hits[first][0]}")
    return policy.Decision(True, "")
