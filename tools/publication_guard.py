"""Publication safeguard for the intelligence-writing-engine project.

This heuristic rejects private file types/paths and high-confidence
credential patterns in tracked files, using this project's own pattern
set — the same shapes the engine's publication guard (engine/guard.py)
screens for. It does not prove absence of all secrets. No tracked
payload, file name, or credential value is emitted in diagnostics.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

MAX_TEXT_BYTES = 2_000_000
PRIVATE_SEGMENTS = frozenset({
    'private', 'secret', 'secrets', 'credentials',
})
BLOCKED_EXTENSIONS = frozenset({
    '.pdf', '.zip', '.7z', '.rar', '.tar', '.tgz', '.gz',
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.pem', '.pfx', '.p12', '.key', '.kdbx', '.sqlite', '.db', '.dump',
})
SECRET_PATTERNS = (
    ('github_pat', re.compile(rb'github_pat_[A-Za-z0-9_]{20,}')),
    ('github_token', re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{20,}')),
    ('openai_key', re.compile(rb'\bsk-[A-Za-z0-9_-]{20,}')),
    ('aws_key', re.compile(rb'\bAKIA[0-9A-Z]{16}\b')),
    ('private_key', re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    ('bearer_value', re.compile(rb'Bearer\s+[A-Za-z0-9._-]{24,}')),
)


def check_entry(name: str, content: bytes, *, symlink: bool = False) -> set[str]:
    """Report categories only; never echo potential private paths or contents."""
    problems: set[str] = set()
    standardized = name.replace('\\', '/')
    path = PurePosixPath(standardized)
    segments = tuple(p.lower() for p in path.parts)
    if path.is_absolute() or not segments or '.' in segments or '..' in segments:
        problems.add('invalid_path')
    if set(segments) & PRIVATE_SEGMENTS:
        problems.add('private_path')
    basename = segments[-1] if segments else ''
    if basename == '.env' or (basename.startswith('.env.') and basename != '.env.example'):
        problems.add('environment_file')
    if path.suffix.lower() in BLOCKED_EXTENSIONS:
        problems.add('proprietary_or_sensitive_file_type')
    if symlink:
        problems.add('symlink_requires_manual_review')
    if len(content) > MAX_TEXT_BYTES:
        problems.add('oversized_unreviewed_file')
        return problems
    for category, expression in SECRET_PATTERNS:
        if expression.search(content):
            problems.add(category)
    return problems


def inspect_entries(entries: list[tuple[str, bytes, bool]]) -> dict:
    failures: list[dict] = []
    counts: Counter = Counter()
    for path, data, symlink in entries:
        problems = sorted(check_entry(path, data, symlink=symlink))
        if problems:
            counts.update(problems)
            failures.append({
                'path_sha256_prefix': hashlib.sha256(path.encode('utf-8')).hexdigest()[:16],
                'classes': problems,
            })
    return {
        'event': 'public_source_snapshot_gate',
        'decision': 'go' if not failures else 'no_go',
        'checked_files': len(entries),
        'violating_files': len(failures),
        'risk_classes': dict(sorted(counts.items())),
        'violations': failures,
        'secret_values_logged': False,
        'file_names_logged': False,
    }


def scan_tracked(root: Path) -> dict:
    root = root.resolve()
    result = subprocess.run([os.environ.get('GIT_EXECUTABLE', 'git'), '-C', str(root), 'ls-files', '--cached', '-z'],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            timeout=10)
    entries: list[tuple[str, bytes, bool]] = []
    for part in result.stdout.split(b'\x00'):
        if not part:
            continue
        name = part.decode('utf-8', errors='surrogateescape')
        path = root.joinpath(*PurePosixPath(name).parts)
        is_link = path.is_symlink()
        if is_link:
            entries.append((name, b'', True))
        elif path.is_file():
            with path.open('rb') as handle:
                entries.append((name, handle.read(MAX_TEXT_BYTES + 1), False))
        else:
            entries.append((name, b'', True))
    return inspect_entries(entries)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description='Read-only public source privacy gate')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    options = parser.parse_args()
    try:
        report = scan_tracked(options.root)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        report = {'event': 'public_source_snapshot_gate', 'decision': 'no_go',
                  'risk_classes': {'scan_failed': 1}, 'secret_values_logged': False,
                  'file_names_logged': False}
    print(json.dumps(report, sort_keys=True))
    return 0 if report['decision'] == 'go' else 2


if __name__ == '__main__':
    raise SystemExit(main())
