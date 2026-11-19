"""Project checkpoint — an auditable snapshot of the code.

Writes checkpoints/cp-<seq>-<timestamp>.md with:
  - HEAD commit, date, git state (clean/dirty)
  - file and line inventory per area of this repository
  - test totals (test files and test methods)

Optionally creates the git tag checkpoint-<seq>-<timestamp> (--tag).

Usage:
    python tools/checkpoint.py --message "engine baseline"
    python tools/checkpoint.py --message "modules complete" --tag

Standard library only. Never touches commit history; tags are additive.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"

AREAS = {
    "engine": ROOT / "engine",
    "techniques": ROOT / "techniques",
    "tests": ROOT / "tests",
    "tools": ROOT / "tools",
    "hooks": ROOT / ".githooks",
}

CODE_EXTS = {".py", ".md"}
SKIP_DIRS = {".git", "__pycache__"}


def sh(args: list[str]) -> str:
    try:
        out = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def inventory() -> tuple[dict[str, dict[str, int]], int, int]:
    per_area: dict[str, dict[str, int]] = {}
    total_files = 0
    total_lines = 0
    for name, base in AREAS.items():
        files = 0
        lines = 0
        if base.exists():
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                if any(part in SKIP_DIRS for part in p.parts):
                    continue
                files += 1
                if p.suffix.lower() in CODE_EXTS:
                    try:
                        with p.open("rb") as fh:
                            lines += sum(1 for _ in fh)
                    except OSError:
                        pass
        per_area[name] = {"files": files, "lines": lines}
        total_files += files
        total_lines += lines
    return per_area, total_files, total_lines


def count_tests() -> dict[str, int]:
    test_dir = ROOT / "tests"
    files = list(test_dir.glob("test_*.py")) if test_dir.exists() else []
    methods = 0
    for path in files:
        try:
            methods += len(re.findall(r"def test_", path.read_text(encoding="utf-8")))
        except OSError:
            pass
    return {"files": len(files), "methods": methods}


def next_seq() -> int:
    seqs = []
    if CHECKPOINTS.exists():
        for p in CHECKPOINTS.glob("cp-*.md"):
            m = re.match(r"cp-(\d+)-", p.name)
            if m:
                seqs.append(int(m.group(1)))
    return max(seqs, default=0) + 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a project checkpoint.")
    ap.add_argument("--message", required=True, help="Short checkpoint description")
    ap.add_argument("--tag", action="store_true", help="Create a git tag for the checkpoint")
    args = ap.parse_args()

    seq = next_seq()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    head = sh(["git", "rev-parse", "--short", "HEAD"]) or "(no git)"
    dirty = sh(["git", "status", "--porcelain"]) != ""
    per_area, total_files, total_lines = inventory()
    tests = count_tests()

    lines = []
    lines.append(f"# Checkpoint cp-{seq:03d} — {stamp}")
    lines.append("")
    lines.append(f"Message: {args.message}")
    lines.append(f"HEAD: {head}")
    lines.append(f"Git tree: {'DIRTY (uncommitted changes)' if dirty else 'clean'}")
    lines.append("")
    lines.append("## Code inventory")
    lines.append("")
    lines.append("| Area | Files | Lines |")
    lines.append("| --- | ---: | ---: |")
    for name, info in per_area.items():
        lines.append(f"| {name} | {info['files']} | {info['lines']} |")
    lines.append(f"| **TOTAL** | **{total_files}** | **{total_lines}** |")
    lines.append("")
    lines.append("## Tests")
    lines.append("")
    lines.append(f"- Test files (test_*.py): {tests['files']}")
    lines.append(f"- Test methods (def test_*): {tests['methods']}")
    lines.append("")

    CHECKPOINTS.mkdir(exist_ok=True)
    out = CHECKPOINTS / f"cp-{seq:03d}-{stamp}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"checkpoint written: {out.relative_to(ROOT)}")
    print(f"files={total_files} lines={total_lines} head={head}")

    if args.tag:
        tag = f"checkpoint-{seq:03d}-{stamp}"
        rc = subprocess.run(
            ["git", "tag", "-a", tag, "-m", f"checkpoint {seq:03d}: {args.message}"],
            cwd=ROOT,
        ).returncode
        print(f"tag {tag}: {'created' if rc == 0 else 'FAILED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
