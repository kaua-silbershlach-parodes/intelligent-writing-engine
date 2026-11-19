"""Corpus loading: the README's technique items + the archived documents.

Units:
- One unit per numbered technique item in README.md (111 units).
- One unit per page of each archived document in techniques/ (94 units).
  Inventory files are indexes, not sources, and are not indexed.

Every unit carries an id, a kind, a title and a body. Tokenization is
shared by all retrieval modes so the modes differ only in ranking.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
    "were", "be", "been", "for", "on", "with", "as", "by", "at", "from",
    "it", "its", "this", "that", "these", "those", "you", "your", "we",
    "our", "not", "no", "if", "then", "than", "so", "such", "into",
    "about", "over", "under", "between",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ITEM_RE = re.compile(r"^(\d+)\. \*\*(.+?)\*\* — (.*)$")
_PAGE_RE = re.compile(r"^===== PAGE (\d+) =====$", re.M)

SECTION_PREFIX = {
    "Intelligence writing techniques": "core",
    "Analyst writing techniques": "analyst",
    "Decision writing techniques": "decision",
    "Analytic thinking techniques": "analytic",
}

DOCUMENTS = [
    "intelligence-writing-tips.md",
    "how-to-write-like-an-intelligence-analyst.md",
    "ic-writing-style.md",
    "analytic-thinking-and-presentation.md",
]


def tokenize(text):
    """Lowercase word tokens, stopwords removed. Shared by every mode."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Unit:
    __slots__ = ("uid", "kind", "title", "body", "tokens", "title_tokens")

    def __init__(self, uid, kind, title, body):
        self.uid = uid
        self.kind = kind
        self.title = title
        self.body = body
        # The title counts twice in the indexed text: titles carry the
        # technique's name and are the strongest relevance signal.
        self.tokens = tokenize(title) * 2 + tokenize(body)
        self.title_tokens = tokenize(title)

    def __repr__(self):
        return f"<Unit {self.uid} {self.title!r}>"


def load_readme_items():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    units = []
    section = None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = _ITEM_RE.match(line)
        if m and section in SECTION_PREFIX:
            num, title, body = int(m.group(1)), m.group(2), m.group(3)
            uid = f"{SECTION_PREFIX[section]}-{num:02d}"
            units.append(Unit(uid, section, title, body))
    return units


# Pages below this token count are covers, dividers or blank conversion
# artifacts: they carry a part title at most, no technique content.
MIN_PAGE_TOKENS = 15


def load_document_pages():
    units = []
    for name in DOCUMENTS:
        text = (ROOT / "techniques" / name).read_text(encoding="utf-8")
        slug = name[:-3].replace("-", "_")
        parts = _PAGE_RE.split(text)
        # parts: [pre, page_no, page_text, page_no, page_text, ...]
        if len(parts) == 1:
            units.append(Unit(f"doc-{slug}", name, name, text))
            continue
        for i in range(1, len(parts), 2):
            page_no, page_text = int(parts[i]), parts[i + 1]
            if len(tokenize(page_text)) < MIN_PAGE_TOKENS:
                continue
            title = f"{name} — page {page_no}"
            units.append(Unit(f"doc-{slug}-p{page_no:02d}", name, title, page_text))
    return units


def load_corpus():
    return load_readme_items() + load_document_pages()
