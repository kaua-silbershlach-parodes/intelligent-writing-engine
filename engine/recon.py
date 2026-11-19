"""Uniform error surface: failures carry stable codes, never corpus
internals, vocabulary, file paths, or the offending input echoed back.
"""


class EngineError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code

    def __str__(self):
        return f"error:{self.code}"


def from_decision(decision):
    """Map a policy Decision to an EngineError (or None when allowed)."""
    if decision.allowed:
        return None
    return EngineError(decision.failed_gate)
