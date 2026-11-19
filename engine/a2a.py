"""Agent-to-agent surface: an agent card, delegated tasks with a
lifecycle, and artifacts with provenance.

Lifecycle: submitted -> working -> completed | failed ; a submitted or
working task may be canceled. Terminal states are final. Delegation is
gated: the caller role must be known and the requested kind must be in
scope — a caller's claim is evidence, never authorization by itself.
"""

from . import policy, raguard

STATES = ("submitted", "working", "completed", "failed", "canceled")
_TRANSITIONS = {
    "submitted": ("working", "canceled"),
    "working": ("completed", "failed", "canceled"),
    "completed": (),
    "failed": (),
    "canceled": (),
}


def agent_card():
    """The engine's agent card: identity, skills, input/output modes."""
    return {
        "name": "intelligence-writing-engine",
        "description": "Retrieval over the intelligence-writing technique corpus.",
        "skills": [
            {"id": mode, "name": f"{mode} search",
             "description": f"Rank corpus units with the {mode} ranker."}
            for mode in policy.MODES
        ],
        "input_modes": ["text/plain"],
        "output_modes": ["application/json"],
        "authentication": "local roles assigned by the embedding application",
    }


class Task:
    _counter = 0

    def __init__(self, caller, query, mode="hybrid", top=5, kind=None):
        Task._counter += 1
        self.id = f"task-{Task._counter:04d}"
        self.caller = caller
        self.query = query
        self.mode = mode
        self.top = top
        self.kind = kind
        self.state = "submitted"
        self.artifacts = []
        self.failed_gate = ""

    def transition(self, target):
        if target not in _TRANSITIONS[self.state]:
            raise ValueError(f"illegal transition {self.state} -> {target}")
        self.state = target

    def delegate_decision(self):
        d = policy.decide_mode(self.caller, self.mode)
        if not d.allowed:
            return d
        return raguard.retrieval_decision(self.caller, self.kind)


def submit(caller, query, mode="hybrid", top=5, kind=None, runner=None):
    """Create a task, gate it, run it, and return it with artifacts.

    runner(query, mode, top, kind) -> [(unit, score), ...]; defaults to
    the engine's own search. A gated-out task ends failed with the gate
    named; it never reaches the runner.
    """
    if runner is None:
        from .search import run
        runner = lambda q, m, t, k: run(q, mode=m, top=t, kind=k)
    task = Task(caller, query, mode=mode, top=top, kind=kind)
    decision = task.delegate_decision()
    if not decision.allowed:
        task.transition("working")
        task.failed_gate = decision.failed_gate
        task.transition("failed")
        return task
    task.transition("working")
    try:
        results = runner(query, mode, top, kind)
    except Exception:
        task.failed_gate = "runner_error"
        task.transition("failed")
        return task
    task.artifacts = [
        {"unit": unit.uid, "title": unit.title, "kind": unit.kind,
         "score": round(score, 6)}
        for unit, score in results
    ]
    task.transition("completed")
    return task
