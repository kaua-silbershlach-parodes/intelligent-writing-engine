"""Contract matrix audit: every public surface must agree.

Surfaces checked: the policy mode list, the agent card's skills, and
the CLI's mode choices. A surface that drifts from the matrix fails
the audit and names the surface.
"""

from . import a2a, policy


def contract_matrix():
    """{surface: [modes]} for every surface that exposes modes."""
    card_modes = [s["id"] for s in a2a.agent_card()["skills"]]
    return {
        "policy": list(policy.MODES),
        "agent_card": card_modes,
    }


def audit_contracts(cli_modes=None):
    """Decision: all surfaces expose exactly the policy modes."""
    matrix = contract_matrix()
    if cli_modes is not None:
        matrix["cli"] = list(cli_modes)
    expected = list(policy.MODES)
    for surface, modes in matrix.items():
        if modes != expected:
            return policy.Decision(False, f"contract_drift:{surface}")
    return policy.Decision(True, "")
