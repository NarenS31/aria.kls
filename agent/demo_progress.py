"""Match authored work states, not merely equations with the same solution set."""
from agent.math_check import equivalent, extract_candidates, normalize, to_expr


def reached(result, texts, achieved):
    if not result:
        return False
    if "=" not in result:
        expr = to_expr(normalize(result))
        return expr is not None and any(equivalent(expr, a) for a in achieved)
    left, right = (to_expr(normalize(s)) for s in result.split("=", 1))
    if left is None or right is None:
        return False
    for text in texts:
        for c in extract_candidates(text):
            if c.kind == "eq" and (
                (equivalent(left, c.lhs) and equivalent(right, c.rhs)) or
                (equivalent(left, c.rhs) and equivalent(right, c.lhs))
            ):
                return True
    return False


def completed_moves(session):
    moves = session.problem.moves
    if any(row.get("correctness") == "correct" for row in session.log):
        return [True] * len(moves)
    done = [reached(m.get("result") or m.get("claim"), session.verified_work, session.achieved) for m in moves]
    # An authored later equation state subsumes earlier states along that same pathway.
    last = max((i for i, m in enumerate(moves) if done[i] and "=" in m.get("result", "")), default=-1)
    return [value or i <= last for i, value in enumerate(done)]
