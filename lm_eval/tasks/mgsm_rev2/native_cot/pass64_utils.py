import re
from math import comb


def _compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _normalize(s: str) -> str:
    """Normalize a numeric string for comparison."""
    s = str(s).strip()
    s = re.sub(r"(\d)\.(\d{3})$", r"\1\2", s)
    s = re.sub(r"(\d),(\d{3})$", r"\1\2", s)
    s = re.sub(r"([+-]?\d+)[,.]0+$", r"\1", s)
    return s


def pass_at_k(references, predictions, k=None):
    """Compute pass@k for exact-match numeric tasks.

    references: [gold_answer_str]
    predictions: [[answer_0, answer_1, ..., answer_N]]
    k: list of k values, e.g. [1, 64]
    """
    if k is None:
        k = [1]
    gold = _normalize(str(references[0]))
    answers = predictions[0]
    n = len(answers)
    c = sum(1 for a in answers if _normalize(str(a)) == gold)
    return {f"pass@{ki}": _compute_pass_at_k(n, c, ki) for ki in k}
