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


def process_results_pass64(doc, results):
    """Process all repeated results for pass@k computation."""
    gold = str(doc["answer_number"])
    # Filter groups all 64 responses into a single instance: results = [[ans_0, ..., ans_63]]
    if len(results) == 1 and isinstance(results[0], list):
        answers = results[0]
    else:
        answers = [r[0] if isinstance(r, list) else r for r in results]
    return pass_at_k(references=[gold], predictions=[answers], k=[1, 64])


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
