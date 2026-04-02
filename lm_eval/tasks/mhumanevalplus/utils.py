import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def _extract_code_from_response(response: str) -> str:
    """Extract the last code block from markdown fences in a model response."""
    for marker in ("```python", "```py", "```"):
        if marker in response:
            code = response.rsplit(marker, 1)[1]
            if "```" in code:
                code = code.split("```", 1)[0]
            return code.strip("\n")
    return response


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [[_extract_code_from_response(r) for r in resp] for resp, doc in zip(resps, docs)]
