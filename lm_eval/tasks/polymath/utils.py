try:
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "polymath requires `math_verify`. Install via `pip install lm-eval[math]`."
    ) from e


_EXTRACTION_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]


def process_results(doc, results):
    response = results[0]
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    target = str(doc[answer_key])

    parsed_pred = parse(response, extraction_config=_EXTRACTION_CONFIG)
    parsed_gold = parse(target, extraction_config=_EXTRACTION_CONFIG)

    return {"exact_match": int(verify(parsed_gold, parsed_pred))}
