ANSWER_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


def doc_to_target(doc):
    ans = doc["correct_answer_num"]
    if isinstance(ans, int):
        return "(%s)" % ANSWER_MAP[str(ans)]
    return "(%s)" % ANSWER_MAP[ans]


def _add_choices(doc):
    doc["choices"] = ["(A)", "(B)", "(C)", "(D)"]
    return doc


def process_docs(dataset):
    return dataset.map(_add_choices)
