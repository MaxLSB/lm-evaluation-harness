# Belebele (CoT, generative)

Chain-of-thought / generative variant of the Belebele multilingual reading-comprehension benchmark, intended for reasoning models (DeepSeek-R1-class, o-series, etc.) that benefit from explicit step-by-step thinking before producing a final answer.

The original `belebele` task uses log-likelihood scoring over A/B/C/D, which prevents reasoning models from using their chain-of-thought. This variant uses `output_type: generate_until` and asks the model to think step by step and place its final answer letter inside `\boxed{...}`, mirroring the prompt and extraction logic of `global_mmlu_<lang>_cot`.

### Languages

Six language variants are provided (one per language, no subject categories — Belebele is a single set of questions):

| Task name           | FLORES code | Language     |
|---------------------|-------------|--------------|
| `belebele_en_cot`   | `eng_Latn`  | English      |
| `belebele_fr_cot`   | `fra_Latn`  | French       |
| `belebele_de_cot`   | `deu_Latn`  | German       |
| `belebele_es_cot`   | `spa_Latn`  | Spanish      |
| `belebele_sw_cot`   | `swh_Latn`  | Swahili      |
| `belebele_zh_cot`   | `zho_Hans`  | Chinese (Simplified) |

A group `belebele_cot` aggregates all six (size-weighted mean of `exact_match`).

### Prompt format

Each prompt is fully translated into the target language and asks the model to:
1. Read a passage (from FLORES-200, via `flores_passage`).
2. Answer a 4-way multiple-choice question.
3. Think step by step.
4. Place the final answer letter inside `\boxed{...}` on the last line.

### Scoring

- `output_type: generate_until`, greedy (`temperature=0`, `do_sample=false`).
- Two filters:
  - `strict-match`: localized "The answer is ..." regex (e.g. `La réponse est`, `Die Antwort ist`, `答案是`, `Jibu ni`).
  - `flexible-extract`: pulls the letter from `\boxed{[A-D]}` (last occurrence).
- Metric: `exact_match` (case- and punctuation-insensitive).

### Citation

```bibtex
@misc{bandarkar2023belebele,
      title={The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
      author={Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      year={2023},
      eprint={2308.16884},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
