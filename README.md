# lm-evaluation-harness

## Setup

```bash
# Create and activate the venv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install the package with vllm backend (fast eval) and api extras (for online server mode)
uv pip install -e ".[vllm,hf,math,ifeval]"
```

> **Required env var for vLLM 0.20.x:** export `VLLM_USE_DEEP_GEMM=0` (and `VLLM_MOE_USE_DEEP_GEMM=0` for MoE models) before running `lm_eval` with the `vllm` backend. vLLM's DeepGEMM warmup runs unconditionally and crashes with `DeepGEMM backend is not available or outdated` on bf16/fp16 models when the optional `deep_gemm` package isn't fully importable. The flag skips the warmup; it has no effect on non-FP8 models.

## Evaluation Benchmarks

### French benchmarks

`mgsm_rev2_native_cot_fr`, `global_mmlu_fr_cot`, `gpqa_diamond_fr_cot`, `aime24_multilingual_fr`, `aime25_multilingual_fr`, `aime_combined_multilingual_fr`, `belebele_fr_cot`, `polymath_fr`, `mhumanevalplus_fr`

> `mhumanevalplus_fr` requires: `HF_ALLOW_CODE_EVAL=1`, `--confirm_run_unsafe_code` in `--model_args` for reasoning models.


```bash
# Using vLLM offline (single process)
nohup lm_eval \
    --model vllm \
    --model_args "pretrained=allenai/Olmo-3-7B-Think-SFT,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.7,max_model_len=32768" \
    --apply_chat_template \
    --tasks mgsm_rev2_native_cot_fr,global_mmlu_fr_cot,gpqa_diamond_fr_cot,aime24_multilingual_fr,belebele_fr_cot,polymath_fr \
    --batch_size auto \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0,max_gen_toks=30000 \
    --output_path eval_results/french_eval_result \
    --log_samples \
    --n_runs 3 \
    > logs/french_bench.log 2>&1 &
```

### English benchmarks

`mgsm_rev2_native_cot_en`, `global_mmlu_en_cot`, `gpqa_diamond_en_cot`, `aime24_multilingual_en`, `aime25_multilingual_en`, `aime_combined_multilingual_en`, `belebele_en_cot`, `polymath_en`, `mhumanevalplus_en`

> `mhumanevalplus_en` requires: `HF_ALLOW_CODE_EVAL=1`, `--confirm_run_unsafe_code` in `--model_args` for reasoning models.


```bash
# Using vLLM offline (single process)
nohup lm_eval \
    --model vllm \
    --model_args "pretrained=allenai/Olmo-3-7B-Think-SFT,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.7,max_model_len=32768" \
    --apply_chat_template \
    --tasks mgsm_rev2_native_cot_en,global_mmlu_en_cot,aime24_multilingual_en,gpqa_diamond_en_cot,belebele_en_cot,polymath_en \
    --batch_size auto \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0,max_gen_toks=30000 \
    --output_path eval_results/english_eval_result \
    --log_samples \
    --n_runs 3 \
    > logs/english_bench.log 2>&1 &
```

### Other languages

All the benchmarks above are available in **de, en, es, fr, sw, zh**. Swap the trailing language code on each task name (e.g. `aime24_multilingual_de`, `gpqa_diamond_es_cot`, `mgsm_rev2_native_cot_zh`, `global_mmlu_sw_cot`, `belebele_de_cot`, `polymath_zh`, `mhumanevalplus_fr`).

> `polymath_<lang>` is a group that aggregates the four difficulty tiers (`top`, `high`, `medium`, `low`); use `polymath_<lang>_<level>` to run a single tier. `belebele_<lang>_cot` is the chain-of-thought / generative variant of Belebele intended for reasoning models — the original log-likelihood `belebele_<flores_code>` task (e.g. `belebele_fra_Latn`) is still available for non-reasoning models.

> `aime_combined_multilingual_<lang>` is a per-language group that averages AIME 2024 + AIME 2025 (60 problems per language, size-weighted) into a single score. `aime_combined_multilingual` (no suffix) is the overall group across all six languages. `aime25_multilingual_<lang>` and the parent group `aime25_multilingual` are also available standalone.
