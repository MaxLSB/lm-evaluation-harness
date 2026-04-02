# lm-evaluation-harness

## Setup

```bash
# Create and activate the venv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install the package with vllm backend (fast eval) and api extras (for online server mode)
uv pip install -e ".[vllm,hf,math,ifeval]"
```

## Evaluation Benchmarks

### French benchmarks

`mgsm_rev2_native_cot_fr`, `global_mmlu_fr_cot`, `gpqa_diamond_fr_cot_zeroshot`, `aime24_fr`, `mhumanevalplus_fr`

> `mhumanevalplus_fr` requires: `HF_ALLOW_CODE_EVAL=1`, `--confirm_run_unsafe_code` in `--model_args` for reasoning models.


```bash
# Using vLLM offline (single process)
nohup lm_eval \
    --model vllm \
    --model_args "pretrained=allenai/Olmo-3-7B-Think-SFT,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.7,max_model_len=32768" \
    --apply_chat_template \
    --tasks mgsm_rev2_native_cot_fr,global_mmlu_fr_cot,gpqa_diamond_fr_cot_zeroshot,aime24_fr \
    --batch_size auto \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0,max_gen_toks=30000 \
    --output_path eval_results/french_eval_result \
    --log_samples \
    --n_runs 3 \
    > logs/french_bench.log 2>&1 &
```

### English benchmarks

`mgsm_rev2_native_cot_en`, `global_mmlu_en_cot`, `gpqa_diamond_cot_zeroshot`, `aime24`, `mhumanevalplus_en`

> `mhumanevalplus_en` requires: `HF_ALLOW_CODE_EVAL=1`, `--confirm_run_unsafe_code` in `--model_args` for reasoning models.


```bash
# Using vLLM offline (single process)
nohup lm_eval \
    --model vllm \
    --model_args "pretrained=allenai/Olmo-3-7B-Think-SFT,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.7,max_model_len=32768" \
    --apply_chat_template \
    --tasks mgsm_rev2_native_cot_en,global_mmlu_en_cot,aime24,gpqa_diamond_cot_zeroshot \
    --batch_size auto \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0,max_gen_toks=30000 \
    --output_path eval_results/english_eval_result \
    --log_samples \
    --n_runs 3 \
    > logs/english_bench.log 2>&1 &
```
