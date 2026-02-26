# lm-evaluation-harness

## Setup

```bash
# Create and activate the venv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install the package with vllm backend (fast eval)
uv pip install -e ".[vllm,hf,math,ifeval]"
```

## Evaluation Benchmarks

Replace the following placeholders:
- `MODEL` — path or HuggingFace model ID
- `TP` — tensor parallelism size
- `OUTPUT_PATH` — directory for results

### French benchmarks

`mgsm_native_cot_fr`, `mmmlu_fr_fr_cot_zeroshot`, `gpqa_diamond_fr_cot_zeroshot`, `aime24_fr`

```bash
lm_eval \
    --model vllm \
    --model_args "pretrained=MODEL,dtype=bfloat16,tensor_parallel_size=TP,gpu_memory_utilization=0.7,max_model_len=36000,enforce_eager=False,think_end_token=</think>" \
    --apply_chat_template \
    --tasks mgsm_native_cot_fr,mmmlu_fr_fr_cot_zeroshot,gpqa_diamond_fr_cot_zeroshot,aime24_fr \
    --batch_size auto \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0 \
    --output_path OUTPUT_PATH \
    --log_samples
```

### English benchmarks

`mgsm_native_cot_en`, `mmlu_flan_cot_zeroshot`, `gpqa_diamond_cot_zeroshot`, `aime24`

```bash
lm_eval \
    --model vllm \
    --model_args "pretrained=MODEL,dtype=bfloat16,tensor_parallel_size=TP,gpu_memory_utilization=0.7,max_model_len=36000,enforce_eager=False,think_end_token=</think>" \
    --apply_chat_template \
    --tasks mgsm_native_cot_en,mmlu_flan_cot_zeroshot,aime24,gpqa_diamond_cot_zeroshot \
    --batch_size auto \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0 \
    --output_path OUTPUT_PATH \
    --log_samples
```
