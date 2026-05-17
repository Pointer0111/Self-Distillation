# A800 Training Startup Guide

## Recommended Setup

- GPUs: `2 x A800-80G`
- Model path: `/root/autodl-fs/Qwen2.5-7B-Instruct`
- Project path: `/root/Self-Distillation`
- DeepSpeed config: `/root/Self-Distillation/ds_config_zero2_a800.json`
- Recommended first run: `LoRA + DeepSpeed + no vLLM`

This repository now supports:

- optional LoRA
- optional DeepSpeed multi-GPU launch
- optional vLLM configuration
- startup config summary printed by `rank 0`

## Before Launch

Run these checks on the A800 server:

```bash
cd /root/Self-Distillation
nvidia-smi
python -V
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
ls /root/autodl-fs/Qwen2.5-7B-Instruct
```

You should confirm:

- there are 2 visible GPUs
- the model directory exists
- the project directory contains `main.py`
- the project directory contains `ds_config_zero2_a800.json`

## Recommended First Launch

This is the safest first command to verify the full training path:

```bash
cd /root/Self-Distillation

torchrun --nproc_per_node=2 main.py \
  --dataset_name tooluse \
  --model_name /root/autodl-fs/Qwen2.5-7B-Instruct \
  --output_dir /root/autodl-fs/outputs/tooluse_lora_a800 \
  --deepspeed /root/Self-Distillation/ds_config_zero2_a800.json \
  --use_lora true \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --torch_dtype bfloat16 \
  --use_vllm false \
  --report_to none \
  --learning_rate 5e-5 \
  --num_train_epochs 2
```

## Science Dataset Launch

To train on the science dataset, only change `--dataset_name` and the output directory:

```bash
cd /root/Self-Distillation

torchrun --nproc_per_node=2 main.py \
  --dataset_name science \
  --model_name /root/autodl-fs/Qwen2.5-7B-Instruct \
  --output_dir /root/autodl-fs/outputs/science_lora_a800 \
  --deepspeed /root/Self-Distillation/ds_config_zero2_a800.json \
  --use_lora true \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --torch_dtype bfloat16 \
  --use_vllm false \
  --report_to none \
  --learning_rate 5e-5 \
  --num_train_epochs 2
```

## Optional vLLM Launch

Only try this after the non-vLLM launch is stable.

```bash
cd /root/Self-Distillation

torchrun --nproc_per_node=2 main.py \
  --dataset_name tooluse \
  --model_name /root/autodl-fs/Qwen2.5-7B-Instruct \
  --output_dir /root/autodl-fs/outputs/tooluse_lora_a800_vllm \
  --deepspeed /root/Self-Distillation/ds_config_zero2_a800.json \
  --use_lora true \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --torch_dtype bfloat16 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size 2 \
  --vllm_gpu_memory_utilization 0.2 \
  --report_to none \
  --learning_rate 5e-5 \
  --num_train_epochs 2
```

## What To Check In Logs

At startup, `rank 0` prints a config summary. Verify these fields:

- `world_size: 2`
- `use_lora: True`
- `deepspeed: /root/Self-Distillation/ds_config_zero2_a800.json`
- `model_name: /root/autodl-fs/Qwen2.5-7B-Instruct`
- `use_vllm: False` for the first run
- `effective_prompt_batch` matches your expectation

If these are wrong, your launch command did not take effect as expected.

## If You Hit OOM

Reduce load in this order:

1. lower `--gradient_accumulation_steps`
2. reduce `--max_prompt_length`
3. reduce `--max_completion_length`
4. keep `--use_vllm false`
5. keep LoRA enabled

Example lower-memory command:

```bash
torchrun --nproc_per_node=2 main.py \
  --dataset_name tooluse \
  --model_name /root/autodl-fs/Qwen2.5-7B-Instruct \
  --output_dir /root/autodl-fs/outputs/tooluse_lora_a800_lowmem \
  --deepspeed /root/Self-Distillation/ds_config_zero2_a800.json \
  --use_lora true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --torch_dtype bfloat16 \
  --max_prompt_length 768 \
  --max_completion_length 512 \
  --use_vllm false \
  --report_to none
```

## Important Note

Your current environment shows a warning that `TRL` prefers `vllm 0.10.2`, while the installed version is `0.12.0`.

Because of that:

- first run should use `--use_vllm false`
- only enable vLLM after the base training path is confirmed stable

