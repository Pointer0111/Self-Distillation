from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist
import os


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps. Defaults to num_prompts_per_batch when omitted.",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science"])
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=1024, help="Maximum completion length")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint save frequency")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging frequency")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporter backend, e.g. wandb or none")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False, help="Enable gradient checkpointing")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config path for multi-GPU training")
    parser.add_argument("--ddp_find_unused_parameters", type=str2bool, default=False, help="DDP unused parameter flag")
    parser.add_argument("--use_vllm", type=str2bool, default=True, help="Whether to use vLLM for generation")
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"], help="vLLM execution mode")
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM. Set >1 when running multi-GPU vLLM.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization reserved for colocated vLLM",
    )
    parser.add_argument("--vllm_enable_sleep_mode", type=str2bool, default=True, help="Enable vLLM sleep mode")
    parser.add_argument("--use_lora", type=str2bool, default=False, help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules for LoRA",
    )
    return parser.parse_args()


def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dir = 'data/tooluse_data/train_data'
    train_dataset = load_from_disk(train_dir) 

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_science_dataset(seed=42) -> Dataset:
    """Load and prepare science dataset with formatted prompts."""
    path = 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


if __name__ == "__main__":
    args = parse_args()
    torch_dtype = getattr(torch, args.torch_dtype)
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else args.num_prompts_per_batch
    )
    report_to = [] if args.report_to.lower() == "none" else args.report_to
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    sync_ref_model = not args.use_lora

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    config = DistilConfig(
        seed=args.seed,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        bf16=args.torch_dtype == "bfloat16",
        fp16=args.torch_dtype == "float16",
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        num_iterations=1,
        num_generations=1,
        save_steps=args.save_steps,
        max_grad_norm=1,
        report_to=report_to,
        output_dir=args.output_dir,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        log_completions=False,  # True for debugging
        sync_ref_model=sync_ref_model,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
    )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[module.strip() for module in args.lora_target_modules.split(",") if module.strip()],
        )

    if rank == 0:
        print("=== Launch Configuration ===")
        print(f"model_name: {args.model_name}")
        print(f"dataset_name: {args.dataset_name}")
        print(f"world_size: {world_size}")
        print(f"torch_dtype: {args.torch_dtype}")
        print(f"use_lora: {args.use_lora}")
        if args.use_lora:
            print(f"lora_r: {args.lora_r}")
            print(f"lora_alpha: {args.lora_alpha}")
            print(f"lora_dropout: {args.lora_dropout}")
            print(f"lora_target_modules: {args.lora_target_modules}")
        print(f"deepspeed: {args.deepspeed}")
        print(f"use_vllm: {args.use_vllm}")
        print(f"sync_ref_model: {sync_ref_model}")
        if args.use_vllm:
            print(f"vllm_mode: {args.vllm_mode}")
            print(f"vllm_tensor_parallel_size: {args.vllm_tensor_parallel_size}")
            print(f"vllm_gpu_memory_utilization: {args.vllm_gpu_memory_utilization}")
        print(f"per_device_train_batch_size: {args.per_device_train_batch_size}")
        print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"effective_prompt_batch: {args.per_device_train_batch_size * world_size * gradient_accumulation_steps}")
        print(f"max_prompt_length: {args.max_prompt_length}")
        print(f"max_completion_length: {args.max_completion_length}")
        print(f"output_dir: {args.output_dir}")
        print("============================")

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
