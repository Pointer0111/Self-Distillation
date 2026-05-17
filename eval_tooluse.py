import argparse
import os
import json
import torch
import numpy as np
from datasets import Dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on tooluse test set")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model or merged model directory")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to the base model directory")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to the LoRA adapter checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=1024, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (defaults to model_path)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "transformers", "vllm"],
                        help="Inference backend to use")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for transformers generation")
    return parser.parse_args()


def resolve_model_paths(args):
    if args.adapter_path:
        with open(os.path.join(args.adapter_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)
        base_model_path = args.base_model_path or adapter_config["base_model_name_or_path"]
        model_path = args.adapter_path
        output_dir = args.output_dir if args.output_dir else args.adapter_path
    else:
        if not args.model_path:
            raise ValueError("Either --model_path or --adapter_path must be provided.")
        base_model_path = args.base_model_path or args.model_path
        model_path = args.model_path
        output_dir = args.output_dir if args.output_dir else args.model_path

    return base_model_path, model_path, output_dir


def load_model_and_tokenizer_transformers(base_model_path, adapter_path=None):
    """Load model using transformers, optionally applying a LoRA adapter."""
    tokenizer_source = adapter_path or base_model_path
    print(f"Loading tokenizer from {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def load_model_and_tokenizer_vllm(model_path, gpu_memory_utilization=0.8):
    """Load model using vLLM."""
    from vllm import LLM

    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return llm, tokenizer


def load_test_data(tokenizer):
    """Load and prepare tooluse test dataset."""
    data_dir = 'data/tooluse_data/eval_data'
    data = load_from_disk(data_dir).to_list()
    
    # Format prompts
    for example in data:
        example['prompt'] = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': example['prompt']}],
            tokenize=False,
            add_generation_prompt=True
        )
    
    return data


def generate_responses_vllm(llm, tokenizer, prompts, max_new_tokens=1024, temperature=0.0):
    """Generate responses from the model using vLLM."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )

    print(f"Generating responses for {len(prompts)} prompts with vLLM...")
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def generate_responses_transformers(model, tokenizer, prompts, max_new_tokens=1024, temperature=0.0, batch_size=4):
    """Generate responses from the model using transformers."""
    responses = []
    do_sample = temperature > 0

    print(f"Generating responses for {len(prompts)} prompts with transformers...")
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_lengths = attention_mask.sum(dim=1).tolist()
        for idx, output_ids in enumerate(outputs):
            generated_ids = output_ids[int(prompt_lengths[idx]):]
            responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True))

        print(f"  Processed {min(start + batch_size, len(prompts))}/{len(prompts)} prompts")

    return responses


def extract_actions(text):
    """Extract all actions from model response."""
    return re.findall(r'Action:\s*(\w+)', text)


def extract_action_inputs(text):
    """Extract and merge all action inputs from model response."""
    json_blocks = re.findall(r'Action Input:\s*({.*?})', text, re.DOTALL)
    combined_dict = {}
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            combined_dict.update(parsed)
        except json.JSONDecodeError:
            continue
    return combined_dict


def evaluate_correctness(responses, golden_answers):
    """
    Evaluate if responses match the golden answers.
    Returns list of scores (1 for correct, 0 for incorrect).
    """
    results = []
    
    for response, golden_answer in zip(responses, golden_answers):
        # Extract predicted actions and inputs
        pred_actions = extract_actions(response)
        pred_inputs = extract_action_inputs(response)
        
        # Extract ground truth actions and inputs
        gt_actions = [item['Action'] for item in golden_answer]
        gt_inputs = {}
        for item in golden_answer:
            try:
                gt_inputs.update(json.loads(item['Action_Input']))
            except:
                pass
        
        # Check if both actions and inputs match
        actions_match = Counter(pred_actions) == Counter(gt_actions)
        inputs_match = pred_inputs == gt_inputs
        
        results.append(1 if (actions_match and inputs_match) else 0)
    
    return results


def main():
    args = parse_args()

    base_model_path, model_path, output_dir = resolve_model_paths(args)

    # Load model and data
    backend = args.backend
    if backend == "auto":
        backend = "transformers" if args.adapter_path else "vllm"

    if backend == "vllm":
        llm, tokenizer = load_model_and_tokenizer_vllm(model_path)
    else:
        llm, tokenizer = load_model_and_tokenizer_transformers(base_model_path, args.adapter_path)

    test_data = load_test_data(tokenizer)

    prompts = [example['prompt'] for example in test_data]
    golden_answers = [example['golden_answer'] for example in test_data]

    # Generate responses
    if backend == "vllm":
        responses = generate_responses_vllm(
            llm, tokenizer, prompts,
            args.max_new_tokens,
            args.temperature
        )
    else:
        responses = generate_responses_transformers(
            llm, tokenizer, prompts,
            args.max_new_tokens,
            args.temperature,
            args.batch_size,
        )

    # Evaluate correctness
    print("\nEvaluating responses...")
    scores = evaluate_correctness(responses, golden_answers)
    accuracy = np.mean(scores)

    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Correct: {sum(scores)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results_to_save = {
        "accuracy": float(accuracy),
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "per_sample_scores": scores,
        "config": {
            "model_path": model_path,
            "base_model_path": base_model_path,
            "adapter_path": args.adapter_path,
            "backend": backend,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
        }
    }

    output_path = os.path.join(output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    # Optionally save responses for inspection
    responses_path = os.path.join(output_dir, "eval_responses.json")
    with open(responses_path, "w") as f:
        json.dump([
            {
                "prompt": test_data[i]['prompt'],
                "response": responses[i],
                "golden_answer": golden_answers[i],
                "correct": bool(scores[i])
            }
            for i in range(len(responses))
        ], f, indent=2)
    print(f"Saved responses to {responses_path}")


if __name__ == "__main__":
    main()
