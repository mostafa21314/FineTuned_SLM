import os
import argparse
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
import wandb
from tqdm import tqdm
from fine_tune import load_and_prepare_data
import time

def load_chunks(chunks_file):
    """Loads text chunks into a dictionary keyed by chunk_id."""
    print(f"Loading context chunks from {chunks_file}...")
    chunks_lookup = {}
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            chunks_lookup[item['chunk_id']] = item['text']
    return chunks_lookup


def evaluate_model(args):
    wandb.init(project="egx30-banks-qa-finetune", tags=["evaluation"], config=args)
    
    # 1. Load Dataset and Context Chunks
    dataset_dict = load_and_prepare_data(args.test_file, args.chunks_file)
    test_data = dataset_dict['test']
    
    chunks_lookup = load_chunks(args.chunks_file)
    
    if args.limit:
        test_data = test_data.select(range(args.limit))
    
    # 2. Metrics & Models
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,  # Change from float16 to bfloat16
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # Add this line
    )

    ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    ft_model.eval()

    # 4. Evaluation Loop
    # Batch Helper
    def get_batches(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data.select(range(i, min(i + batch_size, len(data))))

    # 4. Evaluation Loop
    results = []
    print("Running inference with Context Augmentation...")
    
    for batch in tqdm(get_batches(test_data, args.batch_size), total=(len(test_data) + args.batch_size - 1) // args.batch_size):
        original_questions = batch['question']
        ground_truths = batch['answer']
        chunk_ids = batch['chunk_id']
        
        # Prepare batch messages
        merged_messages = []
        for q, cid in zip(original_questions, chunk_ids):
            context_text = chunks_lookup.get(cid, "No context available.")
            augmented_content = f"Context: {context_text}\n\nQuestion: {q}"
            merged_messages.append([{"role": "user", "content": augmented_content}])

        # Apply chat template to get strings (tokenize=False)
        text_inputs = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in merged_messages]
        
        # Tokenize batch
        inputs = tokenizer(
            text_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, # Safety (model max length assumed)
            max_length=2048
        ).to(base_model.device)

        max_new_tokens = 256
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        with torch.no_grad():
            # Base Model
            with ft_model.disable_adapter():
                start_time = time.time()
                base_out = ft_model.generate(**inputs, **gen_kwargs)
                base_time = time.time() - start_time
            
            # Fine-tuned Model
            start_time = time.time()
            ft_out = ft_model.generate(**inputs, **gen_kwargs)
            ft_time = time.time() - start_time
            
        # ... (decoding) ...
        input_len = inputs.input_ids.shape[1]
        
        # Calculate tokens generated per batch
        base_tokens_generated = base_out.shape[1] - input_len
        ft_tokens_generated = ft_out.shape[1] - input_len
        
        # Avoid zero division
        base_tokens_generated = max(1, base_tokens_generated)
        ft_tokens_generated = max(1, ft_tokens_generated)

        # Throughput (tokens/sec) - Average for the batch
        # Note: This is "batch throughput". Per-sample latency is rougher to estimate without per-beam timestamps.
        base_throughput = (base_tokens_generated * inputs.input_ids.shape[0]) / base_time
        ft_throughput = (ft_tokens_generated * inputs.input_ids.shape[0]) / ft_time
        
        # Latency (ms/token)
        base_latency = (base_time * 1000) / (base_tokens_generated * inputs.input_ids.shape[0])
        ft_latency = (ft_time * 1000) / (ft_tokens_generated * inputs.input_ids.shape[0])

        base_preds = tokenizer.batch_decode(base_out[:, input_len:], skip_special_tokens=True)
        ft_preds = tokenizer.batch_decode(ft_out[:, input_len:], skip_special_tokens=True)

        for q, gt, base, ft, cid in zip(original_questions, ground_truths, base_preds, ft_preds, chunk_ids):
            results.append({
                "question": q,
                "context": chunks_lookup.get(cid, "")[:200] + "...",
                "ground_truth": gt,
                "base_prediction": base.strip() if base.strip() else ".",
                "finetuned_prediction": ft.strip() if ft.strip() else ".",
                "base_latency": base_latency,
                "ft_latency": ft_latency,
                "base_throughput": base_throughput,
                "ft_throughput": ft_throughput
            })
    
    # 5. Compute Metrics
    predictions_base = [r["base_prediction"] if r["base_prediction"].strip() else "." for r in results]
    predictions_ft = [r["finetuned_prediction"] if r["finetuned_prediction"].strip() else "." for r in results]
    references = [r["ground_truth"] for r in results]

    # ROUGE
    rouge_base = rouge.compute(predictions=predictions_base, references=references)
    rouge_ft = rouge.compute(predictions=predictions_ft, references=references)

    # BLEU
    bleu_base = bleu.compute(predictions=predictions_base, references=references)
    bleu_ft = bleu.compute(predictions=predictions_ft, references=references)
    
    # Speed Metrics Average
    avg_base_latency = sum(r['base_latency'] for r in results) / len(results)
    avg_ft_latency = sum(r['ft_latency'] for r in results) / len(results)
    avg_base_throughput = sum(r['base_throughput'] for r in results) / len(results)
    avg_ft_throughput = sum(r['ft_throughput'] for r in results) / len(results)

    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'Base Model':<15} | {'Fine-tuned':<15}")
    print("-" * 55)
    print(f"{'ROUGE-1':<20} | {rouge_base['rouge1']:.4f}          | {rouge_ft['rouge1']:.4f}")
    print(f"{'ROUGE-2':<20} | {rouge_base['rouge2']:.4f}          | {rouge_ft['rouge2']:.4f}")
    print(f"{'ROUGE-L':<20} | {rouge_base['rougeL']:.4f}          | {rouge_ft['rougeL']:.4f}")
    print(f"{'BLEU':<20}    | {bleu_base['bleu']:.4f}            | {bleu_ft['bleu']:.4f}")
    print(f"{'Latency (ms/tok)':<20} | {avg_base_latency:.2f}             | {avg_ft_latency:.2f}")
    print(f"{'Throughput (tok/s)':<20}| {avg_base_throughput:.2f}             | {avg_ft_throughput:.2f}")
    print("="*50 + "\n")

    # Create Table for Comparison (First 10 examples)
    df_results = pd.DataFrame(results)
    print("Sample Comparison:")
    print(df_results[['question', 'ground_truth', 'base_prediction', 'finetuned_prediction']].head(10).to_markdown(index=False))

    # Log to WandB
    wandb.log({
        "base_rouge1": rouge_base['rouge1'],
        "ft_rouge1": rouge_ft['rouge1'],
        "base_rougeL": rouge_base['rougeL'],
        "ft_rougeL": rouge_ft['rougeL'],
        "base_bleu": bleu_base['bleu'],
        "ft_bleu": bleu_ft['bleu'],
        "base_latency_ms": avg_base_latency,
        "ft_latency_ms": avg_ft_latency,
        "base_throughput_tok_s": avg_base_throughput,
        "ft_throughput_tok_s": avg_ft_throughput,
        "comparison_table": wandb.Table(dataframe=df_results)
    })
    
    # Save results to file
    output_file = os.path.join(args.adapter_path, "evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "metrics": {
                "base": {
                    "rouge1": rouge_base['rouge1'],
                    "rouge2": rouge_base['rouge2'],
                    "rougeL": rouge_base['rougeL'],
                    "bleu": bleu_base['bleu']
                },
                "finetuned": {
                    "rouge1": rouge_ft['rouge1'],
                    "rouge2": rouge_ft['rouge2'],
                    "rougeL": rouge_ft['rougeL'],
                    "bleu": bleu_ft['bleu']
                }
            },
            "examples": results[:20]  # Save first 20 examples
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Model vs Baseline")
    parser.add_argument("--test_file", default="data/processed/qna_dataset.jsonl", help="Path to QnA dataset")
    parser.add_argument("--chunks_file", default="data/processed/chunks.jsonl", help="Path to chunks file")
    parser.add_argument("--base_model_id", default="google/gemma-3-270m-it", help="Base model ID")
    parser.add_argument("--adapter_path", default="models/fine_tuned", help="Path to LoRA adapters")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    
    args = parser.parse_args()
    evaluate_model(args)