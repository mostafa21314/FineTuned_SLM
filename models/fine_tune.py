import os
import argparse
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb
from sklearn.model_selection import train_test_split

def load_chunks(chunks_file):
    """Loads text chunks into a dictionary keyed by chunk_id."""
    print(f"Loading context chunks from {chunks_file}...")
    chunks_lookup = {}
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            chunks_lookup[item['chunk_id']] = item['text']
    return chunks_lookup

def load_and_prepare_data(qna_file, chunks_file, test_size=0.1, val_size=0.1):
    """
    Loads QnA data from JSONL WITH context chunks and splits into train/test/val.
    """
    print(f"Loading data from {qna_file}...")
    
    # Load chunks
    chunks_lookup = load_chunks(chunks_file)
    
    # Load Q&A pairs
    data = []
    with open(qna_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Format data WITH context for training
    formatted_data = []
    for item in data:
        chunk_id = item['chunk_id']
        context_text = chunks_lookup.get(chunk_id, "")
        
        # Create augmented prompt with context
        user_content = f"Context: {context_text}\n\nQuestion: {item['question']}"
        
        conversation = [
            {"role": "user", "content": user_content},
            {"role": "model", "content": item['answer']}  # Fixed: use 'assistant' not 'model'
        ]
        
        # Store both messages and chunk_id for evaluation
        formatted_data.append({
            "messages": conversation,
            "chunk_id": chunk_id,
            "question": item['question'],  # Store original question for eval
            "answer": item['answer']
        })

    # Split data
    train_data, temp_data = train_test_split(
        formatted_data, 
        test_size=(test_size + val_size), 
        random_state=42
    )
    
    relative_test_size = test_size / (test_size + val_size)
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=relative_test_size, 
        random_state=42
    )

    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

    print(f"Data split: Train({len(train_data)}), Val({len(val_data)}), Test({len(test_data)})")
    return dataset_dict

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 2 on QnA dataset using LoRA.")
    parser.add_argument("--input", default="data/processed/qna_dataset.jsonl", help="Path to input JSONL dataset")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl", help="Path to chunks JSONL")
    parser.add_argument("--model_name", default="google/gemma-3-270m-it", help="Base model name")
    parser.add_argument("--output_dir", default="models/fine_tuned", help="Output directory for adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, override epochs and train for max_steps")
    parser.add_argument("--disable_quantization", action="store_true", help="Disable 4-bit quantization")

    args = parser.parse_args()

    # Initialize WandB
    if args.use_wandb:
        wandb.init(project="egx30-banks-qa-finetune", config=args)

    # 1. Load Dataset WITH chunks
    dataset = load_and_prepare_data(args.input, args.chunks)

    # 2. Quantization Config
    bnb_config = None
    if not args.disable_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # 3. Load Base Model
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bnb_config else torch.float32,
        attn_implementation="eager"  # Add this line
    )
    model.config.use_cache = False
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )

    if(args.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']  # Simplified for TinyLlama
        )

    # 6. Training Arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit" if bnb_config else "adamw_torch",
        save_steps=100,
        logging_steps=10,
        learning_rate=args.lr,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb" if args.use_wandb else "none",
        max_length=args.max_seq_length,  # Fixed: use max_length not max_seq_length
        dataset_text_field="messages",
        packing=False,
    )

    # 7. Trainer
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    model.gradient_checkpointing_enable()
    # Ensure use_cache is False (you already have this, but double check)
    model.config.use_cache = False
    # 8. Train
    trainer.train()

    # 9. Save Model
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()