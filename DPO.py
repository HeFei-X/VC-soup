
import os
import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_filter_data(train_file, consistency_threshold=0.0):
    """Load training data and filter by consistency."""
    logger.info(f"Loading data from: {train_file}")
    dataset = load_dataset("csv", data_files=train_file, split="train")
    
    initial_size = len(dataset)
    dataset = dataset.filter(lambda x: x["Consistency"] >= consistency_threshold)
    filtered_size = len(dataset)
    
    logger.info(f"Data size: {initial_size} -> {filtered_size} "
               f"(kept {filtered_size/initial_size:.1%})")
    return dataset


def load_eval_data(helpful_file, harm_file):
    """Load and combine evaluation data."""
    eval_helpful = load_dataset("json", data_files=helpful_file, split="train")
    eval_harm = load_dataset("json", data_files=harm_file, split="train")
    combined = concatenate_datasets([eval_helpful, eval_harm])
    return combined.shuffle(seed=42)


def setup_model(model_path, lora_r=64, lora_alpha=128):
    """Setup model with LoRA."""
    logger.info(f"Loading model: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M")
    
    return model, tokenizer


def train_dpo(
    model_path,
    train_file,
    output_dir,
    consistency_threshold=0.0,
    eval_helpful="RLHF/dataset/helpful_val_formed.jsonl",
    eval_harm="RLHF/dataset/harm_val_formed.jsonl",
    num_epochs=3,
    batch_size=2,
    eval_steps=1000,
    gpu_id="0",
    seed=60
):
    """Main training function."""
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    set_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DPO FINE-TUNING")
    logger.info("=" * 60)
    
    # Load data
    train_dataset = load_and_filter_data(train_file, consistency_threshold)
    eval_dataset = load_eval_data(eval_helpful, eval_harm)
    
    # Setup model
    model, tokenizer = setup_model(model_path)
    
    # Training config
    dpo_config = DPOConfig(
        output_dir=f"{output_dir}/checkpoints",
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
    )
    
    # Train
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("=" * 60)
    logger.info(f"Training complete! Model saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning with consistency filtering")
    
    parser.add_argument("--model-path", type=str, required=True,default="./Llama-2-7b", help="Base model path")
    parser.add_argument("--train-file", type=str, required=True, default="outputs/RWDPO/ConsistencySoup/helpful_normed.csv",help="Normalized CSV file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--consistency-threshold", type=float, required=True,default=0.0, help="Min consistency")
    parser.add_argument("--num-epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Eval frequency")
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU ID")
    
    args = parser.parse_args()
    
    train_dpo(
        model_path=args.model_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        consistency_threshold=args.consistency_threshold,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        gpu_id=args.gpu_id
    )


if __name__ == "__main__":
    main()