import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def tokenize_function(examples, tokenizer):
    # ESM models typically expect sequences as strings
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=256)

def train_esm3_lora(config_path: str):
    """
    Fine-tunes the open ESM3 1.4B model using LoRA for Masked Language Modeling.
    """
    config = load_config(config_path)
    model_name = config["model"]["name"]
    dataset_path = config["training"]["dataset_path"]
    output_dir = config["training"]["output_dir"]
    
    print(f"Loading tokenizer and model: {model_name}")
    # Note: EvolutionaryScale/esm3-sm-open-v1 might require the 'esm' library, 
    # but AutoTokenizer is used here as a standard wrapper.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model in half precision to save memory on GCP A100
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config["hardware"]["fp16"] else torch.float32,
        trust_remote_code=True,
        device_map="auto" # requires accelerate library
    )
    
    print(f"Applying LoRA Config: r={config['lora']['lora_r']}, alpha={config['lora']['lora_alpha']}")
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False, 
        r=config['lora']['lora_r'], 
        lora_alpha=config['lora']['lora_alpha'], 
        lora_dropout=config['lora']['lora_dropout'],
        # Target attention modules (query, key, value) - typical for ESM
        target_modules=["query", "key", "value"] 
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print(f"Loading dataset from {dataset_path}")
    # Load the JSONL dataset
    dataset = load_dataset('json', data_files={'train': dataset_path})
    
    # Split into Train and Validation for Sanity Checking
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )
    
    # Data collator for Masked Language Modeling (MLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["hardware"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        fp16=config["hardware"]["fp16"],
        logging_steps=10,
        eval_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=False, # PEFT MaskedLM workaround
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"] # Use the 10% validation split
    )
    
    print("Starting training loop...")
    trainer.train()
    
    print("Evaluating Model Sanity (Validation Loss & Perplexity)...")
    eval_results = trainer.evaluate()
    import math
    try:
        perplexity = math.exp(eval_results['eval_loss'])
    except OverflowError:
        perplexity = float("inf")
    print(f"Final Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"Final Model Perplexity: {perplexity:.4f}")
    
    # Save the metrics to a JSON file for later review
    import json
    with open(f"{output_dir}/sanity_metrics.json", "w") as f:
        json.dump({"eval_loss": eval_results['eval_loss'], "perplexity": perplexity}, f)
    
    print(f"Saving fine-tuned LoRA adapters to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Ensure output dir exists
    os.makedirs("../weights/somasays_lora_v1", exist_ok=True)
    train_esm3_lora("training_config.yaml")
