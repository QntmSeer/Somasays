import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from peft import PeftModel

def generate_peptide_candidates(
    base_model_name: str = "EvolutionaryScale/esm3-sm-open-v1",
    lora_weights_dir: str = "../weights/somasays_lora_v1",
    num_candidates: int = 10,
    output_file: str = "outputs/generated_batch_1.jsonl",
    template: str = "MKAL<mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask>"
):
    """
    Uses the fine-tuned ESM3 model to generate novel plant-like peptides.
    Because ESM is a Masked Language Model (MLM), we generate by iteratively unmasking a template.
    """
    print(f"Loading Base Model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    if os.path.exists(lora_weights_dir):
        print(f"Loading Somasays LoRA Adapters from {lora_weights_dir}")
        model = PeftModel.from_pretrained(model, lora_weights_dir)
    else:
        print("WARNING: LoRA weights not found. Running generation with BASE model for pilot test.")
        
    print("Initializing Masked Language Modeling Pipeline...")
    # ESM models use <mask> token
    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer, device_map="auto")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generated_sequences = {}
    
    print(f"Generating {num_candidates} sequences from template: {template}")
    
    # Simple iterative unmasking for pilot
    # For production, a more sophisticated Gibbs Sampling approach is recommended.
    for i in range(num_candidates):
        current_seq = template
        # Iteratively replace masks
        while "<mask>" in current_seq:
            # fill-mask returns a list of possibilities for the first mask it finds
            # We take the top prediction
            predictions = unmasker(current_seq)
            # If multiple masks, pipeline returns a list of lists. If one, it returns a list.
            if isinstance(predictions[0], list):
                best_pred = predictions[0][0]['sequence']
            else:
                best_pred = predictions[0]['sequence']
            
            # The pipeline usually replaces ALL masks if we pass multiple, 
            # or we handle them one by one. For simplicity, we just take the output sequence.
            current_seq = best_pred.replace(" ", "") # ESM sometimes adds spaces between tokens
            
            # Safety break if no masks are left or if the pipeline replaced them all
            if "<mask>" not in current_seq:
                break
                
        # Clean up sequence (remove any special tokens like <s> or <eos>)
        clean_seq = current_seq.replace(tokenizer.cls_token, "").replace(tokenizer.eos_token, "").replace("<pad>", "").strip()
        
        candidate_id = f"candidate_{i+1:03d}"
        generated_sequences[candidate_id] = clean_seq
        print(f"[{candidate_id}] Generated: {clean_seq}")

    with open(output_file, 'w') as f:
        for cid, seq in generated_sequences.items():
            f.write(json.dumps({"candidate_id": cid, "sequence": seq}) + "\n")
            
    print(f"Saved generated candidates to {output_file}")

if __name__ == "__main__":
    # Generate 5 peptides for the local pilot run
    generate_peptide_candidates(num_candidates=5)
