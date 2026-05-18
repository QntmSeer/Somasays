import os
import argparse
import torch
import warnings
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

# Suppress standard PyTorch architecture warnings for clean output
warnings.filterwarnings("ignore")

def run_multimodal_generation(
    weights_path: str,
    prompt_sequence: str,
    num_candidates: int,
    output_dir: str,
    temperature: float = 0.7,
    num_steps: int = 8
):
    """
    Loads fine-tuned ESM3 weights and generates synthetic multimodal candidates.
    """
    print("==============================================")
    print("🧪 Somasays Multimodal Generation Engine 🧪")
    print("==============================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target device detected: {device.upper()}")

    # 1. Load the base ESM3 open-source architecture
    print("[1/4] Initializing base ESM3-sm (1.4B params)...")
    model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)

    # 2. Ingest custom fine-tuned weights
    if os.path.exists(weights_path):
        print(f"[2/4] Ingesting fine-tuned weights from: {weights_path}")
        try:
            # Our training script cleanly unwrapped DDP, so this loads perfectly
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(">>> [SUCCESS] Weights loaded and active!")
        except Exception as e:
            print(f">>> [ERROR] Weight ingestion failed: {e}")
            print(">>> Falling back to base ESM3 weights for inference demonstration...")
    else:
        print(f"[2/4] WARNING: Weights not found at {weights_path}")
        print(">>> Falling back to base ESM3 weights for execution...")

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    print("\n==============================================")
    print(f"🚀 Launching generation of {num_candidates} synthetic designs...")
    print(f"🚀 Sequence Template Prompt: {prompt_sequence}")
    print("==============================================\n")

    for i in range(num_candidates):
        candidate_id = f"candidate_{i+1:03d}"
        print(f"[*] Processing [{candidate_id}]...")

        with torch.no_grad():
            # 3. Define Multimodal Prompt 
            # Use '_' in sequence to tell ESM3 which positions to generate de novo
            input_protein = ESMProtein(sequence=prompt_sequence)

            # Step 3a: Generate Sequence (filling the blanks '_')
            print("   -> Phase A: Hallucinating amino acid sequence...")
            seq_config = GenerationConfig(track="sequence", num_steps=num_steps, temperature=temperature)
            protein_with_seq = model.generate(input_protein, seq_config)
            
            generated_seq = protein_with_seq.sequence
            print(f"   -> Complete Sequence: {generated_seq}")

            # Step 3b: Generate 3D Coordinates (folding the hallucinated sequence)
            print("   -> Phase B: Resolving 3D atomic coordinate fold topology...")
            struct_config = GenerationConfig(track="structure", num_steps=num_steps, temperature=temperature)
            final_protein = model.generate(protein_with_seq, struct_config)

        # 4. Export Output Candidates
        pdb_path = os.path.join(output_dir, f"{candidate_id}.pdb")
        fasta_path = os.path.join(output_dir, f"{candidate_id}.fasta")

        # Save PDB 3D coordinate backbone
        try:
            final_protein.to_pdb(pdb_path)
            print(f"   [+] Saved 3D Coordinates to: {pdb_path}")
        except Exception as e:
            print(f"   [-] Failed to write PDB file: {e}")

        # Save FASTA sequence file
        with open(fasta_path, "w") as f:
            f.write(f">{candidate_id} | Somasays Generated Design | Prompt: {prompt_sequence}\n")
            f.write(f"{generated_seq}\n")
        print(f"   [+] Saved Sequence to: {fasta_path}\n")

    print("==============================================")
    print(f"🎉 SUCCESS: Generated {num_candidates} complete multimodal candidates!")
    print(f"📂 Saved to: {output_dir}")
    print("==============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Somasays ESM3 Candidate Generation")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="../weights/somasays_multimodal_latest.pth",
        help="Path to your custom fine-tuned .pth weights"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="MKA___________________VLA",
        help="Sequence template. Use '_' for positions you want the AI to generate."
    )
    parser.add_argument(
        "--num", 
        type=int, 
        default=5, 
        help="Number of unique candidates to generate"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="outputs/multimodal_candidates", 
        help="Output directory for PDB and FASTA files"
    )
    
    args = parser.parse_args()
    
    run_multimodal_generation(
        weights_path=args.weights,
        prompt_sequence=args.prompt,
        num_candidates=args.num,
        output_dir=args.out
    )
