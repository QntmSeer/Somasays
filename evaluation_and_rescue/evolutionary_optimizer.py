import os
import sys
import random
import math
import csv
import torch
import gc
import argparse
from typing import List, Tuple

# Add Somasays root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import ESM3 and SaProt tools
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

# Add absolute stability predictor to path
predictor_root = os.path.abspath("evaluation_and_rescue/absolute-stability-predictor")
sys.path.append(predictor_root)

from SaProtdG import SaProtdG, SaProtdG_predict
from evaluation_and_rescue.structural_qc import predict_mhc_epitopes, predict_tcr_activation_epitopes
from evaluation_and_rescue.codon_optimizer_carbon import optimize_codons
from evaluation_and_rescue.manufacturability_profiler import estimate_isoelectric_point

# Hydrophobicity values (Kyte-Doolittle)
GRAVY_VALUES = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def calculate_gravy(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(GRAVY_VALUES.get(aa, 0.0) for aa in sequence) / len(sequence)

def run_evolution():
    parser = argparse.ArgumentParser(description="In Silico Directed Evolution of Cysteine-Free Binder")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulated annealing steps")
    parser.add_argument("--temp-start", type=float, default=0.5, help="Starting temperature for Simulated Annealing")
    parser.add_argument("--cooling-rate", type=float, default=0.95, help="Cooling factor per step")
    parser.add_argument("--test-run", action="store_true", help="Runs a quick 5-step test")
    parser.add_argument("--output", type=str, default="outputs/evolution_history.csv", help="Path to write search history")
    parser.add_argument("--out-dir", type=str, default="outputs/evolved_candidates", help="Directory to save best candidates")
    
    args = parser.parse_args()
    
    if args.test_run:
        args.steps = 5
        print("[*] Running in TEST-RUN mode (5 steps).")
        
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 1. Configuration
    seed_seq = "MKARRLAAGLLAAAEEAKKAAPVLA"
    
    # Protected indices (0-indexed): interface residues + starting Methionine
    protected_indices = {0, 1, 4, 5, 8, 9, 12, 13, 15, 16, 17, 19, 20, 23, 24}
    unprotected_indices = [i for i in range(len(seed_seq)) if i not in protected_indices]
    
    # Non-cysteine, non-proline alphabet
    mutation_alphabet = "ADEFGHIKLMNQRSTVWY"
    
    print("==============================================")
    # Highlight the protected head and project context
    print("  Somasays Cysteine-Free Directed Evolution Loop")
    print("==============================================")
    print(f"[*] Seed sequence: {seed_seq} (length {len(seed_seq)})")
    print(f"[*] Protected positions (0-indexed): {sorted(list(protected_indices))}")
    print(f"[*] Unprotected scaffold positions: {unprotected_indices}")
    print(f"[*] Mutating using alphabet: {mutation_alphabet}")
    print("==============================================\n")
    
    # 2. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading ESM3 model ('esm3_sm_open_v1') on {device.upper()}...")
    esm_model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    esm_model.eval()
    
    print("[*] Loading SaProtΔG ensemble...")
    weights_paths = [
        os.path.join(predictor_root, "saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt"),
        os.path.join(predictor_root, "saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt"),
        os.path.join(predictor_root, "saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt"),
    ]
    saprot_models = []
    for wp in weights_paths:
        if os.path.exists(wp):
            saprot_models.append(SaProtdG(wp))
        else:
            print(f"[ERROR] Missing weight file: {wp}")
            sys.exit(1)
            
    foldseek_bin = os.path.join(predictor_root, "bin/foldseek")
    
    # Helper function to fold with ESM3 and score with SaProt
    def evaluate_candidate(seq: str, filename_prefix: str) -> float:
        # Step A: Fold using ESM3
        p = ESMProtein(sequence=seq)
        cfg = GenerationConfig(track="structure", num_steps=8, temperature=0.7)
        with torch.no_grad():
            folded = esm_model.generate(p, cfg)
            
        temp_pdb = os.path.join(args.out_dir, f"{filename_prefix}.pdb")
        folded.to_pdb(temp_pdb)
        
        # Step B: Score using SaProtΔG ensemble
        preds = []
        for m in saprot_models:
            _, pred_dg_avg, _ = SaProtdG_predict(
                m,
                pdb_path=temp_pdb,
                chain_id='A',
                foldseek_path=foldseek_bin
            )
            if pred_dg_avg is not None and len(pred_dg_avg) > 0:
                preds.append(pred_dg_avg[0])
            else:
                preds.append(9.9) # default penalty
                
        avg_dg = sum(preds) / len(preds)
        return avg_dg, temp_pdb

    # 3. Baseline Evaluation
    print("\n[*] Evaluating baseline Candidate 041 seed sequence...")
    baseline_dg, baseline_pdb = evaluate_candidate(seed_seq, "candidate_041_baseline")
    baseline_gravy = calculate_gravy(seed_seq)
    baseline_mhc = len(predict_mhc_epitopes(seed_seq))
    baseline_pi = estimate_isoelectric_point(seed_seq)
    baseline_tcr = len(predict_tcr_activation_epitopes(seed_seq, threshold=0.5))
    
    print(f"  -> Baseline Predicted dG: {baseline_dg:.3f} kcal/mol")
    print(f"  -> Baseline pI: {baseline_pi:.2f}")
    print(f"  -> Baseline GRAVY: {baseline_gravy:.3f}")
    print(f"  -> Baseline MHC-II Epitopes: {baseline_mhc}")
    print(f"  -> Baseline TCR Epitopes: {baseline_tcr}")
    
    # Keep track of best candidates
    best_candidates = [] # list of tuples: (dg, sequence, mhc, tcr, gravy, pdb_path)
    best_candidates.append((baseline_dg, seed_seq, baseline_mhc, baseline_tcr, baseline_gravy, baseline_pdb))
    
    curr_seq = seed_seq
    curr_dg = baseline_dg
    curr_gravy = baseline_gravy
    curr_mhc = baseline_mhc
    curr_tcr = baseline_tcr
    
    temp = args.temp_start
    cooling_rate = args.cooling_rate
    
    history = []
    history.append({
        "step": 0,
        "sequence": seed_seq,
        "dG": baseline_dg,
        "gravy": baseline_gravy,
        "pi": baseline_pi,
        "mhc_epitopes": baseline_mhc,
        "tcr_epitopes": baseline_tcr,
        "accepted": True,
        "reason": "Seed Sequence"
    })
    
    # 4. Search Loop
    print(f"\n[*] Starting directed evolution search for {args.steps} steps...")
    for step in range(1, args.steps + 1):
        print(f"\n--- Step {step}/{args.steps} (Temp: {temp:.4f}) ---")
        
        # Create mutant sequence
        mut_list = list(curr_seq)
        # Select 1 or 2 positions to mutate
        num_mutations = random.choice([1, 2])
        pos_to_mutate = random.sample(unprotected_indices, num_mutations)
        
        mut_details = []
        for pos in pos_to_mutate:
            old_aa = mut_list[pos]
            new_aa = random.choice([aa for aa in mutation_alphabet if aa != old_aa])
            mut_list[pos] = new_aa
            mut_details.append(f"{old_aa}{pos+1}{new_aa}")
            
        mut_seq = "".join(mut_list)
        mut_desc = ", ".join(mut_details)
        print(f"  Mutation proposed: {mut_desc} -> Sequence: {mut_seq}")
        
        # Constraint checks:
        # A. Solubility Check (GRAVY must not be significantly higher than baseline to prevent hydrophobic aggregation)
        mut_gravy = calculate_gravy(mut_seq)
        max_allowed_gravy = max(baseline_gravy, 0.45)
        if mut_gravy > max_allowed_gravy:
            print(f"  [REJECTED] GRAVY score too hydrophobic: {mut_gravy:.3f} > {max_allowed_gravy:.3f}")
            history.append({
                "step": step, "sequence": mut_seq, "dG": 9.9, "gravy": mut_gravy, "pi": estimate_isoelectric_point(mut_seq),
                "mhc_epitopes": 99, "tcr_epitopes": 99, "accepted": False, "reason": "GRAVY threshold exceeded"
            })
            continue
            
        # B. Immunogenicity Check (MHC-II epitope count must not exceed baseline)
        mut_mhc = len(predict_mhc_epitopes(mut_seq))
        if mut_mhc > baseline_mhc:
            print(f"  [REJECTED] Created new MHC-II epitopes: {mut_mhc} > {baseline_mhc}")
            history.append({
                "step": step, "sequence": mut_seq, "dG": 9.9, "gravy": mut_gravy, "pi": estimate_isoelectric_point(mut_seq),
                "mhc_epitopes": mut_mhc, "tcr_epitopes": 99, "accepted": False, "reason": "New MHC-II epitopes created"
            })
            continue
            
        # B2. TCR Specificity Activation Check (Wang et al., 2026)
        mut_tcr = len(predict_tcr_activation_epitopes(mut_seq, threshold=0.5))
        if mut_tcr > baseline_tcr:
            print(f"  [REJECTED] Created new TCR activation epitopes: {mut_tcr} > {baseline_tcr}")
            history.append({
                "step": step, "sequence": mut_seq, "dG": 9.9, "gravy": mut_gravy, "pi": estimate_isoelectric_point(mut_seq),
                "mhc_epitopes": mut_mhc, "tcr_epitopes": mut_tcr, "accepted": False, "reason": "New TCR activation epitopes created"
            })
            continue
            
        # C. Isoelectric Point Check (pI must not be near physiological pH 7.4)
        mut_pi = estimate_isoelectric_point(mut_seq)
        if 6.8 <= mut_pi <= 8.0:
            print(f"  [REJECTED] pI near physiological pH (aggregation risk): {mut_pi:.2f}")
            history.append({
                "step": step, "sequence": mut_seq, "dG": 9.9, "gravy": mut_gravy, "pi": mut_pi,
                "mhc_epitopes": mut_mhc, "tcr_epitopes": 99, "accepted": False, "reason": f"pI near physiological pH: {mut_pi:.2f}"
            })
            continue
            
        # Fold and score stability
        temp_name = f"step_{step}_mutant"
        try:
            mut_dg, mut_pdb = evaluate_candidate(mut_seq, temp_name)
        except Exception as e:
            print(f"  [WARNING] Folding/scoring failed: {e}")
            if os.path.exists(os.path.join(args.out_dir, f"{temp_name}.pdb")):
                os.remove(os.path.join(args.out_dir, f"{temp_name}.pdb"))
            continue
            
        delta = mut_dg - curr_dg
        accepted = False
        reason = ""
        
        # Metropolis selection criteria
        if delta < 0:
            accepted = True
            reason = f"Stability improved (dG: {mut_dg:.3f} < {curr_dg:.3f})"
        else:
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                accepted = True
                reason = f"Metropolis accepted (dG: {mut_dg:.3f} >= {curr_dg:.3f}, P={prob:.3f})"
            else:
                accepted = False
                reason = f"Stability worsened (dG: {mut_dg:.3f} >= {curr_dg:.3f}, P={prob:.3f})"
                
        if accepted:
            print(f"  [ACCEPTED] {reason}")
            curr_seq = mut_seq
            curr_dg = mut_dg
            curr_gravy = mut_gravy
            curr_mhc = mut_mhc
            curr_tcr = mut_tcr
            
            # Save accepted pdb with custom name
            final_mut_pdb = os.path.join(args.out_dir, f"accepted_step_{step}.pdb")
            if os.path.exists(mut_pdb):
                os.rename(mut_pdb, final_mut_pdb)
                
            # Add to best candidates list
            best_candidates.append((mut_dg, mut_seq, mut_mhc, mut_tcr, mut_gravy, final_mut_pdb))
        else:
            print(f"  [REJECTED] {reason}")
            # Delete temporary pdb
            if os.path.exists(mut_pdb):
                os.remove(mut_pdb)
                
        history.append({
            "step": step,
            "sequence": mut_seq,
            "dG": mut_dg,
            "gravy": mut_gravy,
            "pi": mut_pi,
            "mhc_epitopes": mut_mhc,
            "tcr_epitopes": mut_tcr,
            "accepted": accepted,
            "reason": reason
        })
        
        # Cool temperature
        temp *= cooling_rate
        
    # Write search history
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "sequence", "dG", "gravy", "pi", "mhc_epitopes", "tcr_epitopes", "accepted", "reason"])
        writer.writeheader()
        writer.writerows(history)
        
    print(f"\n[*] Evolution search history saved to: {args.output}")
    
    # 5. Extract and Process Top Candidates
    # Sort best candidates by stability (dG)
    best_candidates = sorted(list(set(best_candidates)), key=lambda x: x[0])
    
    print("\n==============================================")
    print("      🏆 Directed Evolution Top Leads")
    print("==============================================")
    
    top_n = min(3, len(best_candidates))
    for r in range(top_n):
        dg, seq, mhc, tcr, gravy, pdb_path = best_candidates[r]
        rank_name = f"evolved_rank_{r+1}"
        
        # Save FASTA
        fasta_path = os.path.join(args.out_dir, f"{rank_name}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{rank_name} | Evolved Cysteine-Free Binder | dG={dg:.3f} | GRAVY={gravy:.3f} | MHC={mhc} | TCR={tcr}\n{seq}\n")
            
        # Copy best PDB to final name
        final_pdb = os.path.join(args.out_dir, f"{rank_name}.pdb")
        import shutil
        if os.path.exists(pdb_path):
            shutil.copy2(pdb_path, final_pdb)
            
        print(f"Rank {r+1}: {seq} (dG: {dg:.3f} kcal/mol, GRAVY: {gravy:.3f}, MHC: {mhc}, TCR: {tcr})")
        print(f"   -> Saved: {final_pdb}")
        print(f"   -> Saved: {fasta_path}")
        
    # Clean up non-rank accepted files to keep outputs tidy
    all_files = os.listdir(args.out_dir)
    for file in all_files:
        if file.startswith("accepted_step_") or file.startswith("step_"):
            os.remove(os.path.join(args.out_dir, file))
            
    # 6. Codon Optimization on the Top Lead
    if len(best_candidates) > 0:
        best_dg, best_seq, best_mhc, best_tcr, best_gravy, _ = best_candidates[0]
        print(f"\n[*] Running Simulated Annealing Codon Optimization for Top Lead: {best_seq}")
        
        # Optimize for E. Coli and Human hosts
        ecoli_opt = optimize_codons(best_seq, host="e_coli", steps=1000)
        human_opt = optimize_codons(best_seq, host="human", steps=1000)
        
        ecoli_csv = "outputs/evolved_top_lead_ecoli_codon_optimized.csv"
        human_csv = "outputs/evolved_top_lead_human_codon_optimized.csv"
        
        for host_res, path in [(ecoli_opt, ecoli_csv), (human_opt, human_csv)]:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["host", "protein_sequence", "dna_sequence", "gc_content", "cai", "cai_score", "restriction_sites_remaining", "max_homopolymer_run"])
                writer.writerow([
                    "e_coli" if "ecoli" in path else "human",
                    best_seq,
                    host_res["dna_sequence"],
                    host_res["gc_content"],
                    host_res["cai"],
                    host_res["score"],
                    ",".join(host_res["restriction_sites"]) if host_res["restriction_sites"] else "None",
                    host_res["max_homopolymer_run"]
                ])
                
        print(f"   [+] E. Coli Codon Optimized DNA saved to: {ecoli_csv}")
        print(f"   [+] Human Codon Optimized DNA saved to: {human_csv}")
        
    print("\n==============================================")
    print("[SUCCESS] Somasays Directed Evolution Complete!")
    print("==============================================")

if __name__ == "__main__":
    run_evolution()
