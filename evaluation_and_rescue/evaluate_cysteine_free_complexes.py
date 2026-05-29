import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import torch
import shutil
import csv
from pathlib import Path

# Add Somasays root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the predictor directory to path
predictor_root = os.path.abspath("evaluation_and_rescue/absolute-stability-predictor")
sys.path.append(predictor_root)

try:
    from SaProtdG import SaProtdG, SaProtdG_predict
except ImportError:
    print("[WARNING] Could not import SaProtdG from absolute-stability-predictor. Ensure setup completed successfully.")

# Import Somasays biophysical tools
from evaluation_and_rescue.manufacturability_profiler import (
    calculate_gravy,
    estimate_isoelectric_point,
    calculate_solubility_ratio,
    detect_hydrophobic_patches,
    scan_ptm_and_instability
)
from evaluation_and_rescue.structural_qc import predict_mhc_epitopes
import math

def parse_structure_atoms(structure_path: str, chain_id: str = "A") -> dict:
    """Parses a structure file (PDB or CIF) and extracts cysteine count and N-to-C distance."""
    cys_residues = {}
    n_atom_coord = None
    c_atom_coord = None
    first_res_num = None
    last_res_num = None
    atoms = []
    
    if structure_path.endswith(".cif"):
        # ModelCIF parser
        with open(structure_path, 'r', encoding='utf-8') as f:
            in_atom_site = False
            for line in f:
                if line.startswith("loop_"):
                    in_atom_site = False
                if line.startswith("_atom_site.group_PDB"):
                    in_atom_site = True
                    continue
                if in_atom_site and line.startswith("ATOM"):
                    parts = line.split()
                    if len(parts) >= 13:
                        c_id = parts[6]
                        if c_id != chain_id:
                            continue
                        atom_type = parts[3]
                        res_name = parts[5]
                        try:
                            res_num = int(parts[8])
                            x = float(parts[10])
                            y = float(parts[11])
                            z = float(parts[12])
                        except ValueError:
                            continue
                        atoms.append({
                            "type": atom_type,
                            "res_name": res_name,
                            "res_num": res_num,
                            "coord": (x, y, z)
                        })
    else:
        # PDB parser
        with open(structure_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    c_id = line[21].strip()
                    if not c_id:
                        c_id = "A"
                    if c_id != chain_id:
                        continue
                    atom_type = line[12:16].strip()
                    res_name = line[17:20].strip()
                    try:
                        res_num = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                    except ValueError:
                        continue
                    atoms.append({
                        "type": atom_type,
                        "res_name": res_name,
                        "res_num": res_num,
                        "coord": (x, y, z)
                    })
                    
    for a in atoms:
        atom_name = a["type"]
        res_name = a["res_name"]
        res_num = a["res_num"]
        x, y, z = a["coord"]
        
        # N-terminus
        if atom_name == "N":
            if first_res_num is None or res_num < first_res_num:
                first_res_num = res_num
                n_atom_coord = (x, y, z)
        # C-terminus
        if atom_name in ["C", "O"]:
            if last_res_num is None or res_num > last_res_num:
                last_res_num = res_num
                c_atom_coord = (x, y, z)
        # Cysteine
        if res_name == "CYS":
            if res_num not in cys_residues:
                cys_residues[res_num] = {}
            cys_residues[res_num][atom_name] = (x, y, z)
            
    num_cys = len(cys_residues)
    
    # Calculate cyclization distance
    cyclization_dist = 999.9
    cyclization_feasibility = "Infeasible"
    if n_atom_coord and c_atom_coord:
        dx = n_atom_coord[0] - c_atom_coord[0]
        dy = n_atom_coord[1] - c_atom_coord[1]
        dz = n_atom_coord[2] - c_atom_coord[2]
        cyclization_dist = round(math.sqrt(dx*dx + dy*dy + dz*dz), 2)
        if cyclization_dist <= 6.0:
            cyclization_feasibility = "Highly Feasible"
        elif cyclization_dist <= 10.0:
            cyclization_feasibility = "Feasible with Linker"
        else:
            cyclization_feasibility = "Infeasible (High Strain)"
            
    return {
        "num_cys": num_cys,
        "cyclization_dist": cyclization_dist,
        "cyclization_feasibility": cyclization_feasibility
    }

def compute_wet_lab_success_score(row) -> float:
    """Computes the reality-oriented Wet-Lab Success Score (WLSS) from 0 to 100."""
    # 1. Binding Score (0 to 100)
    iptm = row.get('iptm', 0.0) if not pd.isna(row.get('iptm')) else 0.0
    sc = row.get('interface_sc', 0.0) if not pd.isna(row.get('interface_sc')) else 0.0
    bsa = row.get('interface_area', 0.0) if not pd.isna(row.get('interface_area')) else 0.0
    
    binding_score = 0.4 * (iptm * 100) + 0.3 * (sc * 100) + 0.3 * min(100.0, bsa / 15.0)
    
    # 2. Folding Score (0 to 100)
    dg = row.get('predicted_dG', 9.9) if not pd.isna(row.get('predicted_dG')) else 9.9
    folding_score = 100.0 / (1.0 + math.exp(0.8 * (dg - 0.5)))
    
    # 3. Manufacturability Score (0 to 100)
    man_score = 100.0
    gravy = row.get('gravy', 0.0) if not pd.isna(row.get('gravy')) else 0.0
    pi = row.get('pi', 7.4) if not pd.isna(row.get('pi')) else 7.4
    sol_ratio = row.get('solubility_ratio', 1.0) if not pd.isna(row.get('solubility_ratio')) else 1.0
    mhc = row.get('mhc_epitopes', 0) if not pd.isna(row.get('mhc_epitopes')) else 0
    
    seq = row.get('sequence', '')
    if seq:
        ptms = scan_ptm_and_instability(seq)
        patches = detect_hydrophobic_patches(seq)
        man_score -= len(ptms.get("n_glyco", [])) * 20.0
        man_score -= len(ptms.get("deamidation", [])) * 10.0
        man_score -= len(ptms.get("acid_cleavage", [])) * 15.0
        man_score -= len([p for p in ptms.get("oxidation", []) if p[1] == 'M']) * 5.0
        man_score -= len(patches) * 15.0
        
    if gravy > 0.0:
        man_score -= (gravy * 20.0)
    if sol_ratio < 0.8:
        man_score -= 15.0
        
    pi_diff = abs(pi - 7.4)
    if pi_diff < 1.0:
        man_score -= (1.0 - pi_diff) * 20.0
        
    man_score -= mhc * 10.0
    
    num_cys = row.get('cysteines', 0) if not pd.isna(row.get('cysteines')) else 0
    if num_cys > 0:
        man_score -= 25.0
        
    man_score = max(0.0, min(100.0, man_score))
    
    wlss = 0.5 * binding_score + 0.3 * folding_score + 0.2 * man_score
    return round(wlss, 1)

def preprocess_af3_downloads(downloads_dir: str):
    print("[*] Preprocessing AlphaFold 3 Server outputs for AlphaJudge...")
    # Find all subdirectories containing summary confidence JSONs
    all_subdirs = [os.path.join(downloads_dir, d) for d in os.listdir(downloads_dir) if os.path.isdir(os.path.join(downloads_dir, d))]
    fold_dirs = []
    for sd in all_subdirs:
        if glob.glob(os.path.join(sd, "*summary_confidences_*.json")):
            fold_dirs.append(sd)
            
    for fd in fold_dirs:
        dp = Path(fd)
        
        # Find all summary confidence files to extract job prefix
        summary_files = sorted(dp.glob("*summary_confidences_*.json"))
        if not summary_files:
            continue
            
        # The file prefix is everything before _summary_confidences_X.json
        sf_name = summary_files[0].name
        job_prefix = sf_name.split("_summary_confidences_")[0]
        
        # Check if ranking_scores.csv already exists
        csv_path = dp / f"{job_prefix}_ranking_scores.csv"
        if csv_path.exists():
            continue
            
        # Find best model
        best_idx = 0
        best_score = -1.0
        scores = {}
        for sf in summary_files:
            idx_str = sf.stem.split("_")[-1]
            try:
                idx = int(idx_str)
            except ValueError:
                continue
                
            with open(sf, "r") as f:
                data = json.load(f)
                score = data.get("ranking_score", 0.0)
                scores[idx] = score
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    
        # Copy/link best model files
        cif_src = dp / f"{job_prefix}_model_{best_idx}.cif"
        cif_dst = dp / f"{job_prefix}_model.cif"
        
        sum_src = dp / f"{job_prefix}_summary_confidences_{best_idx}.json"
        sum_dst = dp / f"{job_prefix}_summary_confidences.json"
        
        fd_src = dp / f"{job_prefix}_full_data_{best_idx}.json"
        fd_dst = dp / f"{job_prefix}_confidences.json"
        
        for src, dst in [(cif_src, cif_dst), (sum_src, sum_dst), (fd_src, fd_dst)]:
            if src.exists() and not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[WARNING] Failed to copy {src.name} to {dst.name}: {e}")
                    
        # Write ranking_scores.csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ranking_score", "seed", "sample"])
            writer.writeheader()
            writer.writerow({"ranking_score": best_score, "seed": 0, "sample": best_idx})
        print(f"[+] Preprocessed {dp.name} (Best Model: {best_idx}, Score: {best_score})")

def run_evaluation(downloads_dir: str):
    print("=======================================================")
    # Translate path if running in WSL but passed Windows style paths
    if downloads_dir.lower().startswith("c:"):
        translated_downloads = downloads_dir.replace("\\", "/").replace("C:", "/mnt/c").replace("c:", "/mnt/c")
    else:
        translated_downloads = downloads_dir

    print(f"[*] Starting Joint AlphaFold 3 Evaluation Pipeline...")
    print(f"[*] Windows Path: {downloads_dir}")
    print(f"[*] WSL Path: {translated_downloads}")
    print("=======================================================")

    # Preprocess folders first
    preprocess_af3_downloads(translated_downloads)

    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    summary_csv = "outputs/alphajudge_summary.csv"

    # Step 1: Run AlphaJudge CLI
    print("[*] Step 1: Running AlphaJudge to analyze complex interfaces...")
    if os.path.exists(summary_csv):
        os.remove(summary_csv)
    
    # Run alphajudge as a CLI command
    alphajudge_cmd = "alphajudge"
    if shutil.which("alphajudge") is None:
        local_bin = "/home/qntm/.local/bin/alphajudge"
        if os.path.exists(local_bin):
            alphajudge_cmd = local_bin
            
    cmd = f"{alphajudge_cmd} \"{translated_downloads}\" -r -o \"{summary_csv}\" --models_to_analyse best"
    print(f"Executing: {cmd}")
    exit_code = os.system(cmd)
    
    if exit_code != 0 or not os.path.exists(summary_csv):
        print("[ERROR] AlphaJudge execution failed or did not generate summary CSV.")
        sys.exit(1)
        
    print("[+] AlphaJudge complete. Reading summary...")
    df = pd.read_csv(summary_csv)
    
    if df.empty:
        print("[ERROR] AlphaJudge summary is empty. No valid interfaces found.")
        sys.exit(1)

    # Step 2: Load SaProtΔG Ensemble Model
    print("[*] Step 2: Loading SaProtΔG ensemble model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running SaProtΔG on: {device}")

    weights_paths = [
        os.path.join(predictor_root, "saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt"),
        os.path.join(predictor_root, "saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt"),
        os.path.join(predictor_root, "saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt"),
    ]

    models = []
    for wp in weights_paths:
        if not os.path.exists(wp):
            print(f"[ERROR] Weight file not found: {wp}")
            print("Please run setup_wsl_pipeline.sh first to download weights.")
            sys.exit(1)
        models.append(SaProtdG(wp))

    # Step 3: Run SaProtΔG and biophysical QC on each binder chain
    print("[*] Step 3: Evaluating binder folding stability (ΔG) and biophysical QC for each candidate...")
    
    foldseek_bin = os.path.join(predictor_root, "bin/foldseek")
    
    stability_results = []
    gravy_results = []
    pi_results = []
    sol_ratio_results = []
    mhc_results = []
    sequence_results = []
    cysteines_results = []
    cyclization_dist_results = []
    cyclization_feasibility_results = []
    
    for idx, row in df.iterrows():
        job_dir_name = row['jobs']
        model_name = row['model_used']
        
        search_pattern = os.path.join(translated_downloads, job_dir_name, f"*{model_name}*.cif")
        cif_files = glob.glob(search_pattern)
        if not cif_files:
            search_pattern = os.path.join(translated_downloads, job_dir_name, "*_model.cif")
            cif_files = glob.glob(search_pattern)
        if not cif_files:
            search_pattern = os.path.join(translated_downloads, job_dir_name, f"*{model_name}*.pdb")
            cif_files = glob.glob(search_pattern)
        if not cif_files:
            search_pattern = os.path.join(translated_downloads, job_dir_name, "*_model.pdb")
            cif_files = glob.glob(search_pattern)
            
        if not cif_files:
            print(f"[WARNING] Could not find structure file for {job_dir_name} ({model_name})")
            stability_results.append(np.nan)
            gravy_results.append(np.nan)
            pi_results.append(np.nan)
            sol_ratio_results.append(np.nan)
            mhc_results.append(np.nan)
            sequence_results.append("")
            cysteines_results.append(np.nan)
            cyclization_dist_results.append(np.nan)
            cyclization_feasibility_results.append("N/A")
            continue
            
        cif_path = cif_files[0]
        
        # Structure QC parsing
        try:
            struct_qc = parse_structure_atoms(cif_path, chain_id='A')
            cysteines_results.append(struct_qc["num_cys"])
            cyclization_dist_results.append(struct_qc["cyclization_dist"])
            cyclization_feasibility_results.append(struct_qc["cyclization_feasibility"])
        except Exception as e:
            print(f"    [WARNING] Structure QC parsing failed: {e}")
            cysteines_results.append(np.nan)
            cyclization_dist_results.append(np.nan)
            cyclization_feasibility_results.append("N/A")
            
        print(f"  Scoring binder folding stability for: {job_dir_name} (using {os.path.basename(cif_path)})")
        
        try:
            preds = []
            combined_seq = None
            for m in models:
                _, pred_dg_avg, c_seq = SaProtdG_predict(
                    m, 
                    pdb_path=cif_path, 
                    chain_id='A', 
                    foldseek_path=foldseek_bin
                )
                preds.append(pred_dg_avg[0])
                if c_seq is not None:
                    combined_seq = c_seq
            
            avg_dg = sum(preds) / len(preds)
            print(f"    -> Predicted ΔG: {avg_dg:.2f} kcal/mol")
            stability_results.append(avg_dg)
            
            if combined_seq is not None:
                amino_acid_sequence = combined_seq[::2]
                sequence_results.append(amino_acid_sequence)
                
                gravy = calculate_gravy(amino_acid_sequence)
                pi = estimate_isoelectric_point(amino_acid_sequence)
                sol_ratio = calculate_solubility_ratio(amino_acid_sequence)
                mhc_count = len(predict_mhc_epitopes(amino_acid_sequence))
                
                gravy_results.append(gravy)
                pi_results.append(pi)
                sol_ratio_results.append(sol_ratio)
                mhc_results.append(mhc_count)
            else:
                sequence_results.append("")
                gravy_results.append(np.nan)
                pi_results.append(np.nan)
                sol_ratio_results.append(np.nan)
                mhc_results.append(np.nan)
                
        except Exception as e:
            print(f"    [WARNING] Stability prediction failed: {e}")
            stability_results.append(np.nan)
            sequence_results.append("")
            gravy_results.append(np.nan)
            pi_results.append(np.nan)
            sol_ratio_results.append(np.nan)
            mhc_results.append(np.nan)
            
    # Append stability results to DataFrame
    df['predicted_dG'] = stability_results
    df['gravy'] = gravy_results
    df['pi'] = pi_results
    df['solubility_ratio'] = sol_ratio_results
    df['mhc_epitopes'] = mhc_results
    df['sequence'] = sequence_results
    df['cysteines'] = cysteines_results
    df['cyclization_dist'] = cyclization_dist_results
    df['cyclization_feasibility'] = cyclization_feasibility_results

    # Step 4: Consolidate Leaderboard and Save Reports
    print("[*] Step 4: Formatting and sorting consolidated leaderboard...")
    
    df['candidate_name'] = df['jobs'].apply(lambda x: x.replace("fold_", ""))
    
    # Calculate composite WLSS
    wlss_scores = []
    for _, row in df.iterrows():
        wlss_scores.append(compute_wet_lab_success_score(row))
    df['wlss'] = wlss_scores
    
    # Sort: Primary sort by WLSS descending, secondary by ipTM descending
    df = df.sort_values(by=['wlss', 'iptm'], ascending=[False, False])
    
    csv_out = "outputs/joint_evaluation_report.csv"
    df.to_csv(csv_out, index=False)
    
    md_out = "outputs/joint_evaluation_report.md"
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write("# 🏆 Unified Cysteine-Free Binder Validation Leaderboard\n\n")
        f.write("This leaderboard ranks candidates using a reality-oriented **Wet-Lab Success Score (WLSS)** which integrates docking kinetics (AF3/AlphaJudge), monomer thermodynamic folding stability (SaProtΔG), and manufacturing risk factors (PTMs, aggregation, immunogenicity).\n\n")
        f.write("### Metric Explanations:\n")
        f.write("- **WLSS (%)**: Composite reality-oriented probability of wet-lab success (0% to 100%; higher is better). Weighs 50% Binding, 30% Folding, 20% Manufacturability.\n")
        f.write("- **ipTM / pTM**: AlphaFold 3 model confidence (higher is better).\n")
        f.write("- **Int. pLDDT**: Average pLDDT score of residues at the binding interface (higher is more confident).\n")
        f.write("- **folding $\\Delta G$**: Absolute folding stability of the monomer (lower is more stable; $\\Delta G < 0$ folds stably).\n")
        f.write("- **GRAVY**: Kyte-Doolittle hydropathy index (negative values are hydrophilic/soluble; positive are hydrophobic).\n")
        f.write("- **pI**: Isoelectric Point (should be distant from physiological pH 7.4 to avoid aggregation).\n")
        f.write("- **MHC-II**: Predicted HLA-DRB1 immunogenicity binding core count (0 indicates extremely safe).\n")
        f.write("- **Warnings**: Active chemical/physical alerts (PTMs, neutral pI, free cysteines, low solubility ratio).\n\n")
        
        f.write("| Rank | Candidate Name | WLSS (%) | ipTM | pTM | Int. pLDDT | Buried Area (Å²) | folding $\\Delta G$ (kcal/mol) | GRAVY | pI | MHC-II | Warnings | Folder Name |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for rank, (_, row) in enumerate(df.iterrows(), 1):
            dg_str = f"**{row['predicted_dG']:.2f}**" if not pd.isna(row['predicted_dG']) else "N/A"
            if not pd.isna(row['predicted_dG']) and row['predicted_dG'] > 1.5:
                dg_str += " ⚠️ (Unstable)"
                
            wlss_str = f"**{row['wlss']:.1f}%**"
            iptm = f"{row['iptm']:.3f}"
            ptm = f"{row['ptm']:.3f}"
            int_plddt = f"{row['interface_average_plddt']:.1f}" if 'interface_average_plddt' in row and not pd.isna(row['interface_average_plddt']) else "N/A"
            area = f"{row['interface_area']:.1f}" if 'interface_area' in row else "N/A"
            gravy = f"{row['gravy']:.2f}" if not pd.isna(row['gravy']) else "N/A"
            pi = f"{row['pi']:.2f}" if not pd.isna(row['pi']) else "N/A"
            mhc = f"{int(row['mhc_epitopes'])}" if not pd.isna(row['mhc_epitopes']) else "N/A"
            
            # Formulate Warnings
            warnings = []
            if row.get('cysteines', 0) > 0:
                warnings.append("Free Cys")
            if not pd.isna(row.get('pi')) and abs(row['pi'] - 7.4) < 0.6:
                warnings.append("Neutral pI")
            if not pd.isna(row.get('solubility_ratio')) and row['solubility_ratio'] < 0.8:
                warnings.append("Low Sol. Ratio")
            if not pd.isna(row.get('predicted_dG')) and row['predicted_dG'] > 1.5:
                warnings.append("Unstable folding")
                
            seq = row.get('sequence', '')
            if seq:
                ptms = scan_ptm_and_instability(seq)
                if ptms.get("n_glyco"):
                    warnings.append("Glycosylation")
                if ptms.get("deamidation"):
                    warnings.append("Deamidation")
                if ptms.get("acid_cleavage"):
                    warnings.append("Acid-Cleavage")
                    
            warn_str = ", ".join(warnings) if warnings else "None"
            
            f.write(f"| {rank} | {row['candidate_name']} | {wlss_str} | {iptm} | {ptm} | {int_plddt} | {area} | {dg_str} | {gravy} | {pi} | {mhc} | {warn_str} | `{row['jobs']}` |\n")

    print("=======================================================")
    print("[SUCCESS] Joint evaluation complete!")
    print(f"   -> Consolidated CSV: {csv_out}")
    print(f"   -> Markdown Leaderboard: {md_out}")
    print("=======================================================")

if __name__ == "__main__":
    downloads = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Gebruiker/Downloads"
    # If passed --test-run, check for mock run directories
    if "--test-run" in sys.argv:
        downloads = "C:/Users/Gebruiker/Downloads"
        # We can find folders in downloads and run on a single folder to smoke test
    run_evaluation(downloads)
