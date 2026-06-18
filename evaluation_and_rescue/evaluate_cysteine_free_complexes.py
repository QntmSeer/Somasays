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

try:
    from evaluation_and_rescue import esmfold2_local_folder as _ef2
    _HAS_EF2 = True
except ImportError:
    _HAS_EF2 = False

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
from evaluation_and_rescue.structural_qc import predict_mhc_epitopes, predict_tcr_activation_epitopes, parse_atoms_for_qc
from evaluation_and_rescue.binding_interface_analyzer import calculate_distance
import math

def parse_structure_atoms(structure_path: str, chain_id: str = "A") -> dict:
    """Parses a structure file (PDB or CIF) and extracts cysteine count and N-to-C distance."""
    try:
        qc_data = parse_atoms_for_qc(structure_path, chain_id)
        num_cys = len(qc_data["cys_coords"])
        n_term = qc_data["n_terminus"]
        c_term = qc_data["c_terminus"]
        
        cyclization_dist = 999.9
        cyclization_feasibility = "Infeasible"
        if n_term and c_term:
            cyclization_dist = round(calculate_distance(n_term, c_term), 2)
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
    except Exception as e:
        print(f"[WARNING] Fallback structure parsing due to: {e}")
        return {
            "num_cys": 0,
            "cyclization_dist": 999.9,
            "cyclization_feasibility": "Infeasible"
        }

def compute_wet_lab_success_score(row) -> float:
    """Computes the reality-oriented Wet-Lab Success Score (WLSS) from 0 to 100."""
    # 1. Binding Score (0 to 100)
    iptm = row.get('iptm', 0.0) if not pd.isna(row.get('iptm')) else 0.0
    orig_sc = row.get('interface_sc')
    orig_bsa = row.get('interface_area')
    pae = row.get('average_interface_pae', np.nan) if not pd.isna(row.get('average_interface_pae')) else np.nan
    
    if pd.isna(orig_sc) or pd.isna(orig_bsa):
        # ponytail: for models lacking geometric interface metrics (e.g. ESMFold2 mode),
        # use ipTM (50%) and ipSAE/PAE mapped score (50%)
        if not pd.isna(pae):
            # Map PAE from [4.0, 20.0] -> [100.0, 0.0]
            pae_score = max(0.0, min(100.0, 100.0 * (20.0 - pae) / (20.0 - 4.0)))
        else:
            pae_score = 0.0
        binding_score = 0.5 * (iptm * 100) + 0.5 * pae_score
    else:
        # Standard AF3 mode with geometric interface parameters
        sc = float(orig_sc)
        bsa = float(orig_bsa)
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
    tcr = row.get('tcr_epitopes', 0) if not pd.isna(row.get('tcr_epitopes')) else 0
    
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
    man_score -= tcr * 15.0 # Penalty for TCR activation risk (Wang et al., 2026)
    
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


def run_local_fold(fasta_dir: str, out_cif_dir: str) -> pd.DataFrame:
    """Fold all FASTA files in fasta_dir with ESMFold2-Fast. Returns AlphaJudge-compatible DataFrame."""
    if not _HAS_EF2:
        raise ImportError("esmfold2_local_folder not available. Check ESM 3.3 install.")

    os.makedirs(out_cif_dir, exist_ok=True)
    fasta_files = sorted(
        glob.glob(os.path.join(fasta_dir, "*.fasta")) +
        glob.glob(os.path.join(fasta_dir, "*.fa"))
    )
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {fasta_dir}")

    rows = []
    for fa in fasta_files:
        name = os.path.splitext(os.path.basename(fa))[0]
        seq = ""
        with open(fa) as f:
            for line in f:
                if not line.startswith(">"):
                    seq += line.strip()
        if not seq:
            print(f"[WARNING] Empty sequence in {fa}, skipping.")
            continue

        out_cif = os.path.join(out_cif_dir, f"{name}.cif")
        print(f"[ESMFold2] Folding {name} ({len(seq)} aa)...")
        try:
            metrics = _ef2.fold_sequence(seq, out_cif)
        except Exception as e:
            print(f"  [WARNING] Folding failed for {name}: {e}")
            continue

        rows.append({
            "jobs":                    name,
            "model_used":              "ESMFold2-Fast",
            "iptm":                    metrics["iptm"],
            "ptm":                     metrics["ptm"],
            "interface_average_plddt": metrics["plddt_mean"],
            "average_interface_pae":   metrics["interface_pae"],
            "interface_area":          np.nan,
            "interface_sc":            np.nan,
        })
        print(f"  -> ptm={metrics['ptm']:.3f}  plddt={metrics['plddt_mean']:.1f}")

    if not rows:
        raise RuntimeError("No candidates were successfully folded.")
    return pd.DataFrame(rows)


def _score_and_report(df: pd.DataFrame, cif_root: str):
    """SaProtΔG stability, biophysical QC, WLSS ranking, and report generation.
    Shared by both AF3 (run_evaluation) and ESMFold2 (run_esmfold2_evaluation) paths.
    """
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

    # Step 3: Run SaProtΔG + biophysical QC on each binder
    print("[*] Step 3: Evaluating binder folding stability (ΔG) and biophysical QC for each candidate...")
    foldseek_bin = os.path.join(predictor_root, "bin/foldseek")

    stability_results = []
    gravy_results = []
    pi_results = []
    sol_ratio_results = []
    mhc_results = []
    tcr_results = []
    sequence_results = []
    cysteines_results = []
    cyclization_dist_results = []
    cyclization_feasibility_results = []

    for idx, row in df.iterrows():
        job_dir_name = row['jobs']
        model_name = row['model_used']

        # ponytail: flat path first (ESMFold2), then subdirectory search (AF3)
        direct_cif = os.path.join(cif_root, f"{job_dir_name}.cif")
        if os.path.exists(direct_cif):
            cif_path = direct_cif
        else:
            cif_files = []
            for pattern in [
                os.path.join(cif_root, job_dir_name, f"*{model_name}*.cif"),
                os.path.join(cif_root, job_dir_name, "*_model.cif"),
                os.path.join(cif_root, job_dir_name, f"*{model_name}*.pdb"),
                os.path.join(cif_root, job_dir_name, "*_model.pdb"),
            ]:
                cif_files = glob.glob(pattern)
                if cif_files:
                    break
            if not cif_files:
                print(f"[WARNING] Could not find structure file for {job_dir_name} ({model_name})")
                for lst in [stability_results, gravy_results, pi_results, sol_ratio_results,
                            mhc_results, tcr_results, cysteines_results, cyclization_dist_results]:
                    lst.append(np.nan)
                sequence_results.append("")
                cyclization_feasibility_results.append("N/A")
                continue
            cif_path = cif_files[0]

        # Structure QC
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
                    m, pdb_path=cif_path, chain_id='A', foldseek_path=foldseek_bin
                )
                preds.append(pred_dg_avg[0])
                if c_seq is not None:
                    combined_seq = c_seq

            avg_dg = sum(preds) / len(preds)
            print(f"    -> Predicted \u0394G: {avg_dg:.2f} kcal/mol")
            stability_results.append(avg_dg)

            if combined_seq is not None:
                amino_acid_sequence = combined_seq[::2]
                sequence_results.append(amino_acid_sequence)
                gravy_results.append(calculate_gravy(amino_acid_sequence))
                pi_results.append(estimate_isoelectric_point(amino_acid_sequence))
                sol_ratio_results.append(calculate_solubility_ratio(amino_acid_sequence))
                mhc_results.append(len(predict_mhc_epitopes(amino_acid_sequence)))
                tcr_results.append(len(predict_tcr_activation_epitopes(amino_acid_sequence, threshold=0.5)))
            else:
                sequence_results.append("")
                for lst in [gravy_results, pi_results, sol_ratio_results, mhc_results, tcr_results]:
                    lst.append(np.nan)
        except Exception as e:
            print(f"    [WARNING] Stability prediction failed: {e}")
            stability_results.append(np.nan)
            sequence_results.append("")
            for lst in [gravy_results, pi_results, sol_ratio_results, mhc_results, tcr_results]:
                lst.append(np.nan)

    df['predicted_dG']            = stability_results
    df['gravy']                   = gravy_results
    df['pi']                      = pi_results
    df['solubility_ratio']        = sol_ratio_results
    df['mhc_epitopes']            = mhc_results
    df['tcr_epitopes']            = tcr_results
    df['sequence']                = sequence_results
    df['cysteines']               = cysteines_results
    df['cyclization_dist']        = cyclization_dist_results
    df['cyclization_feasibility'] = cyclization_feasibility_results

    # Step 4: Leaderboard
    print("[*] Step 4: Formatting and sorting consolidated leaderboard...")
    df['candidate_name'] = df['jobs'].apply(lambda x: x.replace("fold_", ""))
    df['wlss'] = [compute_wet_lab_success_score(row) for _, row in df.iterrows()]
    df = df.sort_values(by=['wlss', 'iptm'], ascending=[False, False])

    csv_out = "outputs/joint_evaluation_report.csv"
    df.to_csv(csv_out, index=False)

    md_out = "outputs/joint_evaluation_report.md"
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write("# \U0001f3c6 Unified Cysteine-Free Binder Validation Leaderboard\n\n")
        f.write("This leaderboard ranks candidates using a reality-oriented **Wet-Lab Success Score (WLSS)** which integrates docking kinetics (AF3/AlphaJudge), monomer thermodynamic folding stability (SaProtΔG), and manufacturing risk factors (PTMs, aggregation, immunogenicity).\n\n")
        f.write("### Metric Explanations:\n")
        f.write("- **WLSS (%)**: Composite reality-oriented probability of wet-lab success (0% to 100%; higher is better). Weighs 50% Binding, 30% Folding, 20% Manufacturability.\n")
        f.write("- **ipTM / pTM**: Model confidence (higher is better).\n")
        f.write("- **Int. pLDDT**: Average pLDDT score of residues at the binding interface (higher is more confident).\n")
        f.write("- **folding $\\Delta G$**: Absolute folding stability of the monomer (lower is more stable; $\\Delta G < 0$ folds stably).\n")
        f.write("- **GRAVY**: Kyte-Doolittle hydropathy index (negative values are hydrophilic/soluble; positive are hydrophobic).\n")
        f.write("- **pI**: Isoelectric Point (should be distant from physiological pH 7.4 to avoid aggregation).\n")
        f.write("- **MHC-II**: Predicted HLA-DRB1 immunogenicity binding core count (0 indicates extremely safe).\n")
        f.write("- **TCR Risk**: T-cell receptor activation epitope count (0 is optimal; Wang et al., 2026).\n")
        f.write("- **ipSAE (\u00c5)**: Interface Predicted Alignment Error. Mapped from both cross-chain PAE blocks (lower is more confident/better binding).\n")
        f.write("- **Warnings**: Active chemical/physical alerts (PTMs, neutral pI, free cysteines, low solubility ratio, TCR activation risk).\n\n")

        f.write("| Rank | Candidate Name | WLSS (%) | ipTM | pTM | Int. pLDDT | ipSAE (\u00c5) | Buried Area (\u00c5\u00b2) | folding $\\Delta G$ (kcal/mol) | GRAVY | pI | MHC-II | TCR Risk | Warnings | Folder Name |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for rank, (_, row) in enumerate(df.iterrows(), 1):
            dg_str = f"**{row['predicted_dG']:.2f}**" if not pd.isna(row['predicted_dG']) else "N/A"
            if not pd.isna(row['predicted_dG']) and row['predicted_dG'] > 1.5:
                dg_str += " \u26a0\ufe0f (Unstable)"
            wlss_str   = f"**{row['wlss']:.1f}%**"
            iptm_s     = f"{row['iptm']:.3f}"
            ptm_s      = f"{row['ptm']:.3f}"
            int_plddt  = f"{row['interface_average_plddt']:.1f}" if 'interface_average_plddt' in row and not pd.isna(row['interface_average_plddt']) else "N/A"
            ipsae_s    = f"{row['average_interface_pae']:.2f}" if 'average_interface_pae' in row and not pd.isna(row['average_interface_pae']) else "N/A"
            area       = f"{row['interface_area']:.1f}" if 'interface_area' in row and not pd.isna(row.get('interface_area')) else "N/A"
            gravy_s    = f"{row['gravy']:.2f}" if not pd.isna(row['gravy']) else "N/A"
            pi_s       = f"{row['pi']:.2f}" if not pd.isna(row['pi']) else "N/A"
            mhc_s      = f"{int(row['mhc_epitopes'])}" if not pd.isna(row['mhc_epitopes']) else "N/A"
            tcr_s      = f"{int(row['tcr_epitopes'])}" if not pd.isna(row['tcr_epitopes']) else "N/A"

            warnings = []
            if row.get('cysteines', 0) > 0:
                warnings.append("Free Cys")
            if not pd.isna(row.get('pi')) and abs(row['pi'] - 7.4) < 0.6:
                warnings.append("Neutral pI")
            if not pd.isna(row.get('solubility_ratio')) and row['solubility_ratio'] < 0.8:
                warnings.append("Low Sol. Ratio")
            if not pd.isna(row.get('predicted_dG')) and row['predicted_dG'] > 1.5:
                warnings.append("Unstable folding")
            if not pd.isna(row.get('tcr_epitopes')) and row['tcr_epitopes'] > 0:
                warnings.append("TCR-Activation")
            seq = row.get('sequence', '')
            if seq:
                ptms = scan_ptm_and_instability(seq)
                if ptms.get("n_glyco"):       warnings.append("Glycosylation")
                if ptms.get("deamidation"):   warnings.append("Deamidation")
                if ptms.get("acid_cleavage"): warnings.append("Acid-Cleavage")

            f.write(f"| {rank} | {row['candidate_name']} | {wlss_str} | {iptm_s} | {ptm_s} | {int_plddt} | {ipsae_s} | {area} | {dg_str} | {gravy_s} | {pi_s} | {mhc_s} | {tcr_s} | {', '.join(warnings) or 'None'} | `{row['jobs']}` |\n")

    print("=======================================================")
    print("[SUCCESS] Joint evaluation complete!")
    print(f"   -> Consolidated CSV: {csv_out}")
    print(f"   -> Markdown Leaderboard: {md_out}")
    print("=======================================================")


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

    _score_and_report(df, translated_downloads)


def run_esmfold2_evaluation(fasta_dir: str):
    """Fold FASTA candidates locally with ESMFold2-Fast, then run the full scoring pipeline."""
    if not _HAS_EF2:
        raise ImportError("esmfold2_local_folder not importable — check ESM 3.3 install.")
    out_cif_dir = "outputs/esmfold2_structures"
    print("=======================================================")
    print(f"[*] Starting ESMFold2-Fast Local Evaluation Pipeline...")
    print(f"[*] FASTA source : {fasta_dir}")
    print(f"[*] CIF output   : {out_cif_dir}")
    print("=======================================================")
    os.makedirs("outputs", exist_ok=True)
    df = run_local_fold(fasta_dir, out_cif_dir)
    _score_and_report(df, out_cif_dir)

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Somasays evaluation pipeline")
    _parser.add_argument("path", nargs="?", default="C:/Users/Gebruiker/Downloads",
                         help="AF3 downloads dir (--mode af3) or FASTA dir (--mode esmfold2)")
    _parser.add_argument("--mode", choices=["af3", "esmfold2"], default="esmfold2",
                         help="Folding backend (default: esmfold2 = local GPU)")
    _args, _ = _parser.parse_known_args()

    if _args.mode == "esmfold2":
        run_esmfold2_evaluation(_args.path)
    else:
        run_evaluation(_args.path)
