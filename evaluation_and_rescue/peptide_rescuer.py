import os
import sys
import argparse
import csv
import glob
import collections
import math
import json
import numpy as np

# Add Somasays root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation_and_rescue.binding_interface_analyzer import parse_pdb_coordinates, calculate_distance
from evaluation_and_rescue.structural_qc import analyze_structural_qc, predict_mhc_epitopes, parse_atoms_for_qc

# Full length target sequence for AF3 jobs
DEFAULT_TARGET_FULL = (
    "IYELKMECPHTVGLGQGYIIGSTELGLISIEAASDIKLESSCNFDLHTTSMAQKSFTQVEWRKKSDTTDTTNAASTTFEAQTKTVNLRGTC"
    "ILAPELYDTVKKTVLCYDLTCNQTHCQPTVYLIAPVLTCMSIRSCMASVFTSRIQVIYEKTHCVTGQLIEGQCFNPAHTLTLSQPAHTYDT"
    "VTLPISCFFTPKKSEQLKVIKTFEGILTKTGCTENALQGYYVCFLGSHSEPLIVPSLEDIRSAEVVSRMLVHPRGEDHDAIQNSQSHLRIV"
    "GPITAKVPSTSSTDTLKGTAFAGVPMYSSLSTLVRNADPEFVFSPGIVPESNHSTCDKKTVPITWTGYLPISGEME"
)

def identify_binding_interface_residues(pdb_path: str, binder_chain: str = "A", target_chain: str = "B", threshold: float = 5.0) -> set:
    """Identifies residues on the binder chain that are close to the target chain (interface)."""
    try:
        chains = parse_pdb_coordinates(pdb_path)
    except Exception as e:
        print(f"[WARNING] Failed to parse PDB for interface calculation: {e}")
        return set()
        
    if binder_chain not in chains or target_chain not in chains:
        return set()
        
    binder_atoms = chains[binder_chain]
    target_atoms = chains[target_chain]
    
    interface_res = set()
    for b_atom in binder_atoms:
        b_coord = b_atom["coord"]
        b_res_seq = b_atom["res_seq"]
        for t_atom in target_atoms:
            t_coord = t_atom["coord"]
            if calculate_distance(b_coord, t_coord) <= threshold:
                interface_res.add(b_res_seq)
                break
    return interface_res

def perform_rescue(pdb_path: str, mode: str = "cysteine-free", binder_chain: str = "A", target_chain: str = "B") -> dict:
    """
    Performs stabilizing mutations (either salt bridges or disulfides) and de-immunization
    on the binder chain.
    """
    baseline = analyze_structural_qc(pdb_path, binder_chain)
    qc_data = parse_atoms_for_qc(pdb_path, binder_chain)
    
    sequence_chars = list(qc_data["sequence"])
    cys_residues = {c["res_seq"] for c in qc_data["cys_coords"]}
    
    res_numbers = sorted(list(range(1, len(sequence_chars) + 1)))
    res_seq_to_idx = {num: i for i, num in enumerate(res_numbers) if i < len(sequence_chars)}
    idx_to_res_seq = {i: num for num, i in res_seq_to_idx.items()}
    
    interface_residues = identify_binding_interface_residues(pdb_path, binder_chain, target_chain)
    
    # Parse all CA/CB atom coordinates for candidate mutation spots
    atoms_by_res = {}
    with open(pdb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                ch = line[21].strip() or "A"
                if ch != binder_chain:
                    continue
                atom_name = line[12:16].strip()
                res_seq = int(line[22:26].strip())
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue
                if res_seq not in atoms_by_res:
                    atoms_by_res[res_seq] = {}
                atoms_by_res[res_seq][atom_name] = (x, y, z)

    mutated_res = set()
    mutations_introduced = []

    res_nums_sorted = sorted(atoms_by_res.keys())

    if mode == "cysteine-free":
        # 1. Salt Bridge engineering (E-K pairs, 4.0-6.5 A, separation >= 3)
        possible_pairs = []
        for i, r1 in enumerate(res_nums_sorted):
            if r1 in interface_residues:
                continue
            coord1 = atoms_by_res[r1].get("CB", atoms_by_res[r1].get("CA"))
            if not coord1:
                continue
                
            for j in range(i + 1, len(res_nums_sorted)):
                r2 = res_nums_sorted[j]
                if r2 in interface_residues:
                    continue
                coord2 = atoms_by_res[r2].get("CB", atoms_by_res[r2].get("CA"))
                if not coord2:
                    continue
                    
                dist = calculate_distance(coord1, coord2)
                if 4.0 <= dist <= 6.5 and abs(r1 - r2) >= 3:
                    helical_score = 1.0 if abs(r1 - r2) in [3, 4] else 0.0
                    possible_pairs.append((helical_score, dist, r1, r2))

        # Sort: prioritize helical separation, then distance
        possible_pairs.sort(key=lambda x: (-x[0], x[1]))
        
        for _, dist, r1, r2 in possible_pairs:
            if r1 in mutated_res or r2 in mutated_res:
                continue
            idx1 = res_seq_to_idx.get(r1)
            idx2 = res_seq_to_idx.get(r2)
            if idx1 is not None and idx2 is not None:
                sequence_chars[idx1] = 'E'
                sequence_chars[idx2] = 'K'
                mutated_res.add(r1)
                mutated_res.add(r2)
                mutations_introduced.append((r1, r2, round(dist, 2)))
                if len(mutations_introduced) >= 2:
                    break
    else:
        # 2. Disulfide bond engineering (Cys pairs, 3.4-5.2 A for CB, separation >= 4)
        possible_pairs = []
        for i, r1 in enumerate(res_nums_sorted):
            if r1 in interface_residues or r1 in cys_residues:
                continue
            has_cb1 = "CB" in atoms_by_res[r1]
            coord1 = atoms_by_res[r1].get("CB", atoms_by_res[r1].get("CA"))
            if not coord1:
                continue
                
            for j in range(i + 1, len(res_nums_sorted)):
                r2 = res_nums_sorted[j]
                if r2 in interface_residues or r2 in cys_residues:
                    continue
                has_cb2 = "CB" in atoms_by_res[r2]
                coord2 = atoms_by_res[r2].get("CB", atoms_by_res[r2].get("CA"))
                if not coord2:
                    continue
                    
                dist = calculate_distance(coord1, coord2)
                if abs(r1 - r2) >= 4:
                    if has_cb1 and has_cb2:
                        if 3.4 <= dist <= 5.2:
                            possible_pairs.append((dist, r1, r2))
                    else:
                        if 5.0 <= dist <= 7.2:
                            possible_pairs.append((dist, r1, r2))
        possible_pairs.sort()
        for dist, r1, r2 in possible_pairs:
            if r1 in mutated_res or r2 in mutated_res:
                continue
            idx1 = res_seq_to_idx.get(r1)
            idx2 = res_seq_to_idx.get(r2)
            if idx1 is not None and idx2 is not None:
                sequence_chars[idx1] = 'C'
                sequence_chars[idx2] = 'C'
                mutated_res.add(r1)
                mutated_res.add(r2)
                mutations_introduced.append((r1, r2, round(dist, 2)))
                if len(mutations_introduced) >= 2:
                    break

    # 3. De-immunization (HLA core anchor removal)
    current_seq = "".join(sequence_chars)
    epitopes = predict_mhc_epitopes(current_seq)
    deimmunize_mutations = []
    
    for start_idx, sub_seq, _ in epitopes:
        p1_idx = start_idx - 1
        p1_res_seq = idx_to_res_seq.get(p1_idx)
        if p1_res_seq and p1_res_seq not in interface_residues and p1_res_seq not in mutated_res:
            original_aa = sequence_chars[p1_idx]
            new_aa = 'A' if original_aa != 'A' else 'S'
            sequence_chars[p1_idx] = new_aa
            deimmunize_mutations.append({
                "res_seq": p1_res_seq,
                "original": original_aa,
                "mutated": new_aa,
                "epitope": sub_seq
            })
            
    rescued_sequence = "".join(sequence_chars)
    post_epitopes = predict_mhc_epitopes(rescued_sequence)
    post_risk = "Low"
    if len(post_epitopes) >= 3:
        post_risk = "High"
    elif len(post_epitopes) >= 1:
        post_risk = "Moderate"
        
    return {
        "original_sequence": baseline["sequence"],
        "rescued_sequence": rescued_sequence,
        "mutations_introduced": mutations_introduced,
        "original_mhc_epitopes": baseline["mhc_epitopes_count"],
        "rescued_mhc_epitopes": len(post_epitopes),
        "original_risk": baseline["immunogenicity_risk"],
        "rescued_risk": post_risk,
        "deimmunize_mutations": deimmunize_mutations
    }

def run_campaign(mode: str, in_dir: str, out_dir: str, target_full: str):
    os.makedirs(out_dir, exist_ok=True)
    report_csv = os.path.join(out_dir, f"{mode}_rescue_report.csv")
    af3_out_dir = os.path.join(out_dir, f"af3_jobs_{mode}")
    os.makedirs(af3_out_dir, exist_ok=True)
    
    pdb_files = glob.glob(os.path.join(in_dir, "*.pdb"))
    if not pdb_files:
        # Fallback to local top 200
        pdb_files = glob.glob("outputs/local_top_200/*.pdb")
        
    if not pdb_files:
        print(f"[ERROR] No candidate PDB structures found to rescue in {in_dir}.")
        sys.exit(1)
        
    rows = []
    print(f"[*] Processing {len(pdb_files)} structures for {mode} rescue campaign...")
    for fp in pdb_files:
        name = os.path.basename(fp).replace(".pdb", "")
        if "summary" in name:
            continue
            
        res = perform_rescue(fp, mode=mode)
        
        # Save FASTA
        fasta_path = os.path.join(out_dir, f"{name}.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">{name}_{mode}\n{res['rescued_sequence']}\n")
            
        # Create AF3 JSON job
        job = [{
            "name": f"somasays_{name}_vs_HANTAVIRUS_GN_{mode}",
            "modelSeeds": [1],
            "sequences": [
                {"proteinChain": {"sequence": res["rescued_sequence"], "count": 1}},
                {"proteinChain": {"sequence": target_full, "count": 1}}
            ]
        }]
        with open(os.path.join(af3_out_dir, f"somasays_{name}_vs_HANTAVIRUS_GN_{mode}.json"), 'w') as f:
            json.dump(job, f, indent=4)
            
        rows.append({
            "name": name,
            "original_sequence": res["original_sequence"],
            "rescued_sequence": res["rescued_sequence"],
            "mutations_introduced": str(res["mutations_introduced"]),
            "original_mhc_epitopes": res["original_mhc_epitopes"],
            "rescued_mhc_epitopes": res["rescued_mhc_epitopes"],
            "original_risk": res["original_risk"],
            "rescued_risk": res["rescued_risk"]
        })
        
    # Write CSV Report
    with open(report_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "original_sequence", "rescued_sequence", "mutations_introduced",
            "original_mhc_epitopes", "rescued_mhc_epitopes", "original_risk", "rescued_risk"
        ])
        writer.writeheader()
        writer.writerows(rows)
        
    # Compile top 30 combined job file
    top_30_candidates = []
    for r in rows:
        entropy = -sum((c/len(r["rescued_sequence"])) * math.log2(c/len(r["rescued_sequence"])) for c in collections.Counter(r["rescued_sequence"]).values())
        gravy = sum({
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }.get(aa, 0.0) for aa in r["rescued_sequence"]) / len(r["rescued_sequence"])
        
        risk_score = 0 if r["rescued_risk"] == "Low" else (1 if r["rescued_risk"] == "Moderate" else 2)
        score = risk_score * 10.0 + gravy - (entropy * 0.5)
        top_30_candidates.append((score, r["name"], r["rescued_sequence"]))
        
    top_30_candidates.sort(key=lambda x: x[0])
    
    combined_jobs = []
    for score, name, seq in top_30_candidates[:30]:
        combined_jobs.append({
            "name": f"somasays_{name}_vs_HANTAVIRUS_GN_{mode}",
            "modelSeeds": [1],
            "sequences": [
                {"proteinChain": {"sequence": seq, "count": 1}},
                {"proteinChain": {"sequence": target_full, "count": 1}}
            ]
        })
        
    combined_path = os.path.join("outputs", f"combined_top_30_af3_jobs_{mode}.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined_jobs, f, indent=4)
        
    print("=======================================================")
    print(f"[SUCCESS] {mode.capitalize()} rescue campaign completed!")
    print(f"   -> Report: {report_csv}")
    print(f"   -> Combined Top 30 file: {combined_path}")
    print("=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified peptide rescue engine")
    parser.add_argument("--mode", choices=["cysteine-free", "disulfide"], default="cysteine-free",
                        help="Rescue design mode (cysteine-free E-K salt bridges or disulfide covalent C-C bonds)")
    parser.add_argument("--in_dir", type=str, default="outputs/local_top_200", help="Folder containing PDB structures")
    parser.add_argument("--out_dir", type=str, default="outputs/cysteine_free_rescued", help="Output directory")
    parser.add_argument("--target_seq", type=str, default=DEFAULT_TARGET_FULL, help="Target sequence for AF3 jobs")
    args = parser.parse_args()
    
    # Adjust default outputs directory if running in disulfide mode
    if args.mode == "disulfide" and args.out_dir == "outputs/cysteine_free_rescued":
        args.out_dir = "outputs/disulfide_rescued"
        
    run_campaign(args.mode, args.in_dir, args.out_dir, args.target_seq)
