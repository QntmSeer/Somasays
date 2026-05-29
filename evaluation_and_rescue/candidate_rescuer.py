import os
import sys
import argparse
import csv
import glob

# Add Somasays root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation_and_rescue.binding_interface_analyzer import parse_pdb_coordinates, calculate_distance
from evaluation_and_rescue.structural_qc import analyze_structural_qc, predict_mhc_epitopes

def identify_binding_interface_residues(pdb_path: str, binder_chain: str = "A", target_chain: str = "B", threshold: float = 5.0) -> set:
    """
    Identifies residue sequence numbers in the binder chain that are within threshold distance 
    of any atom in the target chain. These should NOT be mutated to preserve binding.
    """
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
            dist = calculate_distance(b_coord, t_coord)
            if dist <= threshold:
                interface_res.add(b_res_seq)
                break
                
    return interface_res

def perform_peptide_rescue(pdb_path: str, binder_chain: str = "A", target_chain: str = "B") -> dict:
    """
    Performs disulfide engineering and de-immunization mutations on the binder chain.
    """
    print(f"[*] Starting rescue workflow for PDB: {os.path.basename(pdb_path)}")
    
    # 1. Analyze baseline structural and immunogenicity properties
    baseline = analyze_structural_qc(pdb_path, binder_chain)
    
    # Get sequence and atom coordinates
    from evaluation_and_rescue.structural_qc import parse_atoms_for_qc
    qc_data = parse_atoms_for_qc(pdb_path, binder_chain)
    
    sequence_chars = list(qc_data["sequence"])
    cys_residues = {c["res_seq"] for c in qc_data["cys_coords"]}
    
    # Map residue sequence number to its sequence index (0-indexed)
    # The sequence could have tags, let's map it based on parsing ATOM residues
    res_numbers = sorted(list({c["res_seq"] for c in qc_data["cys_coords"]} | {r for r in range(1, len(sequence_chars) + 1)}))
    
    # Let's map residue numbers to 0-based indices in sequence_chars
    res_seq_to_idx = {num: i for i, num in enumerate(res_numbers) if i < len(sequence_chars)}
    idx_to_res_seq = {i: num for num, i in res_seq_to_idx.items()}
    
    # 2. Identify binding interface residues (DO NOT MUTATE)
    interface_residues = identify_binding_interface_residues(pdb_path, binder_chain, target_chain)
    print(f"   [+] Identified {len(interface_residues)} residues at the binding interface (Chain {binder_chain} to Chain {target_chain}) to preserve.")
    
    # 3. Disulfide Bond Engineering
    # Find spatial pairs of residues (not already CYS, not on interface) whose C-beta atoms are close
    disulfides_introduced = []
    
    # Get all CA/CB atom coordinates for candidate residues
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
                
    # Propose disulfide mutations (greedy based on CB-CB distance 3.5 to 5.0 A)
    possible_pairs = []
    res_nums_sorted = sorted(atoms_by_res.keys())
    for i, r1 in enumerate(res_nums_sorted):
        if r1 in interface_residues or r1 in cys_residues:
            continue
        # Detect if CB atom is present or if we are falling back to CA
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
            # Enforce sequence separation of at least 4 residues to prevent adjacent/near-adjacent pairings
            if abs(r1 - r2) >= 4:
                if has_cb1 and has_cb2:
                    # If both residues have CB atoms, standard CB-CB distance range is 3.4 to 5.2 A
                    if 3.4 <= dist <= 5.2:
                        possible_pairs.append((dist, r1, r2))
                else:
                    # If falling back to CA atoms, standard CA-CA distance range in disulfides is 5.0 to 7.2 A
                    if 5.0 <= dist <= 7.2:
                        possible_pairs.append((dist, r1, r2))


                
    # Sort pairs by distance and select non-overlapping pairs
    possible_pairs.sort()
    paired_in_rescue = set()
    
    for dist, r1, r2 in possible_pairs:
        if r1 in paired_in_rescue or r2 in paired_in_rescue:
            continue
            
        # Perform Cysteine mutations
        idx1 = res_seq_to_idx.get(r1)
        idx2 = res_seq_to_idx.get(r2)
        
        if idx1 is not None and idx2 is not None:
            sequence_chars[idx1] = 'C'
            sequence_chars[idx2] = 'C'
            paired_in_rescue.add(r1)
            paired_in_rescue.add(r2)
            disulfides_introduced.append((r1, r2, round(dist, 2)))
            
            # Stop if we have engineered 2 disulfide bonds (enough to stabilize small fold)
            if len(disulfides_introduced) >= 2:
                break
                
    print(f"   [+] Engineered {len(disulfides_introduced)} stabilizing disulfide bonds.")
    
    # 4. De-immunization (HLA core anchor removal)
    # Identify HLA-DRB1 binding cores
    current_seq = "".join(sequence_chars)
    epitopes = predict_mhc_epitopes(current_seq)
    immunogenicity_mutations = []
    
    for start_idx, sub_seq, score in epitopes:
        # P1 is the primary anchor residue at start_idx-1 (0-indexed in sequence)
        p1_idx = start_idx - 1
        p1_res_seq = idx_to_res_seq.get(p1_idx)
        
        if p1_res_seq and p1_res_seq not in interface_residues:
            original_aa = sequence_chars[p1_idx]
            # Mutate P1 large hydrophobic/aromatic (F, Y, W, I, L, V, M) to Alanine (A) or Serine (S)
            new_aa = 'A' if original_aa != 'A' else 'S'
            sequence_chars[p1_idx] = new_aa
            immunogenicity_mutations.append({
                "res_seq": p1_res_seq,
                "original": original_aa,
                "mutated": new_aa,
                "epitope": sub_seq
            })
            
    print(f"   [+] Mutated {len(immunogenicity_mutations)} HLA-DRB1 anchor residues to reduce immunogenicity risk.")
    
    rescued_sequence = "".join(sequence_chars)
    
    # Verify post-rescue immunogenicity
    post_epitopes = predict_mhc_epitopes(rescued_sequence)
    post_risk = "Low"
    if len(post_epitopes) >= 3:
        post_risk = "High"
    elif len(post_epitopes) >= 1:
        post_risk = "Moderate"
        
    return {
        "original_sequence": baseline["sequence"],
        "rescued_sequence": rescued_sequence,
        "original_disulfides": baseline["formed_disulfides"],
        "rescued_disulfides": baseline["formed_disulfides"] + len(disulfides_introduced),
        "disulfides_introduced": disulfides_introduced,
        "original_mhc_epitopes": baseline["mhc_epitopes_count"],
        "rescued_mhc_epitopes": len(post_epitopes),
        "original_risk": baseline["immunogenicity_risk"],
        "rescued_risk": post_risk,
        "immunogenicity_mutations": immunogenicity_mutations
    }

def run_rescue_campaign(in_dir: str, out_dir: str, target_chain: str = "B"):
    os.makedirs(out_dir, exist_ok=True)
    report_csv = os.path.join(out_dir, "rescue_report.csv")
    
    # Find PDB files in the input directory
    pdb_files = glob.glob(os.path.join(in_dir, "*.pdb"))
    if not pdb_files:
        # Check in outputs/local_top_200 as fallback
        pdb_files = glob.glob("outputs/local_top_200/*.pdb")
        
    if not pdb_files:
        print("[ERROR] No candidate structures found to rescue.")
        return
        
    records = []
    for fp in sorted(pdb_files):
        name = os.path.basename(fp).replace(".pdb", "")
        # Skip summary configs or non-complex files if running on complex folder
        if "summary" in name:
            continue
            
        res = perform_peptide_rescue(fp, binder_chain="A", target_chain=target_chain)
        
        records.append({
            "name": name,
            "original_sequence": res["original_sequence"],
            "rescued_sequence": res["rescued_sequence"],
            "original_disulfides": res["original_disulfides"],
            "rescued_disulfides": res["rescued_disulfides"],
            "disulfides_introduced": str(res["disulfides_introduced"]),
            "original_mhc_epitopes": res["original_mhc_epitopes"],
            "rescued_mhc_epitopes": res["rescued_mhc_epitopes"],
            "original_risk": res["original_risk"],
            "rescued_risk": res["rescued_risk"],
            "mutations": str(res["immunogenicity_mutations"])
        })
        
    with open(report_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "original_sequence", "rescued_sequence", "original_disulfides", 
            "rescued_disulfides", "disulfides_introduced", "original_mhc_epitopes", 
            "rescued_mhc_epitopes", "original_risk", "rescued_risk", "mutations"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
            
    print("\n==============================================")
    print(f"[SUCCESS] Rescued {len(records)} candidates successfully!")
    print(f"   [+] Rescue Report Saved to: {report_csv}")
    print("==============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescue de novo peptides by introducing disulfides and removing MHC-II epitopes")
    parser.add_argument("--in_dir", type=str, default="outputs/local_top_200", help="Folder containing PDB structures")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--target_chain", type=str, default="B", help="Target antigen chain ID")
    args = parser.parse_args()
    
    run_rescue_campaign(args.in_dir, args.out_dir, args.target_chain)
