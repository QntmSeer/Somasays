import os
import csv
import glob
import math
import argparse
import sys

# Add Somasays root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation_and_rescue.binding_interface_analyzer import parse_pdb_coordinates, calculate_distance

# HLA-DRB1 Consensus Anchoring Preferences for MHC-II binding prediction
# P1: Large hydrophobic / aromatic
MHC_P1_RESIDUES = {"F", "Y", "W", "I", "L", "V", "M"}
# P4: Hydrophobic or charged/polar
MHC_P4_RESIDUES = {"L", "I", "V", "M", "F", "Y", "A", "D", "E"}
# P6: Small neutral
MHC_P6_RESIDUES = {"A", "G", "S", "T", "P"}
# P9: Hydrophobic or small neutral
MHC_P9_RESIDUES = {"L", "I", "V", "M", "A", "G", "S", "T"}

def predict_mhc_epitopes(sequence: str) -> list:
    """
    Predicts potential HLA-DRB1 (MHC-II) binding 9-mer epitopes using consensus anchors.
    Returns list of tuples (start_idx, 9mer_seq, score).
    """
    seq = sequence.upper()
    epitopes = []
    if len(seq) < 9:
        return epitopes
        
    for i in range(len(seq) - 8):
        sub = seq[i:i+9]
        score = 0
        if sub[0] in MHC_P1_RESIDUES:
            score += 3  # P1 is the primary anchor pocket (highly restrictive)
        if sub[3] in MHC_P4_RESIDUES:
            score += 1
        if sub[5] in MHC_P6_RESIDUES:
            score += 1
        if sub[8] in MHC_P9_RESIDUES:
            score += 1
            
        # Threshold score of 5 indicates a strong potential MHC-II binding core
        if score >= 5:
            epitopes.append((i + 1, sub, score))
            
    return epitopes

def parse_atoms_for_qc(pdb_path: str, chain_id: str = "A") -> dict:
    """
    Parses a PDB file and returns:
    - cys_atoms: list of dicts with coordinate info for CYS residues (preferring SG sulfur, falling back to CB/CA).
    - n_terminus: coordinate tuple of first N atom of the chain.
    - c_terminus: coordinate tuple of last C/O atom of the chain.
    - sequence: reconstructed amino acid sequence of the chain.
    """
    cys_residues = {}
    n_atom_coord = None
    c_atom_coord = None
    
    first_res_seq = None
    last_res_seq = None
    
    # Track sequence residue by residue to avoid repeats
    res_seq_to_name = {}
    
    # 3-letter to 1-letter amino acid code
    aa3_to_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }

    with open(pdb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                ch = line[21].strip()
                if not ch:
                    ch = "A"
                if ch != chain_id:
                    continue
                    
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_seq = int(line[22:26].strip())
                
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue
                
                # Reconstruct sequence
                if res_name in aa3_to_1:
                    res_seq_to_name[res_seq] = aa3_to_1[res_name]
                
                # Trace N-terminus (N atom of first residue sequence number)
                if atom_name == "N":
                    if first_res_seq is None or res_seq < first_res_seq:
                        first_res_seq = res_seq
                        n_atom_coord = (x, y, z)
                        
                # Trace C-terminus (C or O atom of last residue sequence number)
                if atom_name in ["C", "O"]:
                    if last_res_seq is None or res_seq > last_res_seq:
                        last_res_seq = res_seq
                        c_atom_coord = (x, y, z)
                        
                # Extract Cysteine coordinates
                if res_name == "CYS":
                    if res_seq not in cys_residues:
                        cys_residues[res_seq] = {}
                    cys_residues[res_seq][atom_name] = (x, y, z)

    # Compile cysteine coordinates (prefer SG, fall back to CB, then CA)
    cys_coords = []
    for res_seq, atoms in sorted(cys_residues.items()):
        coord = None
        atom_used = ""
        if "SG" in atoms:
            coord = atoms["SG"]
            atom_used = "SG"
        elif "CB" in atoms:
            coord = atoms["CB"]
            atom_used = "CB"
        elif "CA" in atoms:
            coord = atoms["CA"]
            atom_used = "CA"
            
        if coord:
            cys_coords.append({
                "res_seq": res_seq,
                "coord": coord,
                "atom": atom_used
            })
            
    # Reconstruct sequence string
    seq_list = [res_seq_to_name[r] for r in sorted(res_seq_to_name.keys())]
    sequence = "".join(seq_list)
    
    return {
        "cys_coords": cys_coords,
        "n_terminus": n_atom_coord,
        "c_terminus": c_atom_coord,
        "sequence": sequence
    }

def analyze_structural_qc(pdb_path: str, chain_id: str = "A") -> dict:
    """
    Performs quality control checks on the target chain structure:
    1. Cysteine pairing & disulfide bonds count.
    2. Head-to-Tail distance.
    3. MHC-II immunogenicity risk.
    """
    qc_data = parse_atoms_for_qc(pdb_path, chain_id)
    
    # 1. Head-to-Tail distance and cyclization feasibility
    n_term = qc_data["n_terminus"]
    c_term = qc_data["c_terminus"]
    cyclization_dist = 999.9
    cyclization_status = "Infeasible"
    
    if n_term and c_term:
        cyclization_dist = round(calculate_distance(n_term, c_term), 2)
        if cyclization_dist <= 6.0:
            cyclization_status = "Highly Feasible"
        elif cyclization_dist <= 10.0:
            cyclization_status = "Feasible with Linker"
        else:
            cyclization_status = "Infeasible (High Strain)"
            
    # 2. Cysteine disulfide connectivity
    cys_list = qc_data["cys_coords"]
    num_cys = len(cys_list)
    formed_disulfides = 0
    free_thiols = 0
    disulfide_pairs = []
    
    # Simple pairing algorithm: greedily pair closest neighbors
    paired = set()
    for i, c1 in enumerate(cys_list):
        if c1["res_seq"] in paired:
            continue
        best_dist = 999.9
        best_partner = None
        for j, c2 in enumerate(cys_list):
            if i == j or c2["res_seq"] in paired:
                continue
            dist = calculate_distance(c1["coord"], c2["coord"])
            if dist < best_dist:
                best_dist = dist
                best_partner = c2
                
        # In a valid disulfide bond:
        # SG-SG distance is around 2.0 A (up to 3.0 A)
        # CB-CB distance is around 3.8 A (up to 4.5 A)
        max_allowed_dist = 3.0 if (c1["atom"] == "SG" and best_partner and best_partner["atom"] == "SG") else 4.6
        
        if best_partner and best_dist <= max_allowed_dist:
            paired.add(c1["res_seq"])
            paired.add(best_partner["res_seq"])
            formed_disulfides += 1
            disulfide_pairs.append((c1["res_seq"], best_partner["res_seq"], round(best_dist, 2)))
        else:
            free_thiols += 1
            
    # 3. Immunogenicity screening
    sequence = qc_data["sequence"]
    epitopes = predict_mhc_epitopes(sequence)
    immunogenicity_score = len(epitopes)
    immunogenicity_risk = "Low"
    if immunogenicity_score >= 3:
        immunogenicity_risk = "High"
    elif immunogenicity_score >= 1:
        immunogenicity_risk = "Moderate"
        
    return {
        "sequence": sequence,
        "num_cys": num_cys,
        "formed_disulfides": formed_disulfides,
        "free_thiols": free_thiols,
        "disulfide_pairs": disulfide_pairs,
        "cyclization_dist_angstrom": cyclization_dist,
        "cyclization_feasibility": cyclization_status,
        "mhc_epitopes_count": immunogenicity_score,
        "immunogenicity_risk": immunogenicity_risk,
        "mhc_epitopes_list": [e[1] for e in epitopes]
    }

def run_structural_qc_pipeline(in_dir: str, out_dir: str, chain_id: str):
    """
    Scans the directory for PDB structures, executes structural QC analysis,
    and saves reports.
    """
    print("==============================================")
    print("  Somasays Structural QC & Immunogenicity Engine")
    print("==============================================")
    
    pdb_files = glob.glob(os.path.join(in_dir, "*.pdb"))
    if not pdb_files:
        print(f"[WARNING] No PDB files found in {in_dir}.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    report_csv = os.path.join(out_dir, "structural_qc_report.csv")
    
    records = []
    
    for fp in sorted(pdb_files):
        name = os.path.basename(fp).replace(".pdb", "")
        print(f"[*] Checking structure QC: {name}.pdb")
        
        qc = analyze_structural_qc(fp, chain_id)
        
        records.append({
            "name": name,
            "length": len(qc["sequence"]),
            "cysteines": qc["num_cys"],
            "disulfides": qc["formed_disulfides"],
            "free_thiols": qc["free_thiols"],
            "cyclization_dist": qc["cyclization_dist_angstrom"],
            "cyclization_feasibility": qc["cyclization_feasibility"],
            "mhc_epitopes": qc["mhc_epitopes_count"],
            "immunogenicity_risk": qc["immunogenicity_risk"]
        })
        
    # Write CSV
    with open(report_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "length", "cysteines", "disulfides", "free_thiols", 
            "cyclization_dist", "cyclization_feasibility", "mhc_epitopes", "immunogenicity_risk"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
            
    print(f"\n[SUCCESS] Completed Structural Quality Control for {len(records)} complexes!")
    print(f"   [+] Structural QC Report: {report_csv}")
    print("==============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate structures for disulfide knot integrity, cyclization, and immunogenicity")
    parser.add_argument("--in_dir", type=str, default="../outputs/af3_results", help="Folder containing candidate PDB structures")
    parser.add_argument("--out_dir", type=str, default="../outputs", help="Output directory for reports")
    parser.add_argument("--chain_id", type=str, default="A", help="Chain ID of the de novo binder")
    args = parser.parse_args()
    
    run_structural_qc_pipeline(args.in_dir, args.out_dir, args.chain_id)
