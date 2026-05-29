import os
import json
import csv
import glob
import math
import argparse

def parse_pdb_coordinates(pdb_path: str) -> dict:
    """
    Parses a PDB file and extracts coordinates, atom types, residues, and B-factors (pLDDT).
    Returns a dict mapping chain IDs to list of atom dicts.
    """
    chains = {}
    with open(pdb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                chain_id = line[21].strip()
                if not chain_id:
                    chain_id = "A" # Default fallback
                
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_seq = int(line[22:26].strip())
                
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    plddt = float(line[60:66].strip()) # B-factor stores pLDDT in structural models
                except ValueError:
                    continue
                
                atom_record = {
                    "atom_name": atom_name,
                    "res_name": res_name,
                    "res_seq": res_seq,
                    "coord": (x, y, z),
                    "plddt": plddt
                }
                
                if chain_id not in chains:
                    chains[chain_id] = []
                chains[chain_id].append(atom_record)
    return chains

def calculate_distance(coord1, coord2) -> float:
    return math.sqrt(
        (coord1[0] - coord2[0]) ** 2 +
        (coord1[1] - coord2[1]) ** 2 +
        (coord1[2] - coord2[2]) ** 2
    )

GLYCAN_RES_NAMES = {"NAG", "NDG", "MAN", "BMA", "GAL", "FUC", "SIA", "NANA", "BGC", "GLC", "XYL"}

def analyze_interface(chains: dict, binder_chain: str = "A") -> dict:
    """
    Analyzes the structural interface between binder_chain and all other chains.
    Calculates contacts, hydrogen bonds, salt bridges, interface pLDDT, and glycan clashes.
    """
    interface_results = {
        "num_contacts": 0,
        "num_h_bonds": 0,
        "num_salt_bridges": 0,
        "avg_binder_plddt": 0.0,
        "avg_interface_plddt": 0.0,
        "contact_residues": set(),
        "num_glycan_clashes": 0
    }
    
    if binder_chain not in chains or len(chains) < 2:
        return interface_results
        
    binder_atoms = chains[binder_chain]
    
    # Calculate average pLDDT for the entire binder
    all_binder_plddts = [atom["plddt"] for atom in binder_atoms]
    interface_results["avg_binder_plddt"] = round(sum(all_binder_plddts) / len(all_binder_plddts), 2) if all_binder_plddts else 0.0
    
    # Collate target protein vs target glycan atoms
    target_protein_atoms = []
    target_glycan_atoms = []
    for ch_id, atoms in chains.items():
        if ch_id != binder_chain:
            for atom in atoms:
                if atom["res_name"] in GLYCAN_RES_NAMES:
                    target_glycan_atoms.append(atom)
                else:
                    target_protein_atoms.append(atom)
                    
    if not target_protein_atoms and not target_glycan_atoms:
        return interface_results
        
    # Group donor/acceptor elements for simple geometry analysis
    polar_atoms = {"N", "ND", "NH", "NZ", "O", "OD", "OE", "OG", "OH"}
    basic_res = {"ARG", "LYS", "HIS"}
    acidic_res = {"ASP", "GLU"}
    
    interface_plddts = []
    
    # Scan all pairs for protein-protein interface
    for b_atom in binder_atoms:
        b_coord = b_atom["coord"]
        is_interface_atom = False
        
        # Check protein contacts
        for t_atom in target_protein_atoms:
            t_coord = t_atom["coord"]
            dist = calculate_distance(b_coord, t_coord)
            
            if dist <= 4.5:
                is_interface_atom = True
                interface_results["num_contacts"] += 1
                interface_results["contact_residues"].add(b_atom["res_seq"])
                
                # Check for Hydrogen Bonds (N/O atoms within 3.5 A)
                b_element = b_atom["atom_name"][0]
                t_element = t_atom["atom_name"][0]
                if dist <= 3.5 and (b_element in ["N", "O"] and t_element in ["N", "O"]):
                    interface_results["num_h_bonds"] += 1
                    
                # Check for Salt Bridges (acidic oxygen and basic nitrogen within 4.0 A)
                if dist <= 4.0:
                    b_is_basic = b_atom["res_name"] in basic_res and b_element == "N"
                    t_is_acidic = t_atom["res_name"] in acidic_res and t_element == "O"
                    b_is_acidic = b_atom["res_name"] in acidic_res and b_element == "O"
                    t_is_basic = t_atom["res_name"] in basic_res and t_element == "N"
                    
                    if (b_is_basic and t_is_acidic) or (b_is_acidic and t_is_basic):
                        interface_results["num_salt_bridges"] += 1
                        
        if is_interface_atom:
            interface_plddts.append(b_atom["plddt"])
            
        # Check glycan clashes (within 3.5 A of any glycan atom)
        for g_atom in target_glycan_atoms:
            dist = calculate_distance(b_coord, g_atom["coord"])
            if dist <= 3.5:
                interface_results["num_glycan_clashes"] += 1
                break # One glycan clash per binder atom is enough
            
    interface_results["avg_interface_plddt"] = round(sum(interface_plddts) / len(interface_plddts), 2) if interface_plddts else 0.0
    return interface_results

def parse_confidence_json(json_path: str) -> dict:
    """
    Parses structural prediction metadata files (e.g. summary_confidences.json from AF3).
    Extracts global pTM, interface ipTM, and interface pAE (ipSAE).
    """
    scores = {
        "ptm": 0.0,
        "iptm": 0.0,
        "ipsae_pae_min": 31.75 # Default max PAE
    }
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            scores["ptm"] = data.get("ptm", data.get("ptm_score", 0.0))
            scores["iptm"] = data.get("iptm", data.get("iptm_score", 0.0))
            
            # Extract min interface PAE (ipAE / ipSAE) if available
            if "chain_pair_pae_min" in data:
                scores["ipsae_pae_min"] = data["chain_pair_pae_min"]
            elif "ipsae" in data:
                scores["ipsae_pae_min"] = data["ipsae"]
            elif "pae" in data:
                scores["ipsae_pae_min"] = data["pae"]
        except Exception as e:
            print(f"[WARNING] Failed to parse {json_path}: {e}")
            
    return scores

def run_binding_analysis(in_dir: str, out_dir: str, binder_chain: str):
    """
    Scans the folder for PDB structure predictions and confidence files,
    then executes interface contact calculations and ranks candidates.
    """
    print("==============================================")
    print("  Somasays Binding Interface & Contact Analyzer")
    print("==============================================")
    
    pdb_files = glob.glob(os.path.join(in_dir, "*.pdb"))
    if not pdb_files:
        print(f"[WARNING] No PDB files found in {in_dir}.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    report_csv = os.path.join(out_dir, "binding_leaderboard.csv")
    
    records = []
    
    for fp in sorted(pdb_files):
        name = os.path.basename(fp).replace(".pdb", "")
        print(f"[*] Analyzing structure: {name}.pdb")
        
        # 1. Parse geometry and contacts
        chains = parse_pdb_coordinates(fp)
        geom = analyze_interface(chains, binder_chain)
        
        # 2. Check for confidence json in the same folder or structure name
        json_pattern = fp.replace(".pdb", "*.json")
        json_files = glob.glob(json_pattern)
        
        if json_files:
            conf = parse_confidence_json(json_files[0])
        else:
            # Look for general config JSON in the folder
            base_dir = os.path.dirname(fp)
            potential_json = os.path.join(base_dir, f"{name}_summary_confidences.json")
            if os.path.exists(potential_json):
                conf = parse_confidence_json(potential_json)
            else:
                conf = {"ptm": 0.0, "iptm": 0.0, "ipsae_pae_min": 31.75}
                
        # 3. Calculate custom Composite Binding Score
        iptm_contrib = conf["iptm"] * 100.0
        interface_plddt_contrib = geom["avg_interface_plddt"] * 0.5
        
        # ipSAE / Min PAE penalty
        pae_val = conf["ipsae_pae_min"]
        pae_penalty = max(0.0, pae_val * 1.5)
        
        contact_contrib = min(50.0, geom["num_contacts"] * 0.1)
        hbond_contrib = geom["num_h_bonds"] * 0.5
        
        # Glycan clash penalty (heavy penalty for targeting sugar-shielded sites)
        glycan_clash_penalty = geom["num_glycan_clashes"] * 5.0
        
        composite_score = iptm_contrib + interface_plddt_contrib + contact_contrib + hbond_contrib - pae_penalty - glycan_clash_penalty
        composite_score = round(max(0.0, composite_score), 2)
        
        records.append({
            "name": name,
            "ptm": conf["ptm"],
            "iptm": conf["iptm"],
            "ipsae": conf["ipsae_pae_min"],
            "avg_binder_plddt": geom["avg_binder_plddt"],
            "avg_interface_plddt": geom["avg_interface_plddt"],
            "contacts": geom["num_contacts"],
            "hbonds": geom["num_h_bonds"],
            "salt_bridges": geom["num_salt_bridges"],
            "glycan_clashes": geom["num_glycan_clashes"],
            "score": composite_score
        })
        
    # Sort by binding score
    records.sort(key=lambda x: x["score"], reverse=True)
    
    # Write CSV
    with open(report_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "ptm", "iptm", "ipsae", "avg_binder_plddt", 
            "avg_interface_plddt", "contacts", "hbonds", "salt_bridges", "glycan_clashes", "score"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
            
    print(f"\n[SUCCESS] Compiled binding rankings for {len(records)} complex structures!")
    print(f"   [+] Interface Leaderboard: {report_csv}")
    print("==============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Programmatically analyze binding interfaces of folded complexes")
    parser.add_argument("--in_dir", type=str, default="../outputs/af3_results", help="Folder containing folded PDB/JSON outputs")
    parser.add_argument("--out_dir", type=str, default="../outputs", help="Output folder for reports")
    parser.add_argument("--binder_chain", type=str, default="A", help="Chain ID of the de novo binder")
    args = parser.parse_args()
    
    run_binding_analysis(args.in_dir, args.out_dir, args.binder_chain)
