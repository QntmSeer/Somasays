import os
import json

def create_mock_pdb_line(atom_id, atom_name, res_name, chain_id, res_seq, x, y, z, plddt):
    # Standard PDB ATOM format:
    # ATOM   atom_id atom_name res_name chain_id res_seq    x y z occupancy temp_factor element
    return f"ATOM  {atom_id:5d}  {atom_name:<3s} {res_name:3s} {chain_id}{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}           {atom_name[0]}\n"

def build_mock_assets(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    pdb_path = os.path.join(out_dir, "somasays_candidate_001_vs_HANTAVIRUS_GN.pdb")
    json_path = os.path.join(out_dir, "somasays_candidate_001_vs_HANTAVIRUS_GN_summary_confidences.json")
    
    # Let's write the PDB file containing a binder (Chain A) and target (Chain B)
    # Residue 1 in Chain A is Lysine (basic, nitrogen at the end)
    # Residue 1 in Chain B is Aspartate (acidic, oxygen at the end)
    # Let's put their atoms close to each other (~3.0 A) to trigger:
    # 1. Contact (dist <= 4.5)
    # 2. Hydrogen bond (N-O dist <= 3.5)
    # 3. Salt Bridge (Lys NZ to Asp OD1 dist <= 4.0)
    
    lines = []
    # Chain A - Binder (Lysine)
    lines.append(create_mock_pdb_line(1, "N", "LYS", "A", 1, 10.0, 10.0, 10.0, 95.0))
    lines.append(create_mock_pdb_line(2, "CA", "LYS", "A", 1, 11.5, 10.0, 10.0, 93.0))
    lines.append(create_mock_pdb_line(3, "C", "LYS", "A", 1, 12.0, 11.5, 10.0, 90.0))
    lines.append(create_mock_pdb_line(4, "O", "LYS", "A", 1, 11.2, 12.5, 10.0, 88.0))
    lines.append(create_mock_pdb_line(5, "NZ", "LYS", "A", 1, 13.0, 10.0, 12.0, 94.0)) # Lys side-chain Nitrogen
    
    # Chain B - Target (Aspartate)
    # Let's put Asp OD1 at (13.0, 10.0, 15.0) -> distance is 3.0 A from Lys NZ
    lines.append(create_mock_pdb_line(6, "N", "ASP", "B", 1, 10.0, 10.0, 20.0, 91.0))
    lines.append(create_mock_pdb_line(7, "CA", "ASP", "B", 1, 11.5, 10.0, 20.0, 92.0))
    lines.append(create_mock_pdb_line(8, "C", "ASP", "B", 1, 12.0, 11.5, 20.0, 89.0))
    lines.append(create_mock_pdb_line(9, "O", "ASP", "B", 1, 11.2, 12.5, 20.0, 87.0))
    lines.append(create_mock_pdb_line(10, "OD1", "ASP", "B", 1, 13.0, 10.0, 15.0, 94.0)) # Asp side-chain Oxygen
    
    # Add a far atom to verify exclusion
    lines.append(create_mock_pdb_line(11, "CA", "ALA", "B", 2, 50.0, 50.0, 50.0, 95.0))
    
    lines.append("END\n")
    
    with open(pdb_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
    # Let's write the confidence JSON
    conf_data = {
        "ptm": 0.82,
        "iptm": 0.75,
        "chain_pair_pae_min": 3.8
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(conf_data, f, indent=4)
        
    print(f"[SUCCESS] Mock structures built at:\n  - PDB: {pdb_path}\n  - JSON: {json_path}")

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "../outputs/af3_results"
    build_mock_assets(out)
