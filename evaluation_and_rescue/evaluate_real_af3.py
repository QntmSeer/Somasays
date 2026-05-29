import os
import math

def calculate_distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def evaluate_af3_cif(cif_path: str):
    print("=======================================================")
    print(f"[*] Auditing AlphaFold 3 Predicted Structure: {os.path.basename(cif_path)}")
    print("=======================================================")
    
    atoms = []
    # Read CIF file
    with open(cif_path, 'r', encoding='utf-8') as f:
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
                    # Parse standard ModelCIF fields
                    # 0: ATOM, 1: id, 2: type_symbol, 3: label_atom_id, 4: label_alt_id, 5: label_comp_id
                    # 6: label_asym_id (Chain ID), 7: label_entity_id, 8: label_seq_id (Residue Number)
                    # 9: pdbx_PDB_ins_code, 10: Cartn_x, 11: Cartn_y, 12: Cartn_z
                    atom_type = parts[3]
                    res_name = parts[5]
                    chain_id = parts[6]
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
                        "chain_id": chain_id,
                        "res_num": res_num,
                        "coord": (x, y, z)
                    })
                    
    if not atoms:
        print("[ERROR] No atoms parsed from CIF file!")
        return

    # 1. Map Chains
    chain_a = [a for a in atoms if a["chain_id"] == "A"]
    chain_b = [a for a in atoms if a["chain_id"] == "B"]
    
    print(f"[+] Parsed {len(chain_a)} atoms for Chain A (Binder)")
    print(f"[+] Parsed {len(chain_b)} atoms for Chain B (Target)")
    
    # 2. Check Disulfide Bonds in Chain A (Binder)
    # Cysteines in Chain A
    cys_sgs = [a for a in chain_a if a["res_name"] == "CYS" and a["type"] == "SG"]
    print(f"\n[*] Auditing Cysteine SG-SG distances in Binder (Chain A):")
    disulfides_formed = []
    checked_pairs = set()
    for i, c1 in enumerate(cys_sgs):
        for j, c2 in enumerate(cys_sgs):
            if i >= j:
                continue
            pair = (min(c1["res_num"], c2["res_num"]), max(c1["res_num"], c2["res_num"]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            d = calculate_distance(c1["coord"], c2["coord"])
            status = "Formed Covalent Bond" if d <= 2.2 else "No Covalent Bond"
            print(f"   -> Cys{c1['res_num']} - Cys{c2['res_num']} SG-SG Distance: {d:.2f} Å [{status}]")
            if d <= 2.2:
                disulfides_formed.append(pair)
                
    # 3. Calculate Binder-Target contacts (Chain A vs Chain B)
    print(f"\n[*] Auditing Binder-Target Interaction Interface:")
    contacts = []
    for a1 in chain_a:
        for a2 in chain_b:
            d = calculate_distance(a1["coord"], a2["coord"])
            if d <= 4.5:
                contacts.append((a1, a2, d))
                
    # Find unique residue level contact pairs
    res_contacts = {}
    for a1, a2, d in contacts:
        key = (a1["res_num"], a1["res_name"], a2["res_num"], a2["res_name"])
        if key not in res_contacts or d < res_contacts[key]:
            res_contacts[key] = d
            
    print(f"   [+] Detected {len(res_contacts)} unique residue-residue contact pairs (<= 4.5 Å).")
    
    # Print contacts sorted by distance
    sorted_contacts = sorted(res_contacts.items(), key=lambda x: x[1])
    if sorted_contacts:
        print("\nTop 15 Closest Residue Contacts at Interface:")
        for (r_a, name_a, r_b, name_b), dist in sorted_contacts[:15]:
            print(f"   - Binder {name_a}{r_a} <--> Target {name_b}{r_b}: {dist:.2f} Å")
    else:
        print("   [WARNING] No close interface contacts detected. The binder has floated away from the target!")
        
    print("=======================================================")

if __name__ == "__main__":
    import sys
    cif_file = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Gebruiker/Downloads/fold_2026_05_20_01_13/fold_2026_05_20_01_13_model_0.cif"
    evaluate_af3_cif(cif_file)
