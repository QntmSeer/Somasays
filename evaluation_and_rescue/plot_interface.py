import os
import math
import matplotlib.pyplot as plt
import numpy as np

def calculate_distance(coord1, coord2):
    return math.sqrt(
        (coord1[0] - coord2[0]) ** 2 +
        (coord1[1] - coord2[1]) ** 2 +
        (coord1[2] - coord2[2]) ** 2
    )

def generate_contact_map(pdb_path, out_img):
    print(f"[*] Reading PDB coordinates from {pdb_path}...")
    
    # Read atoms for Chain A and Chain B
    chain_a = {}
    chain_b = {}
    
    with open(pdb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                chain_id = line[21].strip()
                res_seq = int(line[22:26].strip())
                atom_name = line[12:16].strip()
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue
                
                coord = (x, y, z)
                if chain_id == "A":
                    if res_seq not in chain_a:
                        chain_a[res_seq] = []
                    chain_a[res_seq].append(coord)
                elif chain_id == "B":
                    if res_seq not in chain_b:
                        chain_b[res_seq] = []
                    chain_b[res_seq].append(coord)
                    
    if not chain_a or not chain_b:
        print("[ERROR] Chain A or Chain B is empty!")
        return
        
    res_a_sorted = sorted(list(chain_a.keys()))
    res_b_sorted = sorted(list(chain_b.keys()))
    
    num_a = len(res_a_sorted)
    num_b = len(res_b_sorted)
    
    print(f"[*] Found {num_a} residues in Chain A (Gn) and {num_b} residues in Chain B (Gc).")
    
    # Initialize minimum distance matrix
    dist_matrix = np.full((num_a, num_b), 50.0) # Fill with a large default distance (50 Angstroms)
    
    # Create index maps
    idx_a = {res: i for i, res in enumerate(res_a_sorted)}
    idx_b = {res: i for i, res in enumerate(res_b_sorted)}
    
    print("[*] Computing residue distance matrix...")
    # To make this fast, check distances
    for i, res_a in enumerate(res_a_sorted):
        coords_a = chain_a[res_a]
        for j, res_b in enumerate(res_b_sorted):
            coords_b = chain_b[res_b]
            
            # Find minimum distance between any atom of res_a and res_b
            min_d = 50.0
            for ca in coords_a:
                for cb in coords_b:
                    d = calculate_distance(ca, cb)
                    if d < min_d:
                        min_d = d
            dist_matrix[i, j] = min_d
            
    print("[*] Plotting heatmap...")
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Create contact mask (distances <= 4.5 Angstroms are contacts)
    # Plot as a clean blue/indigo-to-red heatmap of distances
    # Cap distances at 15.0 Angstroms for visual contrast
    clipped_matrix = np.clip(dist_matrix, 0.0, 15.0)
    
    plt.imshow(clipped_matrix, cmap="inferno_r", aspect="auto", origin="lower",
               extent=[res_b_sorted[0], res_b_sorted[-1], res_a_sorted[0], res_a_sorted[-1]])
               
    cbar = plt.colorbar(label="Minimum Distance (Å)")
    cbar.set_ticks([0, 4.5, 8, 12, 15])
    cbar.set_ticklabels(["0 Å", "4.5 Å (Contact)", "8 Å", "12 Å", "15+ Å"])
    
    plt.title("Andes Hantavirus Gn (Chain A) vs Gc (Chain B) Glycoprotein Interface\nResidue-Residue Minimum Distance Contact Map (PDB ID: 6Y5W)", fontsize=11, fontweight="bold", pad=15)
    plt.xlabel("Gc Glycoprotein Residue Index (Chain B)", fontsize=10)
    plt.ylabel("Gn Glycoprotein Residue Index (Chain A)", fontsize=10)
    
    # Add a thin line indicating the 4.5 A contact threshold
    # Highlight regions with high contacts
    plt.tight_layout()
    plt.savefig(out_img)
    plt.close()
    
    print(f"[SUCCESS] Saved contact map to: {out_img}")

if __name__ == "__main__":
    import sys
    pdb_in = sys.argv[1] if len(sys.argv) > 1 else "../outputs/6Y5W.pdb"
    img_out = sys.argv[2] if len(sys.argv) > 2 else "../outputs/hantavirus_interface_contact_map.png"
    generate_contact_map(pdb_in, img_out)
