import os
import glob
import json
import csv
import math

def calculate_distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def parse_cif_atoms(cif_path):
    atoms = []
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
    return atoms

def analyze_interface(atoms):
    chain_a = [a for a in atoms if a["chain_id"] == "A"]
    chain_b = [a for a in atoms if a["chain_id"] == "B"]
    
    if not chain_a or not chain_b:
        return 0, []
        
    contacts = []
    for a1 in chain_a:
        for a2 in chain_b:
            d = calculate_distance(a1["coord"], a2["coord"])
            if d <= 4.5:
                contacts.append((a1, a2, d))
                
    res_contacts = {}
    for a1, a2, d in contacts:
        key = (a1["res_num"], a1["res_name"], a2["res_num"], a2["res_name"])
        if key not in res_contacts or d < res_contacts[key]:
            res_contacts[key] = d
            
    # Detect salt bridges (Lys/Arg in A to Asp/Glu in B under 4.0 A)
    salt_bridges = []
    for (r_a, name_a, r_b, name_b), dist in res_contacts.items():
        if name_a in ["ARG", "LYS"] and name_b in ["ASP", "GLU"] and dist <= 4.0:
            salt_bridges.append(f"{name_a}{r_a}-{name_b}{r_b} ({dist:.2f}Å)")
            
    return len(res_contacts), salt_bridges

def run_batch_evaluation(downloads_dir: str):
    print("=======================================================")
    print("[*] Starting Batch AlphaFold 3 Run Evaluation...")
    print(f"[*] Scanning directory: {downloads_dir}")
    print("=======================================================")
    
    # Find all fold_* directories
    fold_dirs = glob.glob(os.path.join(downloads_dir, "fold_*"))
    
    results = []
    
    for fd in fold_dirs:
        if not os.path.isdir(fd):
            continue
            
        # Find the job request or look at the files to find the job name
        job_request_files = glob.glob(os.path.join(fd, "*job_request.json"))
        if not job_request_files:
            continue
            
        with open(job_request_files[0], 'r', encoding='utf-8') as f:
            job_req = json.load(f)
        
        if isinstance(job_req, list) and len(job_req) > 0:
            job_name = job_req[0].get("name", os.path.basename(fd))
        elif isinstance(job_req, dict):
            job_name = job_req.get("name", os.path.basename(fd))
        else:
            job_name = os.path.basename(fd)
        
        # Load all summary confidences
        conf_files = glob.glob(os.path.join(fd, "*summary_confidences_*.json"))
        if not conf_files:
            continue
            
        best_iptm = -1.0
        best_ptm = -1.0
        best_model_idx = -1
        
        for cf in conf_files:
            # Extract model index from filename
            # e.g., fold_..._summary_confidences_0.json
            parts = cf.split("_")
            try:
                idx = int(parts[-1].split(".")[0])
            except ValueError:
                continue
                
            with open(cf, 'r', encoding='utf-8') as f:
                conf = json.load(f)
            
            iptm = conf.get("iptm", 0.0)
            ptm = conf.get("ptm", 0.0)
            
            if iptm > best_iptm:
                best_iptm = iptm
                best_ptm = ptm
                best_model_idx = idx
                
        if best_model_idx == -1:
            continue
            
        # Analyze the best model's CIF file
        # Check both name formats (with model_idx or model_X)
        cif_pattern = os.path.join(fd, f"*model_{best_model_idx}.cif")
        cif_files = glob.glob(cif_pattern)
        
        num_contacts = 0
        salt_bridges = []
        
        if cif_files:
            try:
                atoms = parse_cif_atoms(cif_files[0])
                num_contacts, salt_bridges = analyze_interface(atoms)
            except Exception as e:
                print(f"[WARNING] Failed to parse coordinates for {job_name}: {e}")
                
        results.append({
            "name": job_name,
            "ipTM": best_iptm,
            "pTM": best_ptm,
            "best_model": f"model_{best_model_idx}",
            "contacts": num_contacts,
            "salt_bridges": ", ".join(salt_bridges) if salt_bridges else "None",
            "folder": os.path.basename(fd)
        })
        print(f"[+] Evaluated {job_name}: ipTM = {best_iptm:.2f}, pTM = {best_ptm:.2f}, Contacts = {num_contacts}")
        
    # Sort results by ipTM descending
    results.sort(key=lambda x: x["ipTM"], reverse=True)
    
    # Save CSV Report
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/af3_batch_evaluation_report.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["name", "ipTM", "pTM", "best_model", "contacts", "salt_bridges", "folder"])
        writer.writeheader()
        writer.writerows(results)
        
    # Save Markdown Leaderboard
    md_path = "outputs/af3_batch_evaluation_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 🏆 AlphaFold 3 Cysteine-Free Binder Leaderboard\n\n")
        f.write("| Rank | Candidate Job Name | ipTM | pTM | Model | Contacts (≤4.5Å) | Key Salt Bridges | Folder Name |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for rank, res in enumerate(results, 1):
            f.write(f"| {rank} | {res['name']} | **{res['ipTM']:.3f}** | {res['pTM']:.3f} | {res['best_model']} | {res['contacts']} | {res['salt_bridges']} | `{res['folder']}` |\n")
            
    print("=======================================================")
    print("[SUCCESS] Batch evaluation completed successfully!")
    print(f"   -> Markdown report: {md_path}")
    print(f"   -> CSV report: {csv_path}")
    print("=======================================================")

if __name__ == "__main__":
    import sys
    downloads = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Gebruiker/Downloads"
    run_batch_evaluation(downloads)
