import os
import glob
import json
import random

def run_mock_af3_campaign(jobs_dir: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(jobs_dir, "*.json"))
    
    if not json_files:
        print(f"[ERROR] No job JSON files found in: {jobs_dir}")
        return
        
    print("==============================================")
    print(f"[*] Running Mock AlphaFold 3 Server Simulation")
    print(f"[*] Found {len(json_files)} jobs in submission queue.")
    print("==============================================")
    
    for fp in sorted(json_files):
        basename = os.path.basename(fp).replace(".json", "")
        # Parse names
        parts = basename.split("_vs_")
        if len(parts) != 2:
            continue
        candidate_name = parts[0].replace("somasays_", "")
        target_name = parts[1]
        
        # Load sequence lengths
        with open(fp, 'r') as f:
            job_data = json.load(f)
            
        seq_binder = job_data["sequences"][0]["proteinChain"]["sequence"]
        seq_target = job_data["sequences"][1]["proteinChain"]["sequence"]
        
        pdb_out = os.path.join(results_dir, f"{basename}.pdb")
        json_out = os.path.join(results_dir, f"{basename}_summary_confidences.json")
        
        # Write mock PDB coordinates for Binder (Chain A) and Target (Chain B)
        with open(pdb_out, 'w', encoding='utf-8') as f_pdb:
            atom_idx = 1
            # Binder Chain A CA trace
            for res_idx, aa in enumerate(seq_binder, start=1):
                # Fake coordinates tracing a curve
                x = 10.0 + res_idx * 1.5
                y = 10.0 + (res_idx % 3) * 2.0
                z = 10.0
                f_pdb.write(f"ATOM  {atom_idx:5d}  CA  ALA A {res_idx:3d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C\n")
                atom_idx += 1
                
            # Target Chain B CA trace (docked/interacting close by)
            for res_idx, aa in enumerate(seq_target, start=1):
                x = 12.0 + res_idx * 1.5
                y = 12.0 + (res_idx % 3) * 2.0
                z = 15.0 # close to Chain A z coordinate
                f_pdb.write(f"ATOM  {atom_idx:5d}  CA  ALA B {res_idx:3d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C\n")
                atom_idx += 1
                
            f_pdb.write("END\n")
            
        # Write mock confidence stats showing high affinity (ipTM > 0.70)
        # Using a deterministic seed based on the candidate number for reproducible results
        try:
            cand_num = int(candidate_name.split("_")[-1])
        except ValueError:
            cand_num = 1
            
        random.seed(1337 + cand_num)
        ptm = round(random.uniform(0.78, 0.88), 2)
        iptm = round(random.uniform(0.70, 0.86), 2)
        pae = round(random.uniform(3.2, 5.0), 1)
        
        confidence_data = {
            "ptm": ptm,
            "iptm": iptm,
            "chain_pair_pae_min": pae
        }
        
        with open(json_out, 'w', encoding='utf-8') as f_json:
            json.dump(confidence_data, f_json, indent=4)
            
        print(f"   [+] Simulated prediction generated: {basename} (ipTM={iptm}, pTM={ptm})")
        
    print("\n==============================================")
    print(f"[SUCCESS] Completed all {len(json_files)} AlphaFold 3 simulations!")
    print(f"   [+] Outputs stored in: {results_dir}")
    print("==============================================")

if __name__ == "__main__":
    run_mock_af3_campaign("outputs/af3_jobs", "outputs/af3_results")
