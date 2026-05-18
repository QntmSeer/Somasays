import os
import argparse
import glob
import subprocess

def format_mpnn_cli_call(pdb_dir: str, output_dir: str, target_chain: str = "A", temp: float = 0.1) -> str:
    """
    Formulates the exact shell command needed to execute ProteinMPNN on an HPC cluster.
    """
    # Build a standardized ProteinMPNN execution instruction
    cmd = (
        f"python /path/to/ProteinMPNN/protein_mpnn_run.py \\\n"
        f"  --pdb_path_dir {pdb_dir} \\\n"
        f"  --output_dir {output_dir} \\\n"
        f"  --chains_to_design {target_chain} \\\n"
        f"  --sampling_temp {temp} \\\n"
        f"  --num_seq_per_target 8 \\\n"
        f"  --save_score 1"
    )
    return cmd

def automated_stability_rescue(candidates_dir: str, out_dir: str, design_chain: str = "A"):
    """
    Iterates over candidate designs and compiles stability rescue pipelines.
    """
    print("================================================")
    print("  Somasays ProteinMPNN Stability Rescue  ")
    print("================================================")
    print(f"[*] Scanning source 3D structures: {candidates_dir}")

    pdb_files = glob.glob(os.path.join(candidates_dir, "*.pdb"))

    if not pdb_files:
        print(f"[WARNING] No candidate PDB files found in {candidates_dir}.")
        print("Generate your 3D multimodal structures first!")
        return

    print(f"[*] Detected {len(pdb_files)} generated backbones.")
    os.makedirs(out_dir, exist_ok=True)

    print("\n================================================")
    print("🚀 HIGH PERFORMANCE CLUSTER (HPC) DEPLOYMENT")
    print("================================================")
    print("Copy-paste the following command directly into your terminal")
    print("to run massive inverse-folding on your GPU cluster:")
    print("------------------------------------------------")
    
    # Absoluting paths for the print-out to ensure cluster execution success
    abs_in = os.path.abspath(candidates_dir)
    abs_out = os.path.abspath(out_dir)
    cluster_command = format_mpnn_cli_call(abs_in, abs_out, design_chain)
    print(cluster_command)
    print("------------------------------------------------")

    # Write the shell command to a local helper script so the user can just './run_mpnn.sh' it!
    helper_script_path = os.path.join(out_dir, "execute_cluster_mpnn.sh")
    with open(helper_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Somasays Generated Inverse Folding Script\n")
        f.write("echo 'Starting ProteinMPNN Stability Campaign...'\n")
        f.write(cluster_command + "\n")
    
    os.chmod(helper_script_path, 0o755)
    print(f"\n[+] Saved executable cluster launcher to: {helper_script_path}")

    # Local Pilot Sandbox execution (creates dummy output files to test subsequent steps)
    print("\n[PILOT MODE] Synchronizing simulation pipeline outputs locally...")
    for pdb in pdb_files:
        basename = os.path.basename(pdb).replace(".pdb", "")
        out_fasta = os.path.join(out_dir, f"{basename}_rescued_sequences.fasta")
        
        # Simulated highly stable rescored sequence for pilot sandbox
        # ProteinMPNN preserves 3D backbone but selects highly expressive/stable amino acid patterns
        stable_sequence_mock = "MKALFVALAALASTAFAQPSTVWR" 
        
        with open(out_fasta, "w") as f:
            f.write(f">{basename}_mpnn_design_1_score=0.05\n")
            f.write(f"{stable_sequence_mock}\n")
        print(f"   -> Local simulation complete for: {basename} (Mock FASTA created)")

    print("\n================================================")
    print(f"[SUCCESS] Stability Rescue pipeline staged for {len(pdb_files)} candidates!")
    print(f"Checkpoint results directed to: {out_dir}")
    print("================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProteinMPNN Automation Module")
    parser.add_argument(
        "--in_dir", 
        type=str, 
        default="../generation_engine/outputs/multimodal_candidates", 
        help="Folder containing generated 3D PDB files"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="mpnn_rescued/", 
        help="Destination folder for rescued stability FASTA sequences"
    )
    parser.add_argument(
        "--chain", 
        type=str, 
        default="A", 
        help="Target structural chain to redesign"
    )

    args = parser.parse_args()
    automated_stability_rescue(args.in_dir, args.out_dir, args.chain)
