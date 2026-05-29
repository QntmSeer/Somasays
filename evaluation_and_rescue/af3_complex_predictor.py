import json
import os
import argparse
import glob

# Production Targets Database (supporting single targets and multi-chain complexes like antibody Fab)
TARGETS = {
    "COVID_SPIKE_RBD": {
        "type": "single",
        "seqs": ["RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST"]
    },
    "HANTAVIRUS_GN": {
        "type": "single",
        "seqs": ["SLGTLVLLCSHLTLVQGQGKSIVDPTDGFVTSSQSLIVTRATPGTPNLIHIECSTGLLTAHCKSTQQINSKQLLGCAFCGGTLNVSPEFGDTVSTKCK"]
    },
    "INFLUENZA_HA": {
        "type": "single",
        "seqs": ["DTICIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDKHNGKLCKLRGVAPLHLGKCNIAGWILGNPECESLSTASSWSYIVETPSSDNGTCYPGDFIDYEELREQLSSVSSFERFEIFPKTSSWPNHDSNKGVTAACPHAGAKSFYKNLIWLVKKGNSYPKLSKSYINDKGKEVLVLWGIHHPSTSADQQSLYQNADAYVFVGSSRYSKKFKPEIAIRPKVRDQEGRMNYYWTLVEPGDKITFEATGNLVVPRYAFAMERNAGSGIIISDTPVHDCNTTCQTPKGAINTSLPFQNIHPITIGKCPKYVKSTKLRLATGLRNVPSIQSR"]
    },
    "HTN_GN1_FAB": {
        "type": "antibody",
        "seqs": [
            "TGQSLVESGGDLVKPEGSLTLTCTASGFSFSSTHWICWVRQAPGKGLEWIACIYVGNTYDSYYANWAKGRFTISKTSSTTVTLQMTTLTAADTATYFCARSGSVFGVVSLWGPGTLVTVSSGQPKAPSVFPLAPCCGDTPSSTVTLGCLVKGYLPEPVTVTWNSGTLTNGVRTFPSVRQSSGLYSLSSVVSVTSSSQPVTCNVAHPATNTKVDKTVAPSTCSGTKHHHHHH", # Fab Heavy Chain
            "TGQVLTQTPASVSEPVEGTVTIKCQASQSINNWLSWYQQRPGQPPKLLIYDASTVASGVSSRFKGSGSGTEFTLTISDLECADAATYACQSYGYGISITDNSAFGGGTEVVVRGDPVAPSVLIFPPAADQVATGTVTIVCVANKYFPDVTVTWEVDGTTQTTGIENSKTPQNSADCTYNLSSTLTLTSTQYNSHKEYTCKVTQGTTSVVQSFNRGDC" # Fab Light Chain
        ]
    }
}

def parse_fasta(fasta_path: str) -> tuple:
    """
    Reads a simple FASTA file and returns the ID and sequence.
    """
    seq_id = os.path.basename(fasta_path).replace(".fasta", "")
    sequence = ""
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            sequence += line.strip()
    return seq_id, sequence

def format_for_alphafold3(binder_seq: str, target_seqs: list, job_name: str) -> dict:
    """
    Formulates the strict JSON schema required for submission to the AlphaFold 3 Server.
    Supports single targets or multi-chain complexes (like antibody Fab heavy + light chains).
    """
    sequences = [
        {
            "proteinChain": {
                "sequence": binder_seq,
                "count": 1
            }
        }
    ]
    for seq in target_seqs:
        sequences.append({
            "proteinChain": {
                "sequence": seq,
                "count": 1
            }
        })
    return {
        "name": job_name,
        "modelSeeds": [1], # AlphaFold 3 Server limits to a maximum of 1 seed per job
        "sequences": sequences
    }


def batch_generate_af3_jobs(candidates_dir: str, target_name: str, output_dir: str):
    """
    Scans the generator outputs directory for FASTA files and generates valid AF3 JSON requests.
    """
    print("==============================================")
    print("  Somasays AlphaFold 3 Job Packager  ")
    print("==============================================")

    target_entry = TARGETS.get(target_name)
    if not target_entry:
        print(f"[ERROR] Target '{target_name}' not found in Database.")
        print(f"Available targets: {list(TARGETS.keys())}")
        return

    target_seqs = target_entry["seqs"]
    print(f"[*] Target Active: {target_name} ({target_entry['type']} type)")
    print(f"[*] Scanning source directory: {candidates_dir}")

    # Pick up all generated FASTA files
    fasta_files = glob.glob(os.path.join(candidates_dir, "*.fasta"))
    
    if not fasta_files:
        print(f"[WARNING] No FASTA candidates found in {candidates_dir}.")
        print("Run your candidate generation script first!")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Discovered {len(fasta_files)} candidates. Compiling AF3 schemas...\n")

    count = 0
    for fp in fasta_files:
        try:
            seq_id, sequence = parse_fasta(fp)
            if not sequence:
                continue
                
            job_name = f"somasays_{seq_id}_vs_{target_name}"
            af3_json = format_for_alphafold3(sequence, target_seqs, job_name)
            
            out_path = os.path.join(output_dir, f"{job_name}.json")
            with open(out_path, 'w') as f:
                json.dump(af3_json, f, indent=4)
                
            print(f"   [+] Job Formulated: {out_path}")
            count += 1
        except Exception as e:
            print(f"   [-] Failed to process {fp}: {e}")

    print("\n==============================================")
    print(f"[SUCCESS] Packaged {count} AlphaFold 3 job files!")
    print(f"Destination: {output_dir}")
    print("Ready for upload directly to the AlphaFold 3 Server portal!")
    print("==============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile AlphaFold 3 JSON Submission Files")
    parser.add_argument(
        "--in_dir", 
        type=str, 
        default="../generation_engine/outputs/multimodal_candidates", 
        help="Folder containing candidate FASTA files"
    )
    parser.add_argument(
        "--target", 
        type=str, 
        default="COVID_SPIKE_RBD", 
        choices=list(TARGETS.keys()),
        help="Which viral/antibody target to dock against"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="af3_jobs/", 
        help="Destination directory for AF3 JSONs"
    )

    args = parser.parse_args()
    batch_generate_af3_jobs(args.in_dir, args.target, args.out_dir)
