import os
import glob

def extract_best_sequences(mpnn_seqs_dir: str, output_dir: str):
    """
    Parses ProteinMPNN FASTA output files, identifies the design with the lowest score (most stable),
    and writes it as a clean single-sequence FASTA file.
    """
    print("==============================================")
    print("  Somasays ProteinMPNN Sequence Extractor  ")
    print("==============================================")
    print(f"[*] Scanning ProteinMPNN output: {mpnn_seqs_dir}")
    
    fa_files = glob.glob(os.path.join(mpnn_seqs_dir, "*.fa")) + glob.glob(os.path.join(mpnn_seqs_dir, "*.fasta"))
    if not fa_files:
        print("[WARNING] No ProteinMPNN sequence files found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Found {len(fa_files)} files. Extracting best stability designs...\n")

    count = 0
    for fp in fa_files:
        try:
            basename = os.path.basename(fp)
            name, ext = os.path.splitext(basename)
            
            # Read all sequences and scores
            designs = []
            current_header = ""
            current_seq = ""
            
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_header and current_seq:
                            designs.append((current_header, current_seq))
                            current_seq = ""
                        current_header = line
                    else:
                        current_seq += line
                if current_header and current_seq:
                    designs.append((current_header, current_seq))

            if not designs:
                continue

            # Find the best design (excluding the original template if redesigned samples exist)
            best_seq = None
            best_score = 999.0
            best_header = ""

            # Check if we have designed samples (usually headers contain 'score=' or 'sample=')
            designed_samples = [d for d in designs if "sample=" in d[0]]
            
            if designed_samples:
                # Parse scores of the redesigned samples
                for header, seq in designed_samples:
                    # Header format: >T=0.1, sample=1, score=1.3206, global_score=1.3206, seq_recovery=0.4167
                    try:
                        score_part = [p for p in header.split(",") if "score=" in p and "global_score" not in p]
                        if score_part:
                            score_val = float(score_part[0].split("=")[1])
                            if score_val < best_score:
                                best_score = score_val
                                best_seq = seq
                                best_header = header
                    except Exception:
                        pass
            
            # Fallback to the first sequence (original) if no designed samples were parsed
            if not best_seq:
                best_header, best_seq = designs[0]
                best_score = 999.0 # template

            # Write clean FASTA file
            out_fp = os.path.join(output_dir, f"{name}.fasta")
            with open(out_fp, "w") as f:
                f.write(f">{name}_best_mpnn_score_{best_score}\n")
                f.write(f"{best_seq}\n")
                
            count += 1
        except Exception as e:
            print(f"[-] Failed to process {fp}: {e}")

    print("\n==============================================")
    print(f"[SUCCESS] Extracted {count} optimized sequences to: {output_dir}")
    print("==============================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract the best ProteinMPNN designs")
    parser.add_argument("--in_dir", type=str, default="outputs/mpnn_rescued/seqs", help="ProteinMPNN seqs directory")
    parser.add_argument("--out_dir", type=str, default="outputs/mpnn_best_sequences", help="Output FASTA directory")
    args = parser.parse_args()
    
    extract_best_sequences(args.in_dir, args.out_dir)
