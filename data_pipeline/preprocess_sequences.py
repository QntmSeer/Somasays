import json
import os

def clean_sequence(seq: str) -> str:
    """Ensures sequence contains only valid standard amino acids."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    return ''.join([c for c in seq.upper() if c in valid_aa])

def preprocess_fasta(input_path: str, output_path: str):
    """
    Reads raw FASTA data from UniProt, cleans sequences, and formats as JSONL for ESM3.
    """
    print(f"Preprocessing data from {input_path}...")
    
    sequences = []
    current_seq = ""
    current_header = ""
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append({"header": current_header, "sequence": clean_sequence(current_seq)})
                current_header = line[1:]
                current_seq = ""
            else:
                current_seq += line
        if current_seq: # Catch the last sequence
            sequences.append({"header": current_header, "sequence": clean_sequence(current_seq)})
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for entry in sequences:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Processed {len(sequences)} sequences. Saved to {output_path}")

if __name__ == "__main__":
    preprocess_fasta(
        input_path="../data/raw/uniprot_global_medicine.fasta",
        output_path="../data/processed/somasays_dataset.jsonl"
    )
