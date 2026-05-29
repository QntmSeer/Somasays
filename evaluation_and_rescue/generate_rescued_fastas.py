import os
import csv

def generate_fastas(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            seq = row["rescued_sequence"]
            out_fp = os.path.join(out_dir, f"{name}.fasta")
            with open(out_fp, 'w', encoding='utf-8') as out_f:
                out_f.write(f">{name}\n{seq}\n")
    print(f"[SUCCESS] Wrote rescued sequences to FASTA files in: {out_dir}")

if __name__ == "__main__":
    generate_fastas("outputs/rescue_report.csv", "outputs/rescued_candidates")
