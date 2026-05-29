import os
import json
import csv

def combine_top_jobs(rescue_report_path: str, jobs_dir: str, out_dir: str, top_n_values=[10, 20, 30]):
    print("=======================================================")
    print("[*] Combining AF3 Job JSONs into Bulk Upload Files...")
    print("=======================================================")

    # 1. Parse and rank candidates (matching select_top_leads.py logic)
    import collections
    def calculate_entropy(seq: str) -> float:
        counts = collections.Counter(seq)
        entropy = 0.0
        for count in counts.values():
            p = count / len(seq)
            entropy -= p * (p / 2.0) # Using simple Shannon base-2 approximation or similar
        return round(-sum((c/len(seq)) * (math.log2(c/len(seq)) if c>0 else 0) for c in counts.values()), 2)
        
    def calculate_gravy(seq: str) -> float:
        kd_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        score = sum(kd_scale.get(aa, 0.0) for aa in seq)
        return round(score / len(seq), 3)

    candidates = []
    with open(rescue_report_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            seq = row["rescued_sequence"]
            mhc_risk = row["rescued_risk"]
            
            # Skip candidates with high immunogenicity risk if possible
            risk_score = 0 if mhc_risk == "Low" else (1 if mhc_risk == "Moderate" else 2)
            
            import math
            entropy = calculate_entropy(seq)
            gravy = calculate_gravy(seq)
            
            # Rank formula:
            # - Lower risk is better
            # - Lower gravy (more hydrophilic) is better for manufacturability
            # - Higher complexity (entropy) is better to avoid repetitive junk
            score = risk_score * 10.0 + gravy - (entropy * 0.5)
            candidates.append({
                "name": name,
                "score": score,
                "seq": seq,
                "gravy": gravy,
                "entropy": entropy,
                "risk": mhc_risk
            })
            
    # Sort candidates (lowest score is best)
    candidates.sort(key=lambda x: x["score"])
    print(f"[+] Total ranked candidates: {len(candidates)}")

    os.makedirs(out_dir, exist_ok=True)

    # 2. For each target count, combine jobs
    for n in top_n_values:
        n_selected = min(n, len(candidates))
        selected_candidates = candidates[:n_selected]
        
        combined_jobs = []
        missing_count = 0
        for cand in selected_candidates:
            job_file = os.path.join(jobs_dir, f"somasays_{cand['name']}_vs_HANTAVIRUS_GN.json")
            if os.path.exists(job_file):
                with open(job_file, 'r', encoding='utf-8') as jf:
                    job_data = json.load(jf)
                    combined_jobs.append(job_data)
            else:
                missing_count += 1
                
        if combined_jobs:
            out_file = os.path.join(out_dir, f"combined_top_{n}_af3_jobs.json")
            with open(out_file, 'w', encoding='utf-8') as out_f:
                json.dump(combined_jobs, out_f, indent=4)
            print(f"[SUCCESS] Wrote combined Top {n} AF3 jobs ({len(combined_jobs)} jobs) to:")
            print(f"   -> {out_file}")
            if missing_count > 0:
                print(f"   [WARNING] {missing_count} job JSON files were missing in the directory.")
        else:
            print(f"[WARNING] No jobs found to combine for Top {n}.")
            
    print("=======================================================")

if __name__ == "__main__":
    combine_top_jobs(
        rescue_report_path="outputs/rescue_report.csv",
        jobs_dir="outputs/af3_jobs",
        out_dir="outputs"
    )
