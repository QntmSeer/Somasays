import os
import json
import csv
import collections
import math

def calculate_entropy(seq: str) -> float:
    counts = collections.Counter(seq)
    entropy = 0.0
    for count in counts.values():
        p = count / len(seq)
        entropy -= p * math.log2(p)
    return round(entropy, 2)
    
def calculate_gravy(seq: str) -> float:
    kd_scale = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    score = sum(kd_scale.get(aa, 0.0) for aa in seq)
    return round(score / len(seq), 3)

def generate_bulk_files():
    rescue_report_path = "outputs/rescue_report.csv"
    out_dir = "outputs"
    
    # 1. Parse and rank candidates
    candidates = []
    with open(rescue_report_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            seq = row["rescued_sequence"]
            mhc_risk = row["rescued_risk"]
            
            risk_score = 0 if mhc_risk == "Low" else (1 if mhc_risk == "Moderate" else 2)
            entropy = calculate_entropy(seq)
            gravy = calculate_gravy(seq)
            
            score = risk_score * 10.0 + gravy - (entropy * 0.5)
            candidates.append({
                "name": name,
                "score": score,
                "seq": seq,
                "risk": mhc_risk
            })
            
    # Sort candidates (lowest score is best)
    candidates.sort(key=lambda x: x["score"])
    top_30 = candidates[:30]
    
    # Target Sequences
    target_c_to_s = "SLGTLVLLSSHLTLVQGQGKSIVDPTDGFVTSSQSLIVTRATPGTPNLIHIESSTGLLTAHSKSTQQINSKQLLGSAFSGGTLNVSPEFGDTVSTKSK"
    target_full = "IYELKMECPHTVGLGQGYIIGSTELGLISIEAASDIKLESSCNFDLHTTSMAQKSFTQVEWRKKSDTTDTTNAASTTFEAQTKTVNLRGTCILAPELYDTVKKTVLCYDLTCNQTHCQPTVYLIAPVLTCMSIRSCMASVFTSRIQVIYEKTHCVTGQLIEGQCFNPAHTLTLSQPAHTYDTVTLPISCFFTPKKSEQLKVIKTFEGILTKTGCTENALQGYYVCFLGSHSEPLIVPSLEDIRSAEVVSRMLVHPRGEDHDAIQNSQSHLRIVGPITAKVPSTSSTDTLKGTAFAGVPMYSSLSTLVRNADPEFVFSPGIVPESNHSTCDKKTVPITWTGYLPISGEME"
    
    # Generate job structures
    jobs_c_to_s = []
    jobs_full = []
    
    for cand in top_30:
        # C-to-S Mutant Target Job
        job_c_to_s = {
            "name": f"somasays_{cand['name']}_vs_HANTAVIRUS_GN_C_to_S_mutant",
            "modelSeeds": [1],
            "sequences": [
                {"proteinChain": {"sequence": cand["seq"], "count": 1}},
                {"proteinChain": {"sequence": target_c_to_s, "count": 1}}
            ]
        }
        jobs_c_to_s.append(job_c_to_s)
        
        # Full-length Target Job
        job_full = {
            "name": f"somasays_{cand['name']}_vs_HANTAVIRUS_GN_full_length",
            "modelSeeds": [1],
            "sequences": [
                {"proteinChain": {"sequence": cand["seq"], "count": 1}},
                {"proteinChain": {"sequence": target_full, "count": 1}}
            ]
        }
        jobs_full.append(job_full)
        
    # Write output files
    c_to_s_path = os.path.join(out_dir, "combined_top_30_af3_jobs_C_to_S_mutant.json")
    full_path = os.path.join(out_dir, "combined_top_30_af3_jobs_full_length.json")
    
    with open(c_to_s_path, 'w', encoding='utf-8') as f:
        json.dump(jobs_c_to_s, f, indent=4)
        
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(jobs_full, f, indent=4)
        
    print("=======================================================")
    print("[SUCCESS] Wrote combined fixed Top 30 jobs files!")
    print(f"   -> C-to-S mutant: {c_to_s_path}")
    print(f"   -> Full-length: {full_path}")
    print("=======================================================")

if __name__ == "__main__":
    generate_bulk_files()
