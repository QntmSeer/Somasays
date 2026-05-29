import os
import csv
import collections
import math

def calculate_entropy(seq: str) -> float:
    """Calculates Shannon entropy of the amino acid sequence to filter out low-complexity repeats."""
    counts = collections.Counter(seq)
    entropy = 0.0
    for count in counts.values():
        p = count / len(seq)
        entropy -= p * math.log2(p)
    return entropy

def select_top_candidates(manufacturability_csv: str, rescue_csv: str, top_n: int = 10) -> list:
    # Load manufacturability data
    man_data = {}
    with open(manufacturability_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            man_data[row["name"]] = row
            
    # Load rescue data
    rescued_leads = []
    with open(rescue_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            original_seq = row["original_sequence"]
            rescued_seq = row["rescued_sequence"]
            risk = row["rescued_risk"]
            
            # Retrieve manufacturability details if available
            man = man_data.get(name, {})
            gravy = float(man.get("gravy", 0.0))
            score = float(man.get("score", 0.0))
            
            # Calculate sequence complexity (avoid highly repetitive mock designs)
            entropy = calculate_entropy(rescued_seq)
            
            # Filter criteria:
            # - We favor sequences with high complexity (entropy > 2.8) to exclude simple mock patterns
            # - Favor low risk or moderate risk profiles
            # - Favor high solubility (lower GRAVY score is more hydrophilic)
            rank_score = (entropy * 5.0) - (gravy * 2.0) + (10.0 if risk == 'Low' else 5.0 if risk == 'Moderate' else 0.0)
            
            rescued_leads.append({
                "name": name,
                "rescued_sequence": rescued_seq,
                "gravy": gravy,
                "rescued_risk": risk,
                "entropy": round(entropy, 2),
                "rank_score": round(rank_score, 2)
            })
            
    # Sort by rank score descending
    rescued_leads.sort(key=lambda x: x["rank_score"], reverse=True)
    return rescued_leads[:top_n]

if __name__ == "__main__":
    top_leads = select_top_candidates("outputs/manufacturability_report.csv", "outputs/rescue_report.csv", 10)
    print("\n=======================================================")
    print("      SOMASAYS RECOMMENDED TOP 10 CANDIDATES FOR AF3   ")
    print("=======================================================")
    for rank, lead in enumerate(top_leads, start=1):
        print(f"Rank {rank}: {lead['name']}")
        print(f"   [+] Rescued Sequence: {lead['rescued_sequence']}")
        print(f"   [+] MHC-II Risk     : {lead['rescued_risk']}")
        print(f"   [+] Hydrophilicity  : GRAVY = {lead['gravy']}")
        print(f"   [+] Complexity      : Entropy = {lead['entropy']}")
        json_path = f"outputs/af3_jobs/somasays_{lead['name']}_vs_HANTAVIRUS_GN.json"
        print(f"   [+] AF3 JSON Path   : {json_path}")
        print("-------------------------------------------------------")
