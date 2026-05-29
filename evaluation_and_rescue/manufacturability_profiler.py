import os
import re
import csv
import glob
import argparse

# Kyte-Doolittle hydropathy values for GRAVY score calculation
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def calculate_gravy(sequence: str) -> float:
    """Calculates the Grand Average of Hydropathy (GRAVY) score."""
    if not sequence:
        return 0.0
    valid_residues = [c for c in sequence.upper() if c in KYTE_DOOLITTLE]
    if not valid_residues:
        return 0.0
    total = sum(KYTE_DOOLITTLE[r] for r in valid_residues)
    return round(total / len(valid_residues), 3)

def estimate_isoelectric_point(sequence: str) -> float:
    """Estimates the Isoelectric Point (pI) of the sequence using bisection search."""
    seq = sequence.upper()
    pka = {
        'K': 10.0, 'R': 12.0, 'H': 6.0,
        'D': 4.0, 'E': 4.4, 'C': 8.5, 'Y': 10.0
    }
    
    def get_net_charge(ph):
        # N-terminus positive charge
        charge = 1.0 / (1.0 + 10 ** (ph - 8.0))
        # C-terminus negative charge
        charge -= 1.0 / (1.0 + 10 ** (3.1 - ph))
        
        # Lys, Arg, His positive contributions
        for aa in ['K', 'R', 'H']:
            charge += seq.count(aa) * (1.0 / (1.0 + 10 ** (ph - pka[aa])))
            
        # Asp, Glu, Cys, Tyr negative contributions
        for aa in ['D', 'E', 'C', 'Y']:
            charge -= seq.count(aa) * (1.0 / (1.0 + 10 ** (pka[aa] - ph)))
            
        return charge

    # Bisection search between pH 0 and 14
    low, high = 0.0, 14.0
    for _ in range(20):
        mid = (low + high) / 2.0
        val = get_net_charge(mid)
        if val > 0:
            low = mid
        else:
            high = mid
    return round((low + high) / 2.0, 2)

def detect_hydrophobic_patches(sequence: str, min_len: int = 5) -> list:
    """Detects contiguous hydrophobic runs (V, I, L, F, W, Y, M) of at least min_len."""
    seq = sequence.upper()
    # Hydrophobic amino acids
    pattern = r'[VILFWYM]{' + str(min_len) + r',}'
    matches = []
    for match in re.finditer(pattern, seq):
        matches.append((match.start() + 1, match.group()))
    return matches

def scan_ptm_and_instability(sequence: str) -> dict:
    """Scans for chemical degradation hotspots and PTM sites."""
    seq = sequence.upper()
    results = {
        "n_glyco": [],     # N-X-[S/T] where X is not P
        "deamidation": [],  # NG or NS
        "oxidation": [],    # Methionine and Cysteine occurrences
        "acid_cleavage": [] # DP
    }
    
    # 1. N-Glycosylation: N followed by any char except P, followed by S or T
    glyco_pattern = re.compile(r'N[^P][ST]')
    for match in glyco_pattern.finditer(seq):
        results["n_glyco"].append((match.start() + 1, match.group()))
        
    # 2. Deamidation: NG or NS
    deam_pattern = re.compile(r'N[GS]')
    for match in deam_pattern.finditer(seq):
        results["deamidation"].append((match.start() + 1, match.group()))
        
    # 3. Acid Cleavage: DP (Aspartate-Proline)
    cleavage_pattern = re.compile(r'DP')
    for match in cleavage_pattern.finditer(seq):
        results["acid_cleavage"].append((match.start() + 1, match.group()))
        
    # 4. Oxidation: Count M (Methionine) and C (Cysteine)
    for idx, char in enumerate(seq, 1):
        if char == 'M':
            results["oxidation"].append((idx, "M"))
        elif char == 'C':
            results["oxidation"].append((idx, "C"))
            
    return results

def parse_fasta(fasta_path: str) -> tuple:
    """Helper to parse candidate FASTA headers and sequences."""
    header = ""
    sequence = ""
    with open(fasta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(">"):
                header = line.strip()
            else:
                sequence += line.strip()
    return header, sequence

def calculate_solubility_ratio(sequence: str) -> float:
    """Calculates the charge-to-hydrophobic ratio.
    Charged residues: D, E, K, R
    Hydrophobic residues: V, I, L, F, W, Y, M
    """
    seq = sequence.upper()
    charged = sum(seq.count(aa) for aa in ['D', 'E', 'K', 'R'])
    hydrophobic = sum(seq.count(aa) for aa in ['V', 'I', 'L', 'F', 'W', 'Y', 'M'])
    return round(charged / max(1, hydrophobic), 3)

def run_profiler(in_dir: str, out_dir: str, tag_n: str = "", tag_c: str = ""):
    """Runs the manufacturability analysis on all candidates in the input directory."""
    print("==============================================")
    print("  Somasays Wet-Lab Manufacturability Profiler")
    print("==============================================")
    if tag_n:
        print(f"[*] Simulating N-terminal expression tag: {tag_n}")
    if tag_c:
        print(f"[*] Simulating C-terminal purification tag: {tag_c}")
    
    fasta_files = glob.glob(os.path.join(in_dir, "*.fasta"))
    if not fasta_files:
        print(f"[WARNING] No FASTA files found in {in_dir}.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    report_csv = os.path.join(out_dir, "manufacturability_report.csv")
    summary_md = os.path.join(out_dir, "manufacturability_summary.md")
    
    records = []
    
    for fp in sorted(fasta_files):
        name = os.path.basename(fp).replace(".fasta", "")
        header, core_seq = parse_fasta(fp)
        if not core_seq:
            continue
            
        # Simulate expressed construct
        seq = tag_n + core_seq + tag_c
            
        # Run diagnostics
        gravy = calculate_gravy(seq)
        pi = estimate_isoelectric_point(seq)
        patches = detect_hydrophobic_patches(seq)
        ptms = scan_ptm_and_instability(core_seq) # Scans core protein sequence for degradation
        sol_ratio = calculate_solubility_ratio(seq)
        
        # Calculate penalties for composite scoring
        # Base stability starts at 10.0
        manufacturability_score = 10.0
        
        # Hydropathy penalties: positive GRAVY indicates hydrophobic patch risk
        if gravy > 0.0:
            manufacturability_score -= (gravy * 2.0)
            
        # Long hydrophobic run penalty
        manufacturability_score -= len(patches) * 1.5
        
        # PTM penalties
        manufacturability_score -= len(ptms["n_glyco"]) * 2.0
        manufacturability_score -= len(ptms["deamidation"]) * 1.0
        manufacturability_score -= len(ptms["acid_cleavage"]) * 1.5
        manufacturability_score -= len([p for p in ptms["oxidation"] if p[1] == 'M']) * 0.5
        
        # Solubility ratio penalty (if charge-to-hydrophobic ratio < 0.8)
        if sol_ratio < 0.8:
            manufacturability_score -= 1.5
            
        # Net charge at pH 7.4 penalty (pI close to 7.4 has higher aggregation risk)
        pi_diff = abs(pi - 7.4)
        if pi_diff < 1.0:
            # high risk region
            manufacturability_score -= (1.0 - pi_diff) * 2.0
            
        manufacturability_score = max(0.0, round(manufacturability_score, 2))
        
        records.append({
            "name": name,
            "length": len(seq),
            "gravy": gravy,
            "pi": pi,
            "hydrophobic_patches": len(patches),
            "n_glyco_count": len(ptms["n_glyco"]),
            "deamidation_count": len(ptms["deamidation"]),
            "oxidation_count": len(ptms["oxidation"]),
            "acid_cleavage_count": len(ptms["acid_cleavage"]),
            "solubility_ratio": sol_ratio,
            "score": manufacturability_score,
            "sequence": seq,
            "details": f"Glyco: {ptms['n_glyco']}, Deamid: {ptms['deamidation']}, Acid: {ptms['acid_cleavage']}, Patches: {patches}"
        })
        
    # Sort candidates by score
    records.sort(key=lambda x: x["score"], reverse=True)
    
    # Write CSV
    with open(report_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "length", "gravy", "pi", "hydrophobic_patches", 
            "n_glyco_count", "deamidation_count", "oxidation_count", 
            "acid_cleavage_count", "solubility_ratio", "score", "sequence", "details"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
            
    # Write Markdown Summary
    with open(summary_md, 'w', encoding='utf-8') as f:
        f.write("# Wet-Lab Manufacturability Leaderboard\n\n")
        f.write("This summary ranks candidates by their predicted biophysical stability and expression success.\n")
        f.write("Scoring starts at `10.0` and is penalized for glycosylation motifs, deamidation hotspots, acid-cleavage bonds, low charge-to-hydrophobic ratios, high hydropathy (GRAVY), and charge neutrality near pH 7.4.\n\n")
        
        f.write("## 🏆 Top 10 Safest Candidates (Lowest Manufacturing Risk)\n\n")
        f.write("| Rank | Candidate | Length (aa) | GRAVY | pI | Solubility Ratio | Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for idx, r in enumerate(records[:10], 1):
            f.write(f"| {idx} | `{r['name']}` | {r['length']} | {r['gravy']} | {r['pi']} | {r['solubility_ratio']} | **{r['score']}** |\n")
            
        f.write("\n## ⚠️ Top 5 High-Risk Candidates (Expression/Aggregation Warnings)\n\n")
        f.write("| Rank | Candidate | Length (aa) | GRAVY | pI | Sol. Ratio | Score | Warnings |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for idx, r in enumerate(reversed(records[-5:]), 1):
            warnings = []
            if r["gravy"] > 0.5: warnings.append("Highly Hydrophobic")
            if abs(r["pi"] - 7.4) < 0.5: warnings.append("Neutral at pH 7.4 (Aggregation)")
            if r["n_glyco_count"] > 0: warnings.append("PTM Glycosylation Risk")
            if r["solubility_ratio"] < 0.8: warnings.append("Inclusion Body / Low Solubility")
            f.write(f"| {idx} | `{r['name']}` | {r['length']} | {r['gravy']} | {r['pi']} | {r['solubility_ratio']} | **{r['score']}** | {', '.join(warnings) if warnings else 'None'} |\n")
            
    print(f"[SUCCESS] Scanned {len(records)} candidates!")
    print(f"   [+] CSV Report: {report_csv}")
    print(f"   [+] Markdown Summary: {summary_md}")
    print("==============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate de novo proteins for expression and degradation risks")
    parser.add_argument("--in_dir", type=str, default="../outputs/mpnn_best_sequences", help="Input directory of sequence FASTA files")
    parser.add_argument("--out_dir", type=str, default="../outputs", help="Output directory for reports")
    parser.add_argument("--tag_n", type=str, default="", help="N-terminal tag to prepend (e.g. HHHHHH)")
    parser.add_argument("--tag_c", type=str, default="", help="C-terminal tag to append")
    args = parser.parse_args()
    
    run_profiler(args.in_dir, args.out_dir, tag_n=args.tag_n, tag_c=args.tag_c)
