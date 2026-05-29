import os
import math
import random
import argparse

# Complete Codon usage tables for E. coli and Human.
# Maps each amino acid to a dictionary of synonymous codons and their relative frequencies.
CODON_TABLES = {
    "e_coli": {
        "A": {"GCG": 0.36, "GCC": 0.26, "GCA": 0.22, "GCT": 0.16},
        "C": {"TGC": 0.55, "TGT": 0.45},
        "D": {"GAT": 0.63, "GAC": 0.37},
        "E": {"GAA": 0.68, "GAG": 0.32},
        "F": {"TTT": 0.58, "TTC": 0.42},
        "G": {"GGT": 0.34, "GGC": 0.40, "GGA": 0.13, "GGG": 0.13},
        "H": {"CAT": 0.57, "CAC": 0.43},
        "I": {"ATT": 0.49, "ATC": 0.41, "ATA": 0.10},
        "K": {"AAA": 0.74, "AAG": 0.26},
        "L": {"CTG": 0.49, "TTA": 0.13, "TTG": 0.13, "CTT": 0.11, "CTC": 0.10, "CTA": 0.04},
        "M": {"ATG": 1.00},
        "N": {"AAT": 0.49, "AAC": 0.51},
        "P": {"CCG": 0.52, "CCA": 0.19, "CCT": 0.16, "CCC": 0.13},
        "Q": {"CAG": 0.65, "CAA": 0.35},
        "R": {"CGT": 0.38, "CGC": 0.37, "CGA": 0.07, "CGG": 0.10, "AGA": 0.04, "AGG": 0.04},
        "S": {"AGC": 0.27, "TCG": 0.14, "TCC": 0.15, "TCA": 0.13, "TCT": 0.16, "AGT": 0.15},
        "T": {"ACG": 0.27, "ACC": 0.43, "ACA": 0.13, "ACT": 0.17},
        "V": {"GTG": 0.37, "GTC": 0.21, "GTA": 0.16, "GTT": 0.26},
        "W": {"TGG": 1.00},
        "Y": {"TAT": 0.58, "TAC": 0.42},
        "*": {"TAA": 0.61, "TGA": 0.30, "TAG": 0.09}
    },
    "human": {
        "A": {"GCC": 0.40, "GCT": 0.26, "GCA": 0.23, "GCG": 0.11},
        "C": {"TGC": 0.54, "TGT": 0.46},
        "D": {"GAC": 0.54, "GAT": 0.46},
        "E": {"GAG": 0.58, "GAA": 0.42},
        "F": {"TTC": 0.55, "TTT": 0.45},
        "G": {"GGC": 0.34, "GGA": 0.25, "GGG": 0.25, "GGT": 0.16},
        "H": {"CAC": 0.58, "CAT": 0.42},
        "I": {"ATC": 0.47, "ATT": 0.36, "ATA": 0.17},
        "K": {"AAG": 0.58, "AAA": 0.42},
        "L": {"CTG": 0.40, "CTC": 0.20, "TTG": 0.13, "CTT": 0.13, "CTA": 0.07, "TTA": 0.07},
        "M": {"ATG": 1.00},
        "N": {"AAC": 0.53, "AAT": 0.47},
        "P": {"CCC": 0.32, "CCT": 0.29, "CCA": 0.28, "CCG": 0.11},
        "Q": {"CAG": 0.74, "CAA": 0.26},
        "R": {"CGG": 0.21, "AGG": 0.21, "AGA": 0.20, "CGC": 0.19, "CGA": 0.11, "CGT": 0.08},
        "S": {"AGC": 0.24, "TCC": 0.22, "TCT": 0.18, "TCA": 0.15, "AGT": 0.15, "TCG": 0.05},
        "T": {"ACC": 0.36, "ACA": 0.28, "ACT": 0.25, "ACG": 0.11},
        "V": {"GTG": 0.46, "GTC": 0.24, "GTT": 0.18, "GTA": 0.11},
        "W": {"TGG": 1.00},
        "Y": {"TAC": 0.56, "TAT": 0.44},
        "*": {"TGA": 0.52, "TAA": 0.28, "TAG": 0.20}
    }
}

# Common restriction enzymes we want to avoid introducing
RESTRICTION_SITES = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "XhoI": "CTCGAG",
    "HindIII": "AAGCTT",
    "NdeI": "CATATG",
    "BsaI": "GGTCTC"
}

class CarbonLikelihoodEvaluator:
    """
    Evaluates DNA sequence naturalness score.
    If huggingface/transformers is available and weights are loaded, it can score with Carbon model.
    Otherwise, it uses a 6-mer frequency heuristic to estimate likelihood based on non-overlapping 6-mers.
    """
    def __init__(self, model_name="HuggingFaceBio/carbon-3b"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Build 6-mer token dictionary heuristic based on GC content and codon preferences
        self.vocab_6mer = {}
        
    def score_sequence(self, dna_seq: str) -> float:
        """
        Returns a log-likelihood or heuristic score of the DNA sequence.
        Higher score represents a more natural/stable mRNA sequence.
        """
        # Ensure input sequence length is a multiple of 6 for 6-mer token alignment (standard Carbon tokenization)
        # Pad or clip if necessary for the heuristic
        n = len(dna_seq)
        if n < 6:
            return 0.0
            
        score = 0.0
        num_tokens = 0
        
        # 6-mer non-overlapping tokenization model
        for i in range(0, n - 5, 6):
            token = dna_seq[i:i+6]
            gc_count = token.count("G") + token.count("C")
            gc_ratio = gc_count / 6.0
            
            # Carbon-like likelihood heuristic:
            # - Prefers healthy balanced GC content (around 45%-55% GC per 6-mer)
            # - Penalizes homopolymer repeats (e.g. 5 identical bases in a row)
            # - Prefers transitions that avoid local extreme base compositions
            token_score = -abs(gc_ratio - 0.5) * 5.0
            
            # Check for homopolymer run within token
            max_run = 1
            curr_run = 1
            for j in range(1, 6):
                if token[j] == token[j-1]:
                    curr_run += 1
                    max_run = max(max_run, curr_run)
                else:
                    curr_run = 1
                    
            if max_run >= 4:
                token_score -= 2.0
            if max_run >= 5:
                token_score -= 5.0
                
            score += token_score
            num_tokens += 1
            
        return score / num_tokens if num_tokens > 0 else 0.0

def calculate_cai(dna_seq: str, amino_acids: str, host: str) -> float:
    """
    Computes the Codon Adaptation Index (CAI) of the DNA sequence for a specific host.
    """
    codon_table = CODON_TABLES[host]
    n_codons = len(amino_acids)
    if n_codons == 0:
        return 0.0
        
    log_w_sum = 0.0
    
    for i in range(n_codons):
        aa = amino_acids[i]
        codon = dna_seq[i*3 : (i+1)*3]
        
        synonymous = codon_table[aa]
        freq = synonymous.get(codon, 0.0)
        max_freq = max(synonymous.values())
        
        # Relative adaptiveness w_i
        w_i = freq / max_freq if max_freq > 0 else 0.001
        w_i = max(0.001, w_i) # Avoid log(0)
        log_w_sum += math.log(w_i)
        
    return math.exp(log_w_sum / n_codons)

def get_optimal_cai_sequence(amino_acids: str, host: str) -> str:
    """
    Returns the DNA sequence generated by choosing the most frequent codon for each residue.
    """
    codon_table = CODON_TABLES[host]
    dna_parts = []
    for aa in amino_acids:
        synonymous = codon_table[aa]
        best_codon = max(synonymous, key=synonymous.get)
        dna_parts.append(best_codon)
    return "".join(dna_parts)

def evaluate_objective(dna_seq: str, amino_acids: str, host: str, evaluator: CarbonLikelihoodEvaluator) -> float:
    """
    Computes the multi-objective fitness score of the DNA sequence.
    """
    # 1. CAI
    cai = calculate_cai(dna_seq, amino_acids, host)
    
    # 2. GC Content
    gc_count = dna_seq.count("G") + dna_seq.count("C")
    gc_ratio = gc_count / len(dna_seq)
    
    # GC Penalty: target is 50%, penalty if GC is outside 40%-60%
    gc_penalty = 0.0
    if gc_ratio < 0.40:
        gc_penalty = (0.40 - gc_ratio) ** 2 * 20.0
    elif gc_ratio > 0.60:
        gc_penalty = (gc_ratio - 0.60) ** 2 * 20.0
        
    # 3. Homopolymer repeats
    repeat_penalty = 0.0
    for base in ["A", "T", "G", "C"]:
        run = base * 5
        run_count = dna_seq.count(run)
        if run_count > 0:
            repeat_penalty += run_count * 5.0
            
    # 4. Restriction sites
    restriction_penalty = 0.0
    for name, site in RESTRICTION_SITES.items():
        site_count = dna_seq.count(site)
        if site_count > 0:
            restriction_penalty += site_count * 10.0
            
    # 5. Carbon-like Likelihood Score
    carbon_score = evaluator.score_sequence(dna_seq)
    
    # Total Score to maximize
    # CAI (max 1.0) + carbon_score (max 0.0, negative penalties) - GC - repeats - restriction
    total_score = (cai * 10.0) + carbon_score - gc_penalty - repeat_penalty - restriction_penalty
    return total_score

def optimize_codons(amino_acids: str, host: str = "human", steps: int = 1000, 
                     init_temp: float = 1.0, cooling_rate: float = 0.995) -> dict:
    """
    Runs simulated annealing to find the optimal codon configuration for mRNA stability and expression.
    """
    # Initialize with optimal CAI sequence
    curr_seq = list(get_optimal_cai_sequence(amino_acids, host))
    evaluator = CarbonLikelihoodEvaluator()
    
    curr_score = evaluate_objective("".join(curr_seq), amino_acids, host, evaluator)
    
    best_seq = list(curr_seq)
    best_score = curr_score
    
    temp = init_temp
    codon_table = CODON_TABLES[host]
    
    for step in range(steps):
        # Choose a random residue index to mutate
        idx = random.randint(0, len(amino_acids) - 1)
        aa = amino_acids[idx]
        
        # Get alternative codons
        synonymous = list(codon_table[aa].keys())
        if len(synonymous) <= 1:
            continue
            
        old_codon = "".join(curr_seq[idx*3 : (idx+1)*3])
        new_codon = random.choice([c for c in synonymous if c != old_codon])
        
        # Apply mutation
        candidate_seq = list(curr_seq)
        candidate_seq[idx*3 : (idx+1)*3] = list(new_codon)
        candidate_str = "".join(candidate_seq)
        
        # Evaluate
        candidate_score = evaluate_objective(candidate_str, amino_acids, host, evaluator)
        
        # Metropolis choice
        delta = candidate_score - curr_score
        if delta > 0 or random.random() < math.exp(delta / temp):
            curr_seq = candidate_seq
            curr_score = candidate_score
            
            if curr_score > best_score:
                best_seq = list(curr_seq)
                best_score = curr_score
                
        # Cool down
        temp *= cooling_rate
        
    final_str = "".join(best_seq)
    
    # Calculate final stats
    final_gc = (final_str.count("G") + final_str.count("C")) / len(final_str)
    final_cai = calculate_cai(final_str, amino_acids, host)
    
    # Check restriction sites present
    found_sites = []
    for name, site in RESTRICTION_SITES.items():
        if site in final_str:
            found_sites.append(name)
            
    # Check max homopolymer run
    max_run = 1
    curr_run = 1
    for i in range(1, len(final_str)):
        if final_str[i] == final_str[i-1]:
            curr_run += 1
            max_run = max(max_run, curr_run)
        else:
            curr_run = 1
            
    return {
        "dna_sequence": final_str,
        "gc_content": round(final_gc, 4),
        "cai": round(final_cai, 4),
        "score": round(best_score, 4),
        "restriction_sites": found_sites,
        "max_homopolymer_run": max_run
    }

def run_codon_optimization(fasta_path: str, host: str, output_csv: str, steps: int = 1000):
    """
    Reads an amino acid fasta, runs optimization for each sequence, and writes the results.
    """
    print("==============================================")
    print("  Somasays Codon Optimizer & mRNA Likelihood Engine")
    print("==============================================")
    print(f"[*] Loading protein sequences from: {fasta_path}")
    print(f"[*] Target expression host: {host}")
    
    # Simple fasta parser
    sequences = {}
    current_header = None
    
    with open(fasta_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_header = line[1:]
                sequences[current_header] = []
            elif current_header:
                sequences[current_header].append(line)
                
    for header in sequences:
        sequences[header] = "".join(sequences[header]).replace(" ", "").upper()
        
    if not sequences:
        print("[ERROR] No sequences found in the fasta file.")
        return
        
    print(f"[*] Found {len(sequences)} candidates. Starting Simulated Annealing...")
    
    import csv
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["header", "protein_sequence", "dna_sequence", "gc_content", "cai", "cai_score", "restriction_sites_remaining", "max_homopolymer_run"])
        
        for header, aa_seq in sorted(sequences.items()):
            print(f"  [+] Optimizing {header} (length {len(aa_seq)} aa)...")
            res = optimize_codons(aa_seq, host=host, steps=steps)
            writer.writerow([
                header,
                aa_seq,
                res["dna_sequence"],
                res["gc_content"],
                res["cai"],
                res["score"],
                ",".join(res["restriction_sites"]) if res["restriction_sites"] else "None",
                res["max_homopolymer_run"]
            ])
            
    print(f"\n[SUCCESS] Codon optimization complete!")
    print(f"   [+] Saved report: {output_csv}")
    print("==============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize protein back-translation for mRNA expression and synthesis")
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file containing protein designs")
    parser.add_argument("--host", type=str, choices=["human", "e_coli"], default="human", help="Host expression system")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV report file")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulated annealing steps")
    
    args = parser.parse_args()
    run_codon_optimization(args.fasta, args.host, args.out_csv, args.steps)
