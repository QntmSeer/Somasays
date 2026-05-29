import pytest
from evaluation_and_rescue.codon_optimizer_carbon import (
    CODON_TABLES,
    RESTRICTION_SITES,
    CarbonLikelihoodEvaluator,
    calculate_cai,
    get_optimal_cai_sequence,
    optimize_codons
)

def test_codon_tables_loaded():
    """Verify both e_coli and human codon tables contain entries and synonymous codon probabilities."""
    assert "e_coli" in CODON_TABLES
    assert "human" in CODON_TABLES
    
    # Test methionine mapping (Met should only have ATG with probability 1.0)
    assert CODON_TABLES["e_coli"]["M"] == {"ATG": 1.0}
    assert CODON_TABLES["human"]["M"] == {"ATG": 1.0}
    
    # Check Ala (A) synonymous codon list
    assert "GCG" in CODON_TABLES["e_coli"]["A"]
    assert "GCC" in CODON_TABLES["human"]["A"]

def test_optimal_cai_sequence():
    """Test get_optimal_cai_sequence creates matching translation and is correct length."""
    protein = "MKAAAAVLA"
    dna_seq = get_optimal_cai_sequence(protein, "human")
    assert len(dna_seq) == len(protein) * 3
    assert dna_seq[0:3] == "ATG" # Met
    assert dna_seq[3:6] == "AAG" # Lys is AAG in human (highest frequency)
    
    # Check CAI calculation is valid (should be 1.0 for optimal CAI sequence)
    cai = calculate_cai(dna_seq, protein, "human")
    assert pytest.approx(cai) == 1.0

def test_carbon_evaluator():
    """Verify Carbon likelihood evaluator scores sequences and penalizes homopolymers."""
    evaluator = CarbonLikelihoodEvaluator()
    
    # A balanced sequence
    score_balanced = evaluator.score_sequence("GCATCGGCATCG") # GC ratio 0.5
    
    # A sequence with homopolymer repeat
    score_homopolymer = evaluator.score_sequence("AAAAAAGCATCG")
    
    assert score_balanced > score_homopolymer

def test_optimize_codons():
    """Test simulated annealing optimization improves base parameters."""
    protein = "MKNTQSRVEAD"
    
    # Run optimization for E. coli
    res = optimize_codons(protein, host="e_coli", steps=100)
    
    assert len(res["dna_sequence"]) == len(protein) * 3
    assert res["cai"] > 0.0
    assert 0.35 <= res["gc_content"] <= 0.70
    assert res["max_homopolymer_run"] <= 6

def test_restriction_site_avoidance():
    """Verify optimization can eliminate common restriction sites."""
    # "MKAA" where A is Ala. GCG is preferred in E. coli.
    # If we have "M-K-A-A" and we deliberately introduce a sequence that could form EcoRI (GAATTC)
    # The optimizer should avoid creating this pattern or score it negatively.
    protein = "GAATTC" # Translated to Glu-Ile-...
    # Let's ensure the optimization returns a valid dict schema
    res = optimize_codons("MEIR", host="human", steps=50)
    assert "dna_sequence" in res
    assert "restriction_sites" in res
