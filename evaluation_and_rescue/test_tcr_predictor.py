import os
import sys

# Add Somasays root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation_and_rescue.tcr_peptide_predictor import score_tcr_activation
from evaluation_and_rescue.structural_qc import predict_tcr_activation_epitopes

def test_tcr_activation_scoring():
    print("[*] Running TCR activation scoring unit tests...")
    
    # 1. Test positive controls from Nature Biotech (2026) paper (high activation risk)
    positives = ["GRLPLLNPI", "TRLALIAPK", "RVMMLAPF"]
    for p in positives:
        score = score_tcr_activation(p, "CASSPATYSTDTQYF")
        print(f"  [Positive Control] {p} -> Score: {score}")
        assert score >= 0.8, f"Expected high activation score for positive control {p}, got {score}"

    # 2. Test negative controls (low activation risk)
    negatives = ["AAAAAARAA", "KKKKKKPKK", "GRLPLLNQI"]
    for n in negatives:
        score = score_tcr_activation(n, "CASSPATYSTDTQYF")
        print(f"  [Negative Control] {n} -> Score: {score}")
        assert score < 0.5, f"Expected low activation score for negative control {n}, got {score}"

    print("[+] TCR activation scoring unit tests passed successfully!")

def test_epitope_search():
    print("[*] Running TCR epitope search unit tests...")
    
    # Sequence containing PSG5 autoantigen at index 8 (GRLPLLNPI)
    seq = "MKARRLAAGRLPLLNPIAAEEAKKAAPVLA"
    epitopes = predict_tcr_activation_epitopes(seq, threshold=0.5)
    
    print(f"  Sequence: {seq}")
    print(f"  Found epitopes: {epitopes}")
    
    assert len(epitopes) == 1, f"Expected 1 activating epitope in sequence, found {len(epitopes)}"
    assert epitopes[0][1] == "GRLPLLNPI", f"Expected activating epitope sequence 'GRLPLLNPI', got {epitopes[0][1]}"
    assert epitopes[0][0] == 9, f"Expected activating epitope starting index 9, got {epitopes[0][0]}"

    print("[+] TCR epitope search unit tests passed successfully!")

if __name__ == "__main__":
    try:
        test_tcr_activation_scoring()
        print("-" * 50)
        test_epitope_search()
        print("\n[ALL TESTS PASSED] TCR-Peptide Specificity Predictor verified!")
    except AssertionError as e:
        print(f"\n[TEST FAILURE] {e}")
        sys.exit(1)
