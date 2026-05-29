import os
import pytest
import tempfile
import json
from unittest.mock import patch

# Add Somasays root to Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation_and_rescue.manufacturability_profiler import (
    calculate_gravy,
    estimate_isoelectric_point,
    detect_hydrophobic_patches,
    scan_ptm_and_instability
)

from evaluation_and_rescue.binding_interface_analyzer import (
    calculate_distance,
    parse_pdb_coordinates,
    analyze_interface,
    parse_confidence_json
)

# --- 1. Manufacturability Profiler Tests ---

def test_calculate_gravy():
    # Alanine has Kyte-Doolittle index of 1.8
    assert calculate_gravy("AAA") == 1.8
    # Arginine has index of -4.5
    assert calculate_gravy("RRR") == -4.5
    # Mixed sequence
    assert calculate_gravy("AR") == round((1.8 - 4.5) / 2.0, 3)

def test_estimate_isoelectric_point():
    # A sequence rich in Lysine (K) should be basic (high pI)
    pI_basic = estimate_isoelectric_point("KKKKKKK")
    assert pI_basic > 9.0
    
    # A sequence rich in Aspartate (D) should be acidic (low pI)
    pI_acidic = estimate_isoelectric_point("DDDDDDD")
    assert pI_acidic < 5.0

def test_detect_hydrophobic_patches():
    # No hydrophobic patch
    assert len(detect_hydrophobic_patches("KKKKKKKK")) == 0
    
    # Valid hydrophobic run of length >= 5 (VILFW)
    patches = detect_hydrophobic_patches("KKVILFWYKK", min_len=5)
    assert len(patches) == 1
    assert patches[0][0] == 3 # Start position (1-indexed)
    assert patches[0][1] == "VILFWY"

def test_scan_ptm_and_instability():
    # Test N-Glycosylation motif N-X-S/T where X is not P
    res = scan_ptm_and_instability("NAT") # Has N-A-T
    assert len(res["n_glyco"]) == 1
    
    # Test glycosylation with proline blocker (should be skipped)
    res_blocked = scan_ptm_and_instability("NPT")
    assert len(res_blocked["n_glyco"]) == 0
    
    # Test deamidation
    res_deamid = scan_ptm_and_instability("NGNS")
    assert len(res_deamid["deamidation"]) == 2
    
    # Test acid cleavage
    res_cleavage = scan_ptm_and_instability("DP")
    assert len(res_cleavage["acid_cleavage"]) == 1


# --- 2. Binding Interface Analyzer Tests ---

def test_calculate_distance():
    coord1 = (0.0, 0.0, 0.0)
    coord2 = (3.0, 4.0, 0.0)
    assert calculate_distance(coord1, coord2) == 5.0

def test_pdb_coordinate_parser_and_interface_analysis():
    # Create temporary PDB file to test parsing and geometry calculations
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w", encoding="utf-8") as temp_pdb:
        temp_pdb_name = temp_pdb.name
        # Chain A - Binder (Lysine)
        temp_pdb.write("ATOM      1  N   LYS A   1      10.000  10.000  10.000  1.00 95.00           N\n")
        temp_pdb.write("ATOM      2  NZ  LYS A   1      13.000  10.000  12.000  1.00 94.00           N\n")
        # Chain B - Target (Aspartate) - distance to Lys NZ is 3.0 A
        temp_pdb.write("ATOM      3  OD1 ASP B   1      13.000  10.000  15.000  1.00 90.00           O\n")
        temp_pdb.write("END\n")

    try:
        # Parse PDB
        chains = parse_pdb_coordinates(temp_pdb_name)
        assert "A" in chains
        assert "B" in chains
        assert len(chains["A"]) == 2
        assert len(chains["B"]) == 1
        
        # Analyze interface
        geom = analyze_interface(chains, binder_chain="A")
        assert geom["num_contacts"] == 1 # NZ-OD1 is 3.0 A
        assert geom["num_h_bonds"] == 1  # NZ (N) and OD1 (O) are polar within 3.5 A
        assert geom["num_salt_bridges"] == 1 # Lys and Asp sidechain within 4.0 A
        assert geom["avg_binder_plddt"] == 94.5 # (95 + 94) / 2
        assert geom["avg_interface_plddt"] == 94.0 # NZ is the only interface atom (pLDDT=94)
    finally:
        os.remove(temp_pdb_name)

def test_parse_confidence_json():
    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as temp_json:
        temp_json_name = temp_json.name
        json.dump({
            "ptm": 0.85,
            "iptm": 0.76,
            "chain_pair_pae_min": 4.2
        }, temp_json)
        
    try:
        conf = parse_confidence_json(temp_json_name)
        assert conf["ptm"] == 0.85
        assert conf["iptm"] == 0.76
        assert conf["ipsae_pae_min"] == 4.2
    finally:
        os.remove(temp_json_name)

def test_calculate_solubility_ratio():
    from evaluation_and_rescue.manufacturability_profiler import calculate_solubility_ratio
    # A highly charged sequence
    seq_charged = "DKRDKRDKR" # 9 charged residues, 0 hydrophobic
    assert calculate_solubility_ratio(seq_charged) == 9.0
    
    # A highly hydrophobic sequence
    seq_hydrophobic = "VILFWYVILFWY" # 0 charged residues, 12 hydrophobic
    assert calculate_solubility_ratio(seq_hydrophobic) == 0.0

def test_glycan_clash_detection():
    # Create temporary PDB file with a glycan residue NAG in Chain B
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w", encoding="utf-8") as temp_pdb:
        temp_pdb_name = temp_pdb.name
        # Chain A - Binder
        temp_pdb.write("ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 95.00           C\n")
        # Chain B - Target (Glycan NAG) - distance to ALA CA is 3.0 A
        temp_pdb.write("HETATM    2  O1  NAG B   1      10.000  10.000  13.000  1.00 90.00           O\n")
        temp_pdb.write("END\n")
        
    try:
        chains = parse_pdb_coordinates(temp_pdb_name)
        geom = analyze_interface(chains, binder_chain="A")
        assert geom["num_glycan_clashes"] == 1 # O1 in NAG is 3.0 A from ALA CA
    finally:
        os.remove(temp_pdb_name)

def test_structural_qc_analysis():
    from evaluation_and_rescue.structural_qc import analyze_structural_qc, predict_mhc_epitopes
    
    # 1. Test sequence-based epitope prediction
    seq_high_risk = "FYWILVMFLAGS" # contains many hydrophobic anchors
    epitopes = predict_mhc_epitopes(seq_high_risk)
    assert len(epitopes) > 0
    
    # 2. Test structure-based QC checks
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w", encoding="utf-8") as temp_pdb:
        temp_pdb_name = temp_pdb.name
        # N-terminus N atom of first residue
        temp_pdb.write("ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 90.00           N\n")
        # Cysteine 1 SG
        temp_pdb.write("ATOM      2  SG  CYS A   2      12.000  12.000  12.000  1.00 90.00           S\n")
        # Cysteine 2 SG (distance to Cys 1 is 2.0 A)
        temp_pdb.write("ATOM      3  SG  CYS A   3      12.000  12.000  14.000  1.00 90.00           S\n")
        # C-terminus O atom of last residue (close to N-term for cyclization, 4.0 A)
        temp_pdb.write("ATOM      4  O   ALA A   4      10.000  10.000  14.000  1.00 90.00           O\n")
        temp_pdb.write("END\n")
        
    try:
        qc = analyze_structural_qc(temp_pdb_name, chain_id="A")
        assert qc["num_cys"] == 2
        assert qc["formed_disulfides"] == 1
        assert qc["free_thiols"] == 0
        assert qc["cyclization_dist_angstrom"] == 4.0
        assert qc["cyclization_feasibility"] == "Highly Feasible"
    finally:
        os.remove(temp_pdb_name)

def test_peptide_rescue_workflow():
    from evaluation_and_rescue.candidate_rescuer import perform_peptide_rescue
    
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w", encoding="utf-8") as temp_pdb:
        temp_pdb_name = temp_pdb.name
        # Chain A - Binder (length 10 contiguous residues)
        temp_pdb.write("ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 90.00           N\n")
        temp_pdb.write("ATOM      2  CB  ALA A   1      10.000  10.000  10.000  1.00 90.00           C\n")
        temp_pdb.write("ATOM      3  CB  ALA A   2      20.000  20.000  20.000  1.00 90.00           C\n")
        temp_pdb.write("ATOM      4  CB  ALA A   3      20.000  20.000  20.000  1.00 90.00           C\n")
        temp_pdb.write("ATOM      5  CB  ALA A   4      20.000  20.000  20.000  1.00 90.00           C\n")
        temp_pdb.write("ATOM      6  CB  ALA A   5      10.000  10.000  14.000  1.00 90.00           C\n") # 4.0 A from ALA 1 CB, separated by 4 residues
        temp_pdb.write("ATOM      7  CB  ALA A   6      20.000  20.000  20.000  1.00 90.00           C\n")
        temp_pdb.write("ATOM      8  CB  ALA A   7      20.000  20.000  20.000  1.00 90.00           C\n")
        temp_pdb.write("ATOM      9  CB  ALA A   8      20.000  20.000  20.000  1.00 90.00           C\n")
        temp_pdb.write("ATOM     10  CB  ALA A   9      20.000  20.000  20.000  1.00 90.00           C\n")
        temp_pdb.write("ATOM     11  CB  ALA A  10      20.000  20.000  20.000  1.00 90.00           C\n")
        # Chain B - Target (far away, no interface clashes)
        temp_pdb.write("ATOM     12  CA  LYS B   1      50.000  50.000  50.000  1.00 90.00           C\n")
        temp_pdb.write("END\n")
        
    try:
        res = perform_peptide_rescue(temp_pdb_name, binder_chain="A", target_chain="B")
        # Should pair ALA A1 and ALA A5 for disulfide
        assert res["rescued_disulfides"] == 1
        assert len(res["disulfides_introduced"]) == 1
    finally:
        os.remove(temp_pdb_name)
