"""
ESMFold2-Fast local folding wrapper for Somasays.

Single-chain  → ptm + plddt_mean (target_seq=None)
Complex fold  → real iptm + ipSAE (default: HTNV nucleoprotein as chain B)

Model: biohub/ESMFold2-Experimental-Fast (cached at ~/.cache/huggingface/hub/)
       ESMC-6B backbone (~12 GB) — loads in ~3 min first call, instant thereafter.

Data source: UniProt (https://www.uniprot.org) — CC BY 4.0.
"""
import os
import argparse
import numpy as np
import torch
from transformers.models.esmfold2 import ESMFold2ExperimentalModel

# ── Hantaan virus nucleoprotein — UniProt P05133, strain 76-118, PE=1 SV=1 ─────
# Target for the Somasays binder design campaign.
# https://www.uniprot.org/uniprotkb/P05133
HANTAV_N_SEQ = (
    "MATMEELQREINAHEGQLVIARQKVRDAEKQYEKDPDELNKRTLTDREGVAVSIQAKIDE"
    "LKRQLADRIATGKNLGKEQDPTGVEPGDHLKERSMLSYGNVLDLNHLDIDEPTGQTADWL"
    "SIIVYLTSFVVPILLKALYMLTTRGRQTTKDNKGTRIRFKDDSSFEDVNGIRKPKHLYVS"
    "LPNAQSSMKAEEITPGRYRTAVCGLYPAQIKARQMISPVMSVIGFLALAKDWSDRIEQWL"
    "IEPCKLLPDTAAVSLLGGPATNRDYLRQRQVALGNMETKESKAIRQHAEAAGCSMIEDIE"
    "SPSSIWVFAGAPDRCPPTCLFIAGIAELGAFFSILQDMRNTIMASKTVGTSEEKLRKKSS"
    "FYQSYLRRTQSMGIQLGQRIIVLFMVAWGKEAVDNFHLGDDMDPELRTLAQSLIDVKVKE"
    "ISNQEPLKL"
)

# Default target exposed at module level so evaluate_cysteine_free_complexes can import it
DEFAULT_TARGET_SEQ = HANTAV_N_SEQ

# ponytail: singleton — load once per process, fold many
_MODEL = None
_BUILDER = None


def _get_model():
    global _MODEL, _BUILDER
    if _MODEL is None:
        from esm.models.esmfold2 import ESMFold2InputBuilder
        print("[ESMFold2] Loading biohub/ESMFold2-Experimental-Fast from local cache...")
        print("           (ESMC-6B backbone = 6 safetensor shards, ~3 min first load)")
        _MODEL = (
            ESMFold2ExperimentalModel
            .from_pretrained(
                "biohub/ESMFold2-Experimental-Fast",
                local_files_only=True,   # weights pre-cached — never hit network
            )
            .to(torch.bfloat16)
            .cuda()
            .eval()
        )
        _BUILDER = ESMFold2InputBuilder()
        print("[ESMFold2] Model ready.")
    return _MODEL, _BUILDER


def fold_sequence(
    sequence: str,
    out_cif: str,
    target_seq: str | None = DEFAULT_TARGET_SEQ,
) -> dict:
    """
    Fold a binder sequence with ESMFold2-Fast.

    Args:
        sequence:   Binder amino acid sequence (chain A).
        out_cif:    Destination mmCIF file path.
        target_seq: Optional target sequence (chain B). When provided, folds as a
                    complex → real ipTM and ipSAE from cross-chain PAE blocks.
                    Defaults to the Hantaan virus nucleoprotein (P05133).
                    Pass None to fold binder alone (ptm proxy only).

    Returns:
        dict: iptm, ptm, plddt_mean, interface_pae, cif_path
    """
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    model, builder = _get_model()

    len_a = len(sequence)
    if target_seq:
        # 2-chain complex → real interface metrics
        inputs = [
            ProteinInput(id="A", sequence=sequence),
            ProteinInput(id="B", sequence=target_seq),
        ]
    else:
        # single-chain → ptm/plddt only
        inputs = [ProteinInput(id="A", sequence=sequence)]

    spi = StructurePredictionInput(sequences=inputs)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # ponytail: num_loops=4 is fast-mode (paper default 20); num_sampling_steps=100
            result = builder.fold(model, spi, num_loops=4, num_sampling_steps=100)

    os.makedirs(os.path.dirname(os.path.abspath(out_cif)), exist_ok=True)
    with open(out_cif, "w") as f:
        f.write(result.complex.to_mmcif())

    # ipSAE: mean of both cross-chain PAE blocks (binder→target and target→binder)
    # Lower PAE = more confident interface placement = better predicted binding.
    if target_seq is not None and result.pae is not None:
        pae = result.pae.cpu().float().numpy()
        ipsae = float((pae[:len_a, len_a:].mean() + pae[len_a:, :len_a].mean()) / 2)
    else:
        ipsae = float("nan")

    return {
        "iptm":          float(result.iptm) if result.iptm is not None else 0.0,
        "ptm":           float(result.ptm)  if result.ptm  is not None else 0.0,
        "plddt_mean":    float(result.plddt.mean()),
        "interface_pae": ipsae,   # ipSAE — lower is better
        "cif_path":      out_cif,
    }


# ── CLI smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESMFold2-Fast local folder")
    parser.add_argument("--seq", required=True, help="Binder amino acid sequence")
    parser.add_argument("--target", default="hantav_n",
                        choices=["hantav_n", "none"],
                        help="Target for complex folding (default: hantav_n = HTNV-N P05133)")
    parser.add_argument("--out", default="outputs/esmfold2_structures/test.cif")
    args = parser.parse_args()

    target = HANTAV_N_SEQ if args.target == "hantav_n" else None
    metrics = fold_sequence(args.seq, args.out, target_seq=target)
    print(f"\n[+] Folded → {metrics['cif_path']}")
    print(f"    pLDDT mean   : {metrics['plddt_mean']:.3f}")
    print(f"    pTM          : {metrics['ptm']:.3f}")
    print(f"    ipTM         : {metrics['iptm']:.3f}")
    print(f"    ipSAE (PAE)  : {metrics['interface_pae']:.3f}  (lower = better)")
