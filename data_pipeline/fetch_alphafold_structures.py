import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def fetch_pdb_from_afdb(uniprot_id, output_dir):
    """
    Downloads the predicted 3D structure (PDB) from AlphaFold DB for a given UniProt ID.
    """
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
    output_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        return True
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            return True
        else:
            # Some UniProt IDs (e.g., very new or uncharacterized) might not have AF predictions
            return False
    except Exception as e:
        return False

def download_dataset_structures(jsonl_path: str, output_dir: str, max_workers: int = 10):
    """
    Parses the JSONL dataset to extract UniProt IDs and downloads all corresponding 3D PDBs.
    """
    print(f"Reading UniProt IDs from {jsonl_path}...")
    
    uniprot_ids = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Assuming the sequence dictionary has an ID or we extract it from the FASTA header
            # The preprocess_sequences.py needs to pass the header ID to the JSONL.
            if "header" in data:
                # Extract actual UniProt accession (e.g. "sp|P12345|..." -> "P12345")
                header_parts = data["header"].split('|')
                if len(header_parts) >= 2:
                    uniprot_ids.append(header_parts[1])
    
    if not uniprot_ids:
        print("No UniProt IDs found in the JSONL file. Ensure preprocessing saves the 'header' field.")
        return
        
    print(f"Found {len(uniprot_ids)} targets. Initiating massive 3D structural fetch from AlphaFold DB...")
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_pdb_from_afdb, uid, output_dir): uid for uid in uniprot_ids}
        
        for future in tqdm(futures, total=len(uniprot_ids), desc="Downloading 3D PDBs"):
            if future.result():
                success_count += 1
                
    print(f"\nSuccessfully downloaded {success_count}/{len(uniprot_ids)} 3D structures to {output_dir}.")

if __name__ == "__main__":
    download_dataset_structures(
        jsonl_path="../data/processed/somasays_dataset.jsonl",
        output_dir="../data/processed/3d_structures"
    )
