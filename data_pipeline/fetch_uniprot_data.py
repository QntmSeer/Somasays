import os
import requests
import urllib.parse

# Targeted Medicine Keywords
MEDICINAL_KEYWORDS = [
    "KW-0044", # Antimicrobial
    "KW-0214", # Defensin
    "KW-0800", # Toxin
    "KW-0045", # Antiviral
]

def fetch_uniprot_global_medicine(output_path: str):
    """
    Fetches targeted medicinal proteins from the entire Global Plant Kingdom (Viridiplantae).
    """
    print("Fetching ALL Plant Medicinal Protein Data from UniProt...")
    
    # Constructing the complex query
    taxa_query = "(taxonomy_id:33090)" # Viridiplantae (All Plants)
    kw_query = " OR ".join([f"(keyword:{kw})" for kw in MEDICINAL_KEYWORDS])
    
    # Final query: All plants AND any of the medicinal functions
    full_query = f"({taxa_query}) AND ({kw_query})"
    
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": full_query,
        "format": "fasta",
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Executing UniProt Query: {full_query}")
    with requests.get(url, params=params, stream=True) as response:
        response.raise_for_status()
        with open(output_path, 'w') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk.decode('utf-8'))
                
    print(f"Data successfully saved to {output_path}")

if __name__ == "__main__":
    fetch_uniprot_global_medicine("../data/raw/uniprot_global_medicine.fasta")
