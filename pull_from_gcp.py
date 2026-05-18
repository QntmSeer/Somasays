import os
from google.cloud import storage

def download_weights():
    bucket_name = "somasays-storage"
    source_blob_name = "weights/v3/esm3_multimodal_weights.pth"
    destination_file_name = "weights/esm3_multimodal_weights.pth"

    os.makedirs("weights", exist_ok=True)
    
    # Use service account key
    storage_client = storage.Client.from_service_account_json("gcp_key.json")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    print(f"Downloading {source_blob_name} to {destination_file_name}...")
    blob.download_to_filename(destination_file_name)
    print("Download complete!")

if __name__ == "__main__":
    download_weights()
