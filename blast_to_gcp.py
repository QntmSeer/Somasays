import os
import glob
import argparse
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor

def upload_blob(bucket_name, source_file_name, destination_blob_name, storage_client):
    """Uploads a file to the bucket if it doesn't exist or size differs."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    try:
        # Check if already exists with same size
        if blob.exists():
            if blob.size == os.path.getsize(source_file_name):
                return # Skip
    except Exception:
        pass # If we can't check, just upload

    # print(f"   [SYNC] Uploading {os.path.basename(source_file_name)}...")
    blob.upload_from_filename(source_file_name)

def main():
    parser = argparse.ArgumentParser(description="Somasays High-Speed Cloud Sync")
    parser.add_argument("--src", type=str, required=True, help="Local directory to sync")
    parser.add_argument("--dest", type=str, required=True, help="Destination prefix in GCS bucket")
    parser.add_argument("--bucket", type=str, default="somasays-storage", help="GCS Bucket Name")
    parser.add_argument("--key", type=str, default="gcp_key.json", help="Path to GCP key")
    parser.add_argument("--pattern", type=str, default="*", help="File pattern (e.g. *.pdb)")
    parser.add_argument("--threads", type=int, default=16, help="Parallel upload threads")
    
    args = parser.parse_args()

    # 1. Auth
    if os.path.exists(args.key):
        client = storage.Client.from_service_account_json(args.key)
    else:
        # Fallback to env var or default
        client = storage.Client()
    
    # 2. Collect files
    source_dir = args.src.rstrip('/')
    dest_prefix = args.dest.strip('/')
    if dest_prefix:
        dest_prefix += "/"

    files = glob.glob(os.path.join(source_dir, args.pattern))
    
    if not files:
        # print(f"[INFO] No files found in {source_dir} matching {args.pattern}")
        return

    # print(f"[SYNC] Syncing {len(files)} files to gs://{args.bucket}/{dest_prefix}...")
    
    # 3. Parallel Upload
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        for file_path in files:
            blob_name = dest_prefix + os.path.basename(file_path)
            executor.submit(upload_blob, args.bucket, file_path, blob_name, client)

if __name__ == "__main__":
    main()
