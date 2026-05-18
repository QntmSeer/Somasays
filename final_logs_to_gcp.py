import os
from google.cloud import storage

def final_cleanup():
    KEY_PATH = "/home/shadeform/gcp_key.json"
    BUCKET_NAME = "somasays-storage"
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    # 1. Upload Tensorboard Logs
    tb_dir = "/home/shadeform/Somasays/weights/somasays_multimodal_v3/logs/"
    if os.path.exists(tb_dir):
        for f in os.listdir(tb_dir):
            src = os.path.join(tb_dir, f)
            if os.path.isfile(src):
                dest = f"weights/v3/tensorboard/{f}"
                print(f"   [SYNC] {f} -> {dest}")
                bucket.blob(dest).upload_from_filename(src)
                
    # 2. Upload Raw Execution Log
    raw_log = "/home/shadeform/Somasays/multimodal_run.log"
    if os.path.exists(raw_log):
        dest = "weights/v3/multimodal_run.log"
        print(f"   [SYNC] multimodal_run.log -> {dest}")
        bucket.blob(dest).upload_from_filename(raw_log)

if __name__ == "__main__":
    final_cleanup()
    print("\n[SUCCESS] GOLDEN LOGS SECURED")
