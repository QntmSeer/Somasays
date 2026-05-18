import os
from huggingface_hub import login

token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
    print("Hugging Face Auth Successful (via HF_TOKEN)")
else:
    print("HF_TOKEN environment variable not set. Attempting to use cached credentials...")
    try:
        # huggingface_hub automatically loads credentials from standard cache directory
        login()
    except Exception:
        print("WARNING: Hugging Face authentication not configured. Make sure to set HF_TOKEN or run 'huggingface-cli login'.")
