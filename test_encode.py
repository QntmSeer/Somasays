from esm.sdk.api import ESMProtein
from esm.models.esm3 import ESM3
import torch
import warnings
warnings.filterwarnings("ignore")

model = ESM3.from_pretrained("esm3_sm_open_v1").cuda()
protein = ESMProtein(sequence="ACDEFGH")
try:
    print("Testing CPU protein...")
    tensor = model.encode(protein)
    print("Success!")
except Exception as e:
    print("Failed with CPU protein:", e)

try:
    print("Testing with torch.set_default_device...")
    torch.set_default_device('cuda')
    protein = ESMProtein(sequence="ACDEFGH")
    tensor = model.encode(protein)
    print("Success!")
except Exception as e:
    print("Failed with default device:", e)
