import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
import json

model = ESM3.from_pretrained("esm3_sm_open_v1").cuda()
with open("data/processed/somasays_dataset.jsonl") as f:
    line = f.readline()
data = json.loads(line)
with open("data/processed/3d_structures/" + data["uniprot_id"] + ".pdb") as f:
    pdb_str = f.read()

protein = ESMProtein(sequence=data["sequence"], coordinates=pdb_str)
tensor = model.encode(protein)
print("seq shape:", tensor.sequence.shape)
print("seq min/max:", tensor.sequence.min().item(), tensor.sequence.max().item())
print("struct shape:", tensor.structure.shape)
print("struct min/max:", tensor.structure.min().item(), tensor.structure.max().item())

logits = model(sequence_tokens=tensor.sequence.unsqueeze(0).cuda(), structure_tokens=tensor.structure.unsqueeze(0).cuda())
print("seq logits shape:", logits.sequence_logits.shape)
print("struct logits shape:", logits.structure_logits.shape)
