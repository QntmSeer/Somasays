import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Silence huggingface fork deadlock warnings

import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from esm.models.esm3 import ESM3
from tqdm import tqdm
from io import StringIO
from torch.utils.tensorboard import SummaryWriter
import csv

class SomasaysMultimodalDataset(Dataset):
    def __init__(self, jsonl_path, pdb_dir):
        """
        Pre-caches all 1D Sequences and 3D PDB text files into RAM to completely
        eliminate the hard drive I/O bottleneck during the PyTorch training loop.
        """
        print("Pre-caching dataset into Host RAM...")
        self.data = []
        
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Loading JSONL to RAM"):
            entry = json.loads(line)
            sequence = entry.get("sequence", "")
            header = entry.get("header", "")
            
            # Extract UniProt ID to find PDB
            uniprot_id = header.split('|')[1] if '|' in header else None
            pdb_path = os.path.join(pdb_dir, f"{uniprot_id}.pdb")
            
            if uniprot_id and os.path.exists(pdb_path):
                # Load the entire PDB file string directly into RAM!
                with open(pdb_path, 'r') as pdb_f:
                    pdb_string = pdb_f.read()
                
                self.data.append({
                    "sequence": sequence,
                    "pdb_string": pdb_string
                })
                
        print(f"Successfully cached {len(self.data)} multimodal targets in RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Here we would use the ESM3 tokenization logic to convert sequence + pdb_string into tensors
        item = self.data[idx]
        return item

def setup_distributed():
    # Initialize the process group for torchrun (FSDP requirement)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def train_multimodal_esm3(jsonl_path, pdb_dir, output_dir):
    local_rank = setup_distributed()
    
    # 1. Hardware Optimizations
    torch.backends.cudnn.benchmark = True # Auto-tunes convolutions for A100
    
    if local_rank == 0:
        print("Initializing Prod-Level Multimodal ESM3 Architecture...")
    
    # 2. Dataset & DataLoader (pin_memory=True for DMA to A100 VRAM)
    dataset = SomasaysMultimodalDataset(jsonl_path, pdb_dir)
    # Use DistributedSampler to shard data across GPUs (or processes)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=2, # Reduced to prevent OOM on long geometric attention sequences
        sampler=sampler,
        num_workers=8, # Multi-threaded pre-processing
        pin_memory=True # DMA Transfer directly to GPU VRAM
    )
    
    # 3. Load Model & Wrap in FSDP (CPU Offloading)
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    model = model.to(local_rank)
    
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # The 1.4B parameter model easily fits in the A100 80GB VRAM.
    # We replace FSDP with DDP (Distributed Data Parallel), which is the industry standard
    # for models < 3B params and natively handles unused multimodality heads without crashing.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # PyTorch DDP has a known re-entrancy bug when combining find_unused_parameters=True
    # with Transformer Gradient Checkpointing. We freeze the graph topology to fix this.
    model._set_static_graph()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Phase 4 Tracking Setup
    writer = None
    if local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
        csv_log_path = os.path.join(output_dir, "loss_log.csv")
        with open(csv_log_path, mode='w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["Epoch", "Batch", "Seq_Loss", "Struct_Loss", "Total_Loss"])
            
    global_step = 0
    # PyTorch default ignore_index is -100. ESM3 encoded structure tokens can contain -1 for missing coordinates,
    # which crashes CrossEntropyLoss if ignore_index=0. We standardize all padding/missing to -100.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # 5. High-Performance Training Loop
    for epoch in range(50): # Phase 4 target epochs
        sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f"\n--- Epoch {epoch+1} ---")
        
        # We only want the progress bar on the main process
        batch_iterator = tqdm(dataloader, desc="Training") if local_rank == 0 else dataloader
        
        for batch_idx, batch_data in enumerate(batch_iterator):
            optimizer.zero_grad()
            
            # 5. Automatic Mixed Precision (AMP)
            # Utilizing the A100's bfloat16 Tensor Cores for 2x speed and half memory
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                
                # ESM3 Multimodal Forward Pass
                # Convert our pre-cached raw sequence/PDB data into ESMProtein objects
                from esm.sdk.api import ESMProtein
                from io import StringIO
                from torch.nn.utils.rnn import pad_sequence
                
                sequence_tokens_list = []
                structure_tokens_list = []
                
                for i in range(len(batch_data["sequence"])):
                    # We create an ESMProtein object for each item in the batch
                    protein = ESMProtein.from_pdb(StringIO(batch_data["pdb_string"][i]))
                    
                    # Ensure atomic coordinates are on the GPU before the VQ-VAE processes them
                    if hasattr(protein, 'coordinates') and protein.coordinates is not None:
                        if isinstance(protein.coordinates, torch.Tensor):
                            protein.coordinates = protein.coordinates.to(local_rank)
                        else:
                            protein.coordinates = torch.tensor(protein.coordinates, device=local_rank)
                            
                    # Isolate the encoder from the FSDP autograd graph tracking!
                    # Calling model.encode directly touches FSDP weights outside the forward loop, 
                    # crashing the state machine. We bypass FSDP by using .module and no_grad.
                    with torch.no_grad():
                        protein_tensor = model.module.encode(protein)
                    
                    # Truncate sequence length to 1024 to prevent O(N^2) memory spikes
                    # during Geometric Attention on massive multi-domain proteins
                    seq_toks = protein_tensor.sequence[:1024]
                    struct_toks = protein_tensor.structure[:1024]
                    
                    sequence_tokens_list.append(seq_toks)
                    structure_tokens_list.append(struct_toks)
                    
                # Pad variable-length proteins into a single rigid tensor batch (B, L)
                # The model requires valid embedding indices, so padding must be 0 (not -100)
                seq_batch = pad_sequence(sequence_tokens_list, batch_first=True, padding_value=0).to(local_rank).detach()
                struct_batch = pad_sequence(structure_tokens_list, batch_first=True, padding_value=0).to(local_rank).detach()
                
                # Single Unified Forward Pass for the whole batch
                output = model(
                    sequence_tokens=seq_batch,
                    structure_tokens=struct_batch
                )
                
                # Multi-Objective Cross-Entropy Loss
                # Reshape logits to (B*L, VocabSize) and targets to (B*L) to match PyTorch CrossEntropy signature
                seq_logits = output.sequence_logits
                struct_logits = output.structure_logits
                
                seq_targets = seq_batch.view(-1).clone()
                struct_targets = struct_batch.view(-1).clone()
                
                # We used 0 for padding above, but we must ignore padded positions in the loss.
                seq_targets[seq_targets == 0] = -100
                struct_targets[struct_targets == 0] = -100
                
                # Protect against VQ-VAE returning out-of-bounds special tokens (like <BOS>/<EOS> which exceed logits dimensions)
                # by masking anything < 0 or >= VocabSize to -100 (ignore_index)
                seq_targets = torch.where((seq_targets < 0) | (seq_targets >= seq_logits.size(-1)), torch.tensor(-100, device=seq_targets.device), seq_targets)
                struct_targets = torch.where((struct_targets < 0) | (struct_targets >= struct_logits.size(-1)), torch.tensor(-100, device=struct_targets.device), struct_targets)
                
                seq_loss = criterion(seq_logits.view(-1, seq_logits.size(-1)), seq_targets)
                struct_loss = criterion(struct_logits.view(-1, struct_logits.size(-1)), struct_targets)
                
                loss = seq_loss + struct_loss
                        
            # Native Backward Pass
            # bfloat16 has the same exponent range as float32, so it natively prevents
            # underflow and DOES NOT require a GradScaler (unlike float16).
            loss.backward()
            optimizer.step()
            
            # Phase 4 Production Metrics Logging
            if local_rank == 0:
                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Loss/Sequence", seq_loss.item(), global_step)
                writer.add_scalar("Loss/Structure", struct_loss.item(), global_step)
                
                with open(csv_log_path, mode='a', newline='') as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow([epoch+1, batch_idx, seq_loss.item(), struct_loss.item(), loss.item()])
                    
                global_step += 1
        
        # --- EPOCH COMPLETE CHECKPOINT SAVE ---
        if local_rank == 0:
            epoch_save_path = os.path.join(output_dir, f"esm3_multimodal_epoch_{epoch+1}.pth")
            print(f"\n[Model Checkpointer] Saving Epoch {epoch+1} weights to {epoch_save_path}...")
            # Unwrap the DDP model to save pure, clean weights
            torch.save(model.module.state_dict(), epoch_save_path)
            
            # Keep a 'latest' checkpoint path updated for easy continuous transfers!
            latest_path = os.path.join(output_dir, "esm3_multimodal_latest.pth")
            torch.save(model.module.state_dict(), latest_path)
            
    if local_rank == 0:
        print(f"Saving Final Checkpoint to {output_dir}")
        # Unwrap the DDP model to save pure, clean weights
        torch.save(model.module.state_dict(), os.path.join(output_dir, "esm3_multimodal_weights.pth"))

    dist.destroy_process_group()

if __name__ == "__main__":
    train_multimodal_esm3(
        jsonl_path="../data/processed/somasays_dataset.jsonl",
        pdb_dir="../data/processed/3d_structures",
        output_dir="../weights/somasays_multimodal_v3"
    )
