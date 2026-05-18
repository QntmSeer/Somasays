import os
import time
import argparse
import json
import torch
import gc
import warnings
from typing import Dict, Any

# Suppress PyTorch architecture and tokenization warnings
warnings.filterwarnings("ignore")

class ESM3Profiler:
    def __init__(self, model_name: str = "esm3_sm_open_v1", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.model = None
        
        print(f"[PROFILER] Initializing ESM3Profiler on device: {self.device.upper()}")
        if self.device == "cuda":
            print(f"[PROFILER] GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"[PROFILER] Initial GPU VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    def reset_gpu_stats(self):
        """Resets peak memory statistics to isolate measurements per operation."""
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def get_memory_stats(self) -> Dict[str, float]:
        """Returns peak and current memory stats in MB."""
        stats = {"peak_mb": 0.0, "current_mb": 0.0}
        if self.device == "cuda":
            stats["peak_mb"] = torch.cuda.max_memory_allocated(0) / 1024**2
            stats["current_mb"] = torch.cuda.memory_allocated(0) / 1024**2
        else:
            try:
                import psutil
                process = psutil.Process(os.getpid())
                stats["current_mb"] = process.memory_info().rss / 1024**2
            except ImportError:
                stats["current_mb"] = 0.0
        return stats

    def load_model(self) -> float:
        """Loads ESM3 and profiles loading latency and memory footprint."""
        print(f"[PROFILER] Loading model '{self.model_name}'...")
        self.reset_gpu_stats()
        
        start_time = time.perf_counter()
        
        # Import dynamically to allow robust dependency handling
        from esm.models.esm3 import ESM3
        self.model = ESM3.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        latency = time.perf_counter() - start_time
        mem_stats = self.get_memory_stats()
        
        print(f"[PROFILER] Model loaded in {latency:.4f}s.")
        print(f"[PROFILER] Peak Memory during Load: {mem_stats['peak_mb']:.2f} MB")
        
        return latency

    def profile_generation(
        self, 
        prompt: str, 
        num_steps: int = 8, 
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Profiles sequence generation and structure folding stages."""
        if self.model is None:
            raise RuntimeError("Model must be loaded via load_model() before profiling generation.")
            
        from esm.sdk.api import ESMProtein, GenerationConfig
        
        # Track total sequence length (prompt length)
        seq_len = len(prompt)
        num_masks = prompt.count('_')
        
        print(f"\n[PROFILER] Starting generation profiling for prompt length: {seq_len} ({num_masks} de novo tokens)")
        
        # Prepare input
        input_protein = ESMProtein(sequence=prompt)
        
        # --- PHASE A: SEQUENCE GENERATION ---
        self.reset_gpu_stats()
        seq_config = GenerationConfig(track="sequence", num_steps=num_steps, temperature=temperature)
        
        start_seq = time.perf_counter()
        with torch.no_grad():
            protein_with_seq = self.model.generate(input_protein, seq_config)
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        seq_latency = time.perf_counter() - start_seq
        seq_mem = self.get_memory_stats()
        
        # --- PHASE B: STRUCTURAL FOLDING ---
        self.reset_gpu_stats()
        struct_config = GenerationConfig(track="structure", num_steps=num_steps, temperature=temperature)
        
        start_struct = time.perf_counter()
        with torch.no_grad():
            final_protein = self.model.generate(protein_with_seq, struct_config)
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        struct_latency = time.perf_counter() - start_struct
        struct_mem = self.get_memory_stats()
        
        # Calculate tokens per second (throughput)
        seq_throughput = num_masks / seq_latency if seq_latency > 0 else 0.0
        struct_throughput = seq_len / struct_latency if struct_latency > 0 else 0.0
        
        results = {
            "prompt_length": seq_len,
            "generated_tokens": num_masks,
            "sequence_generation": {
                "latency_sec": seq_latency,
                "throughput_tokens_per_sec": seq_throughput,
                "peak_memory_mb": seq_mem["peak_mb"] if self.device == "cuda" else seq_mem["current_mb"]
            },
            "structural_folding": {
                "latency_sec": struct_latency,
                "throughput_residues_per_sec": struct_throughput,
                "peak_memory_mb": struct_mem["peak_mb"] if self.device == "cuda" else struct_mem["current_mb"]
            },
            "total_inference_latency_sec": seq_latency + struct_latency
        }
        
        print(f"[PROFILER] --- Sequence Generation Results ---")
        print(f"            Latency: {seq_latency:.4f} s")
        print(f"            Throughput: {seq_throughput:.2f} tok/s")
        print(f"            Peak Memory: {results['sequence_generation']['peak_memory_mb']:.2f} MB")
        
        print(f"[PROFILER] --- Structural Folding Results ---")
        print(f"            Latency: {struct_latency:.4f} s")
        print(f"            Throughput: {struct_throughput:.2f} res/s")
        print(f"            Peak Memory: {results['structural_folding']['peak_memory_mb']:.2f} MB")
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Somasays ESM3 Inference Bottleneck Profiler")
    parser.add_argument("--model", type=str, default="esm3_sm_open_v1", help="ESM3 model variant")
    parser.add_argument("--prompt", type=str, default="MKA___________________VLA", help="Peptide generation prompt template")
    parser.add_argument("--steps", type=int, default=8, help="Number of sampling steps")
    parser.add_argument("--output", type=str, default="outputs/profiling_report.json", help="Path to write JSON profiling results")
    args = parser.parse_args()
    
    profiler = ESM3Profiler(model_name=args.model)
    
    try:
        load_time = profiler.load_model()
        metrics = profiler.profile_generation(prompt=args.prompt, num_steps=args.steps)
        metrics["model_load_latency_sec"] = load_time
        metrics["device"] = profiler.device
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n[PROFILER] Successful execution! Profiling data saved to: {args.output}")
        
    except ImportError as e:
        print(f"\n[PROFILER] Environment setup error: {e}")
        print("[PROFILER] Make sure the 'esm' library and PyTorch are installed in the active environment.")
    except Exception as e:
        print(f"\n[PROFILER] Profiling encountered an unexpected error: {e}")
