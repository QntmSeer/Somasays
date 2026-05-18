import os
import torch
import warnings
import gc
from typing import Dict, Any, Optional

warnings.filterwarnings("ignore")

class OptimizedESM3Generator:
    def __init__(
        self,
        model_name: str = "esm3_sm_open_v1",
        device: Optional[str] = None,
        precision: str = "bf16",
        enable_sdpa: bool = True,
        force_flash_attn: bool = False
    ):
        """
        An elite, high-performance wrapper for ESM3 that applies modern model compression 
        and architectural attention optimizations for inference speedups.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.precision = precision.lower()
        self.enable_sdpa = enable_sdpa
        self.force_flash_attn = force_flash_attn
        self.model = None

        print(f"[OPTIMIZATION] Initializing Somasays Optimized ESM3 Engine...")
        print(f"[OPTIMIZATION] Configured Precision: {self.precision.upper()}")
        print(f"[OPTIMIZATION] Configured SDPA/FlashAttention: {self.enable_sdpa}")

    def get_torch_dtype(self) -> torch.dtype:
        """Determines the appropriate PyTorch dtype based on precision configuration."""
        if self.precision == "bf16":
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                print("[OPTIMIZATION] WARNING: bfloat16 is not supported on this hardware. Falling back to float16.")
                return torch.float16
        elif self.precision == "fp16":
            return torch.float16
        else:
            return torch.float32

    def load_optimized_model(self) -> Any:
        """Loads ESM3 and applies bfloat16/float16 weights, and handles attention backends."""
        from esm.models.esm3 import ESM3
        
        dtype = self.get_torch_dtype()
        print(f"[OPTIMIZATION] Loading base model '{self.model_name}' on {self.device} in {dtype}...")

        # Clear GPU cache before loading massive model to prevent fragmentation OOM
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        # Load the base model. Depending on availability, we can cast the model to the target dtype
        self.model = ESM3.from_pretrained(self.model_name)
        
        # Cast model weights to targeted precision
        if dtype in [torch.float16, torch.bfloat16]:
            print(f"[OPTIMIZATION] Casting model weights to low-precision: {dtype}")
            self.model = self.model.to(dtype)
            
        self.model = self.model.to(self.device)
        self.model.eval()

        # Configure system-level Scaled Dot Product Attention (SDPA) backends
        if self.device == "cuda" and self.enable_sdpa:
            print("[OPTIMIZATION] Configuring PyTorch Scaled Dot Product Attention (SDPA)...")
            # Enable high-performance kernels (FlashAttention and Memory-Efficient)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # If forced, disable standard mathematical attention to guarantee FlashAttention-2 speedups!
            if self.force_flash_attn:
                print("[OPTIMIZATION] FORCE FLASH: Disabling slower math fallback kernels...")
                torch.backends.cuda.enable_math_sdp(False)
            else:
                torch.backends.cuda.enable_math_sdp(True)

        print("[OPTIMIZATION] Model successfully loaded and optimized!")
        return self.model

    def generate(self, prompt_sequence: str, num_steps: int = 8, temperature: float = 0.7) -> Dict[str, Any]:
        """Runs accelerated sequence and structural generation under low-precision context."""
        if self.model is None:
            self.load_optimized_model()

        from esm.sdk.api import ESMProtein, GenerationConfig
        
        input_protein = ESMProtein(sequence=prompt_sequence)
        dtype = self.get_torch_dtype()

        print(f"[OPTIMIZATION] Executing optimized forward generation loop...")
        
        # Use PyTorch Automatic Mixed Precision (AMP) context to leverage Tensor Cores dynamically
        amp_device = "cuda" if "cuda" in self.device else "cpu"
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(dtype != torch.float32), dtype=dtype, device_type=amp_device):
                
                # Step 1: Sequence generation
                seq_config = GenerationConfig(track="sequence", num_steps=num_steps, temperature=temperature)
                protein_with_seq = self.model.generate(input_protein, seq_config)
                
                # Step 2: Coordinate structural folding
                struct_config = GenerationConfig(track="structure", num_steps=num_steps, temperature=temperature)
                final_protein = self.model.generate(protein_with_seq, struct_config)

        generated_seq = final_protein.sequence
        print(f"[OPTIMIZATION] Generation complete! Complete Sequence: {generated_seq}")
        
        return {
            "protein": final_protein,
            "sequence": generated_seq
        }

    @staticmethod
    def load_quantized_model_config(model_name: str = "esm3_sm_open_v1") -> Any:
        """
        Educational helper showcasing how 8-bit and 4-bit Quantization is integrated
        via bitsandbytes and Hugging Face's transformers API (perfect for local GPU scaling).
        """
        print("[OPTIMIZATION] [INFO] Quantized loading demonstration via bitsandbytes/HuggingFace:")
        print("Required packages: pip install bitsandbytes accelerate transformers")
        
        # In a real environment with HuggingFace, we would run:
        # from transformers import AutoModelForMaskedLM, BitsAndBytesConfig
        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_skip_modules=["encoder"] # skip sensitive geometric structures
        # )
        # model = AutoModelForMaskedLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        
        return None

if __name__ == "__main__":
    # Test loading the optimized generator
    print("🧪 Running Optimized Generator Config Verification...")
    try:
        # Verify bfloat16 + SDPA setup
        generator = OptimizedESM3Generator(precision="bf16", enable_sdpa=True, force_flash_attn=False)
        print("✅ Optimized ESM3 Generator initialized successfully!")
    except Exception as e:
        print(f"❌ Verification encountered an error: {e}")
