import os
import torch
from diffusers import AutoencoderKL, DiffusionPipeline

SDXL_MODEL_CACHE = "./sdxl-cache"
VAE_CACHE = "./vae-cache"

better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

# SDXL CACHE CHECKER
if not os.path.exists(SDXL_MODEL_CACHE):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=better_vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.save_pretrained(SDXL_MODEL_CACHE, safe_serialization=True)
