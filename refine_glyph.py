#!/usr/bin/env python3
"""
Generate a shape-varied ס by elastically warping font #10, then cleaning with SDXL.
The elastic warp changes the actual form; SDXL makes it look intentional.
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

MODEL_DIR   = "/mnt/ssd2/cyttic/models/sdxl-refiner-1.0"
INPUT_IMG   = "/mnt/ssd2/cyttic/models/samekh_ref_font10.png"
WARPED_IMG  = "/mnt/ssd2/cyttic/models/samekh_warped.png"
OUTPUT_IMG  = "/mnt/ssd2/cyttic/models/samekh_refined.png"

# ── Elastic warp parameters ───────────────────────────────────────────────────
ALPHA  = 80    # displacement magnitude  — higher = more form change
SIGMA  = 12    # smoothness of the warp  — lower = more local bending
SEED   = 42    # change this to get different variations

STRENGTH = 0.40
STEPS    = 50
GUIDANCE = 6.0

PROMPT = (
    "Hebrew letter samekh, handwritten calligraphy, "
    "open gap at top-left with pointed entry stroke, "
    "angular strokes, dark counter inside, "
    "white letter on dark navy background, high quality"
)
NEGATIVE_PROMPT = (
    "blurry, noisy, deformed, extra strokes, closed circle, "
    "filled blob, low quality, two letters"
)

# ── Apply elastic distortion ──────────────────────────────────────────────────
def elastic_warp(img_arr, alpha, sigma, seed):
    rng = np.random.RandomState(seed)
    h, w = img_arr.shape[:2]
    dx = gaussian_filter(rng.randn(h, w) * alpha, sigma)
    dy = gaussian_filter(rng.randn(h, w) * alpha, sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords_x = np.clip(x + dx, 0, w - 1)
    coords_y = np.clip(y + dy, 0, h - 1)
    warped = np.zeros_like(img_arr)
    for c in range(img_arr.shape[2]):
        warped[..., c] = map_coordinates(img_arr[..., c], [coords_y, coords_x], order=1)
    return warped.astype(np.uint8)

print(f"Warping input  (alpha={ALPHA}, sigma={SIGMA}, seed={SEED})…")
raw = np.array(Image.open(INPUT_IMG).convert("RGB"))
warped_arr = elastic_warp(raw, ALPHA, SIGMA, SEED)
warped_img = Image.fromarray(warped_arr)
warped_img.save(WARPED_IMG)
print(f"Warped image saved → {WARPED_IMG}")

# ── SDXL refiner ─────────────────────────────────────────────────────────────
print("Loading model…")
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")
pipe.enable_attention_slicing()

print(f"Refining warped image  (strength={STRENGTH})…")
result = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    image=warped_img,
    strength=STRENGTH,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
).images[0]

result.save(OUTPUT_IMG)
print(f"Saved → {OUTPUT_IMG}")
