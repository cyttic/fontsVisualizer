#!/usr/bin/env python3
"""
Download stabilityai/stable-diffusion-xl-refiner-1.0 to /mnt/ssd2/cyttic/models
"""

import subprocess
import sys

def ensure_deps():
    for pkg in ("huggingface_hub",):
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"])

ensure_deps()

from huggingface_hub import snapshot_download
import os

MODEL_ID   = "stabilityai/stable-diffusion-xl-refiner-1.0"
LOCAL_DIR  = "/mnt/ssd2/cyttic/models/sdxl-refiner-1.0"

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"Downloading {MODEL_ID}")
print(f"Destination: {LOCAL_DIR}")
print("This is ~6 GB — grab a coffee.\n")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # skip non-PyTorch weights
)

print(f"\nDone. Model saved to {LOCAL_DIR}")
