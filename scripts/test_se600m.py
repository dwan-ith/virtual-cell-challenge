"""Test SE-600M embedding with manual safetensors loading."""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import numpy as np
from safetensors.torch import load_file

print("Loading model weights from safetensors...", flush=True)
weights = load_file("data/state/se600m/model.safetensors")
print(f"Number of tensors: {len(weights)}")
print("Key prefixes:", set(k.split(".")[0] for k in weights.keys()))
for k in sorted(weights.keys())[:10]:
    print(f"  {k}: {weights[k].shape}")
print("...")
for k in sorted(weights.keys())[-10:]:
    print(f"  {k}: {weights[k].shape}")
