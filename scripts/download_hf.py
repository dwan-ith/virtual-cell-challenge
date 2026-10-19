"""Download VCC training data from HuggingFace with resume support.

Priority order:
  1. tahoebio/replogle-nadig-de-rhaister: definition metadata + cell_eval deltas
     (pseudobulk responses for hepg2/jurkat/k562/rpe1, ~2000 knockdowns)
  2. Arcticbun/2025_Virtual_Cell_Challenge_Test_Data: the 2025 H1 held-out
     test set (real Flex-protocol truth -> honest LOCO context)
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from huggingface_hub import hf_hub_download

OUT = Path("data/hf")
OUT.mkdir(parents=True, exist_ok=True)

jobs = [
    ("tahoebio/replogle-nadig-de-rhaister", "definition/feature_names.json", "dataset"),
    ("tahoebio/replogle-nadig-de-rhaister", "definition/dataset.toml", "dataset"),
    ("tahoebio/replogle-nadig-de-rhaister", "cell_eval/all_delta.parquet", "dataset"),
    ("Arcticbun/2025_Virtual_Cell_Challenge_Test_Data", "adata_Test.h5ad", "dataset"),
    ("Arcticbun/2025_Virtual_Cell_Challenge_Test_Data", "pert_counts_Test.csv", "dataset"),
]

for repo, fname, rtype in jobs:
    t0 = time.time()
    print(f"[dl] {repo} :: {fname} ...", flush=True)
    try:
        p = hf_hub_download(
            repo_id=repo, filename=fname, repo_type=rtype,
            local_dir=str(OUT), resume_download=True,
        )
        sz = Path(p).stat().st_size / 1e6
        print(f"[dl] done {fname} ({sz:.1f} MB in {time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"[dl] FAILED {repo}/{fname}: {e}", flush=True)

print("[dl] ALL DONE", flush=True)
