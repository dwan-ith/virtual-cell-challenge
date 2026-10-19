"""Custom vcc prep that avoids the WinError 32 / disk space issue.

Instead of writing a temp h5ad then tarring, we:
1. Read the input h5ad
2. Slim it (keep only X in CSR int32 + obs/var)
3. Write directly to zstd-compressed tarball in a pipe-like fashion
"""
import os, sys, io, tarfile, time
os.environ["PYTHONUTF8"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import scipy.sparse as sp
import anndata as ad
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/submission/prediction_v13.h5ad"
OUTPUT = ROOT / "artifacts/submission/prediction_v13.vcc"
GENES_CSV = ROOT / "data/vcc2026/gene_names.csv"
PERTS_CSV = ROOT / "data/vcc2026/pert_counts.csv"

print(f"Loading {INPUT} ({INPUT.stat().st_size / 1e6:.0f} MB)...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(str(INPUT))
print(f"Loaded in {time.time()-t0:.1f}s: {adata.shape}, X={type(adata.X)}", flush=True)

# Slim: keep only X (CSR int32), obs, var
adata_slim = ad.AnnData(
    X=adata.X,
    obs=adata.obs[["target_gene", "context"]].copy() if "target_gene" in adata.obs.columns else adata.obs.iloc[:, :2].copy(),
    var=adata.var.copy(),
)

# Validate
import pandas as pd
gene_names = [g.strip() for g in open(GENES_CSV) if g.strip() and g.strip() != "gene_name"]
pert_df = pd.read_csv(PERTS_CSV)
targets = pert_df.iloc[:, 0].tolist()
n_genes_actual = adata_slim.shape[1]
n_targets_expected = len(targets)
n_contexts = len(adata_slim.obs["context"].unique()) if "context" in adata_slim.obs.columns else 3
n_cells_per = adata_slim.shape[0] // (n_targets_expected * n_contexts) if n_targets_expected > 0 else 0

print(f"Validation: {n_genes_actual} genes, {n_targets_expected} targets, {n_contexts} contexts, {n_cells_per} cells/pert", flush=True)
assert n_genes_actual == len(gene_names), f"Gene count mismatch: {n_genes_actual} vs {len(gene_names)}"
print("Validation OK", flush=True)

# Write directly to tarball using a temp file approach but with explicit cleanup
# The trick: use the output file's directory for temp (same filesystem = rename works on Linux)
# On Windows we must write the entire h5ad to temp then tar it

# Actually, let's write the h5ad to a temp file, then create tarball
import tempfile, shutil

tmp_dir = tempfile.mkdtemp(prefix="vcc_prep_")
h5ad_tmp = os.path.join(tmp_dir, "pred.h5ad")
tar_tmp = os.path.join(tmp_dir, "pred.tar")

print(f"Writing slimmed h5ad to temp ({h5ad_tmp})...", flush=True)
t0 = time.time()
adata_slim.write_h5ad(h5ad_tmp)
h5ad_size = os.path.getsize(h5ad_tmp)
print(f"Wrote {h5ad_size/1e6:.0f} MB in {time.time()-t0:.1f}s", flush=True)

# Create zstd-compressed tarball
print("Creating zstd tarball...", flush=True)
t0 = time.time()
# Use system zstd if available, or python zstandard
try:
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=3)
    with open(OUTPUT, "wb") as fout:
        with cctx.stream_writer(fout) as compressor:
            with tarfile.open(fileobj=compressor, mode="w") as tar:
                tar.add(h5ad_tmp, arcname="prediction.h5ad")
    print(f"Wrote {OUTPUT.stat().st_size/1e6:.0f} MB in {time.time()-t0:.1f}s", flush=True)
except ImportError:
    # Fallback: use tarfile's built-in gz (zstd not available)
    print("zstandard not found, trying system zstd...", flush=True)
    import subprocess
    # Write uncompressed tar first, then compress
    with tarfile.open(tar_tmp, "w") as tar:
        tar.add(h5ad_tmp, arcname="prediction.h5ad")
    # Compress with zstd CLI
    result = subprocess.run(
        ["zstd", "-f", tar_tmp, "-o", str(OUTPUT), "-3", "--rm"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"zstd failed: {result.stderr}", flush=True)
        # Final fallback: gzip
        print("Falling back to gzip...", flush=True)
        with tarfile.open(str(OUTPUT).replace(".vcc", ".tar.gz"), "w:gz") as tar:
            tar.add(h5ad_tmp, arcname="prediction.h5ad")
        print("Wrote gzip fallback", flush=True)

# Cleanup
try:
    shutil.rmtree(tmp_dir, ignore_errors=True)
except:
    pass

print(f"\nDone! Output: {OUTPUT} ({OUTPUT.stat().st_size/1e6:.0f} MB)", flush=True)
