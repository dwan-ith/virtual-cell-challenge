"""Build the exact H1 store from the 2025 challenge test-set mirror.

data/hf/adata_Test.h5ad: 170,846 cells x 18,080 genes (raw counts), obs =
target_gene / guide_id / batch. Controls: 38k non-targeting cells. This is
REAL Flex-protocol truth -> our most valuable exact training context and the
honest LOCO holdout.

Output: artifacts/stores/vcc2025_h1.npz (+ __controls.npz) on the official
18,533-gene axis.
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.dataset import build_pseudobulk_store, load_gene_names_csv

GENE_CSV = Path("data/vcc2026/gene_names.csv")
TEST_H5AD = Path("data/hf/adata_Test.h5ad")
OUT = Path("artifacts/stores")

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gene_axis = load_gene_names_csv(GENE_CSV)
    print(f"[h1] official axis: {len(gene_axis)} genes", flush=True)

    t0 = time.time()
    store = build_pseudobulk_store(
        sources=[{
            "path": str(TEST_H5AD),
            "name": "vcc2025/H1",
            "target_col": "target_gene",
            "control_label": "non-targeting",
            "force_single_context": True,   # pool all Flex batches: ONE cell line
        }],
        gene_names=gene_axis,
        min_cells=60,
        max_control_cells=0,                # controls come from build_h1_controls.py
        chunk_rows=4096,
    )
    out_path = OUT / "vcc2025_h1.npz"
    store.save(out_path)
    n_ctrl_ctx = {c: m.shape[0] for c, m in store.controls.items()}
    print(f"[h1] pairs={len(store.pairs)} contexts={sorted(store.basal)} "
          f"controls={n_ctrl_ctx} -> {out_path} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
