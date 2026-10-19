"""Generate ESM2 embeddings for panel genes. Uses facebook/esm2_t6_8M (31MB) for fast R&D; swap to t33_650M for final."""
import json
from pathlib import Path
import torch
import numpy as np

SEQ_JSON = Path("data/protein/sequences.json")
OUT = Path("data/protein/esm2_t6_8m_panel.npz")
OUT.parent.mkdir(parents=True, exist_ok=True)

genes = json.loads(SeqJson := open(SEQ_JSON).read()) if SEQ_JSON else {}
# Actually need to read correctly
import json as _j
seqs = _j.loads(Path("data/protein/sequences.json").read_text())
# Filter found
items = [(g,s) for g,s in seqs.items() if s]
print(f"Generating ESM for {len(items)} genes...")
from transformers import AutoTokenizer, AutoModel

model_name = "facebook/esm2_t6_8M_UR50D"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Device: {device}")

embeddings = {}
import tqdm
for gene, seq in items:
    # ESM truncates to 1022 + 2 specials
    seq = seq[:1022]
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        # mean pool over sequence length (excluding specials)
        hidden = out.last_hidden_state[0]  # L x D
        # remove BOS/EOS if present (first and last)
        if hidden.shape[0] > 2:
            hidden = hidden[1:-1]
        emb = hidden.mean(dim=0).cpu().numpy()
        embeddings[gene] = emb
    if len(embeddings) % 20 == 0:
        print(f"  {len(embeddings)}/{len(items)} {gene} {emb.shape}")

# Save
genes_list = list(embeddings.keys())
mat = np.stack([embeddings[g] for g in genes_list])
np.savez_compressed(OUT, gene_names=np.array(genes_list), embeddings=mat)
print(f"Saved {OUT}: {mat.shape} ({OUT.stat().st_size/1e6:.1f} MB)")

# Also compute per-gene gain map for quick test
from vcc_model.protein import build_per_gene_gain_map
# Use dummy consistency for now
cons_dummy = {g: 0.45 for g in genes_list}
pg = build_per_gene_gain_map({k: np.array([0.45]) for k in genes_list}, genes_list, default_cons=0.45, scale=0.5)
print(f"Per-gene gain example: {list(pg.items())[:3]}")
