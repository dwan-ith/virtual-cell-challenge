"""Fetch AA sequences for panel genes via UniProt REST and cache for ESM."""
import csv, time, sys
from pathlib import Path
import urllib.request, urllib.parse, json

PANEL = Path("data/vcc2026/pert_counts.csv")
OUT = Path("data/protein/sequences.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

genes = [r[0] for r in csv.reader(open(PANEL)) if r[0] != "target_gene"]
print(f"Fetching {len(genes)} sequences from UniProt...")

seqs = {}
if OUT.exists():
    try:
        seqs = json.loads(OUT.read_text())
        print(f"Loaded {len(seqs)} cached")
    except: pass

for g in genes:
    if g in seqs: continue
    q = f"gene:{g} AND organism_id:9606 AND reviewed:true"
    url = f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(q)}&fields=sequence&format=json&size=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"VCC-RnD/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
            results = data.get("results", [])
            if results and "sequence" in results[0]:
                seq = results[0]["sequence"]["value"]
                seqs[g] = seq[:2000]  # truncate very long
                print(f"  {g}: {len(seq)} aa")
            else:
                # fallback without reviewed filter
                q2 = f"gene:{g} AND organism_id:9606"
                url2 = f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(q2)}&fields=sequence&format=json&size=1"
                req2 = urllib.request.Request(url2, headers={"User-Agent":"VCC-RnD/1.0"})
                with urllib.request.urlopen(req2, timeout=20) as r2:
                    data2 = json.loads(r2.read())
                    results2 = data2.get("results", [])
                    if results2 and "sequence" in results2[0]:
                        seq = results2[0]["sequence"]["value"]
                        seqs[g] = seq[:2000]
                        print(f"  {g} (unreviewed): {len(seq)} aa")
                    else:
                        print(f"  {g}: NOT FOUND")
                        seqs[g] = None
    except Exception as e:
        print(f"  {g} error: {e}")
        seqs[g] = None
    time.sleep(0.4)  # be nice to UniProt
    # checkpoint every 20
    if len([v for v in seqs.values() if v]) % 20 == 0:
        OUT.write_text(json.dumps(seqs, indent=2))
        print(f"  checkpoint {len([v for v in seqs.values() if v])}/{len(genes)}")

OUT.write_text(json.dumps(seqs, indent=2))
found = len([v for v in seqs.values() if v])
print(f"Done: {found}/{len(genes)} found, saved to {OUT}")
