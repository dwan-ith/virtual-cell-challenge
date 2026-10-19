"""Probe HF dataset file sizes before downloading."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import urllib.request, json

for repo in ["tahoebio/replogle-nadig-de-rhaister", "Arcticbun/2025_Virtual_Cell_Challenge_Test_Data"]:
    print(f"=== {repo} ===")
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        items = json.load(r)
    for it in items:
        sz = it.get("size", 0)
        print(f"  {sz/1e6:12.2f} MB  {it['path']}")
