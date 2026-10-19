"""List Replogle figshare article files (sizes + URLs)."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import urllib.request, json

with urllib.request.urlopen("https://api.figshare.com/v2/articles/20029387", timeout=60) as r:
    art = json.load(r)
for f in sorted(art["files"], key=lambda x: x["size"]):
    gb = f["size"] / 1e9
    if gb < 40:
        print(f"{gb:8.2f} GB  {f['name']}\n            {f['download_url']}")
