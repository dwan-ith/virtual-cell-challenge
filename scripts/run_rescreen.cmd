@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
python -u scripts\rescreen.py --preds artifacts\loco_v8.preds.npz --store artifacts\stores\vcc2025_h1.npz >> artifacts\rescreen.log 2> artifacts\rescreen.err.log
