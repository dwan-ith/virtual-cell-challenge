@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
set PYTHONUTF8=1
set HF_HUB_ENABLE_HF_TRANSFER=0
python -u scripts\download_state_models.py >> artifacts\state_dl.log 2> artifacts\state_dl.err.log
