@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
set VCC_TOKEN=vcc_pat_OAidTjJwOLuOWXZIa2TdydMnfci-HWDl3UKoI6WzTsg
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
vcc submit artifacts\submission\prediction_v8.vcc -m "hybrid-v8-gain3.7" -d "K562-GW transfer prior; shared_mix=0; alpha off; exact-metric-calibrated gain=3.7; probes the gain curve vs v6" --resume --wait >> artifacts\submit_v8.log 2> artifacts\submit_v8.err.log
