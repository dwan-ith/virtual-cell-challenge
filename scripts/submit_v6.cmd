@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
set VCC_TOKEN=vcc_pat_OAidTjJwOLuOWXZIa2TdydMnfci-HWDl3UKoI6WzTsg
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
vcc submit artifacts\submission\prediction_v6.vcc -m "hybrid-v6-gain1.8" -d "K562-GW transfer prior; shared_mix=0; alpha on; exact-metric-calibrated gain=1.8; LOCO(H1) honest eval" --wait >> artifacts\submit_v6.log 2> artifacts\submit_v6.err.log
