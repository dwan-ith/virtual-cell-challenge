@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
set VCC_TOKEN=vcc_pat_OAidTjJwOLuOWXZIa2TdydMnfci-HWDl3UKoI6WzTsg
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
vcc submit artifacts\submission\prediction_v9.vcc -m "hybrid-v9-gain2.6-q0.7" -d "K562-GW transfer prior; shared_mix=0; alpha off; gain=2.6 with per-target quantile floor 0.7; midpoint of the measured gain curve" --wait >> artifacts\submit_v9.log 2> artifacts\submit_v9.err.log
