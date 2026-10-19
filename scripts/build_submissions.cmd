@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
python -u scripts\make_submission.py ^
  --stores artifacts\stores\replogle_gw_k562.npz artifacts\stores\replogle_ess_k562.npz artifacts\stores\replogle_gw_rpe1.npz artifacts\stores\tahoe_hepg2.npz artifacts\stores\tahoe_jurkat.npz artifacts\stores\tahoe_k562.npz artifacts\stores\tahoe_rpe1.npz artifacts\stores\vcc2025_h1.npz ^
  --controls-dir data\vcc2026 ^
  --out artifacts\submission\prediction_v6.h5ad ^
  --config artifacts\loco_config.json ^
  --epochs 200 --seed 0 --decoder poisson >> artifacts\sub_v6.log 2> artifacts\sub_v6.err.log
python -u scripts\make_submission.py ^
  --stores artifacts\stores\replogle_gw_k562.npz artifacts\stores\replogle_ess_k562.npz artifacts\stores\replogle_gw_rpe1.npz artifacts\stores\tahoe_hepg2.npz artifacts\stores\tahoe_jurkat.npz artifacts\stores\tahoe_k562.npz artifacts\stores\tahoe_rpe1.npz artifacts\stores\vcc2025_h1.npz ^
  --controls-dir data\vcc2026 ^
  --out artifacts\submission\prediction_v8.h5ad ^
  --config artifacts\loco_config_v8.json ^
  --epochs 200 --seed 0 --decoder poisson >> artifacts\sub_v8.log 2> artifacts\sub_v8.err.log
