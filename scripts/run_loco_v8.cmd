@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
python -u scripts\train_eval_loco.py ^
  --stores artifacts\stores\replogle_gw_k562.npz artifacts\stores\replogle_ess_k562.npz artifacts\stores\replogle_gw_rpe1.npz artifacts\stores\tahoe_hepg2.npz artifacts\stores\tahoe_jurkat.npz artifacts\stores\tahoe_k562.npz artifacts\stores\tahoe_rpe1.npz artifacts\stores\vcc2025_h1.npz ^
  --holdout vcc2025/H1 ^
  --out artifacts\loco_v8.json ^
  --config-out artifacts\loco_config_v8.json ^
  --calib-cache artifacts\loco_calib_v8.json ^
  --fixed-weights 1.0 0.0 0.0 ^
  --shared-mix 0.0 ^
  --no-alpha ^
  --exact-calibration >> artifacts\loco_v8.log 2> artifacts\loco_v8.err.log
