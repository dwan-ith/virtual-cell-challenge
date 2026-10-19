@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
python -u scripts\train_eval_loco.py ^
  --stores artifacts\stores\replogle_gw_k562.npz artifacts\stores\replogle_ess_k562.npz artifacts\stores\replogle_gw_rpe1.npz artifacts\stores\tahoe_hepg2.npz artifacts\stores\tahoe_jurkat.npz artifacts\stores\tahoe_k562.npz artifacts\stores\tahoe_rpe1.npz artifacts\stores\vcc2025_h1.npz ^
  --holdout replogle_gw/RPE1 ^
  --out artifacts\loco_v11.json ^
  --config-out artifacts\loco_config_v11.json ^
  --calib-cache artifacts\loco_calib_v11.json ^
  --fixed-weights 1.0 0.0 0.0 ^
  --shared-mix 0.0 ^
  --no-alpha ^
  --basal-gate-cpm 5.0 ^
  --james-stein ^
  --exact-calibration >> artifacts\loco_v11.log 2> artifacts\loco_v11.err.log
