@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
python -u scripts\train_eval_loco.py ^
  --stores artifacts\stores\replogle_gw_k562.npz artifacts\stores\replogle_ess_k562.npz artifacts\stores\replogle_gw_rpe1.npz artifacts\stores\tahoe_hepg2.npz artifacts\stores\tahoe_jurkat.npz artifacts\stores\tahoe_k562.npz artifacts\stores\tahoe_rpe1.npz artifacts\stores\vcc2025_h1.npz ^
  --holdout vcc2025/H1 ^
  --out artifacts\loco_v5.json ^
  --config-out artifacts\loco_config.json ^
  --epochs 200 ^
  --max-pairs-per-ctx 48 ^
  --n-eval-cells 180 ^
  --exact-calibration ^
  --model-cache artifacts\loco_model_v5.pt >> artifacts\loco_v5.log 2> artifacts\loco_v5.err.log
