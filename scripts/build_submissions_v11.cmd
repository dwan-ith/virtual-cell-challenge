@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
python -u scripts\make_submission.py ^
  --stores artifacts\stores\replogle_gw_k562.npz artifacts\stores\replogle_ess_k562.npz artifacts\stores\replogle_gw_rpe1.npz artifacts\stores\tahoe_hepg2.npz artifacts\stores\tahoe_jurkat.npz artifacts\stores\tahoe_k562.npz artifacts\stores\tahoe_rpe1.npz artifacts\stores\vcc2025_h1.npz ^
  --controls-dir data\vcc2026 ^
  --out artifacts\submission\prediction_v11.h5ad ^
  --config artifacts\loco_config_v11.json ^
  --epochs 200 --seed 0 --decoder poisson >> artifacts\sub_v11.log 2> artifacts\sub_v11.err.log
python scripts\h5ad_to_f32.py artifacts\submission\prediction_v11.h5ad >> artifacts\sub_v11.log 2>> artifacts\sub_v11.err.log
$env:PYTHONUTF8='1'
vcc prep artifacts\submission\prediction_v11.h5ad -g data\vcc2026\gene_names.csv --perts data\vcc2026\pert_counts.csv -o artifacts\submission\prediction_v11.vcc --force >> artifacts\sub_v11.log 2>> artifacts\sub_v11.err.log
