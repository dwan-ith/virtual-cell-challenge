@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
set VCC_TOKEN=vcc_pat_OAidTjJwOLuOWXZIa2TdydMnfci-HWDl3UKoI6WzTsg
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
:retry
vcc submit artifacts\submission\prediction_v8.vcc -m "hybrid-v8-gain3.7" -d "K562-GW transfer prior; shared_mix=0; alpha off; exact-metric-calibrated gain=3.7; probes the gain curve vs v6" --wait >> artifacts\submit_v8.log 2>> artifacts\submit_v8.err.log
findstr /C:"final status: published" artifacts\submit_v8.log >nul
if %errorlevel%==0 goto done
findstr /C:"in progress at a time" artifacts\submit_v8.err.log >nul 2>&1
if %errorlevel%==0 (
  echo [%date% %time%] slot busy; waiting 20 min >> artifacts\submit_v8.log
  timeout /t 1200 /nobreak >nul
  goto retry
)
findstr /C:"Nothing to resume\|daily\|allowance" artifacts\submit_v8.log artifacts\submit_v8.err.log >nul 2>&1
if %errorlevel%==0 (
  echo [%date% %time%] transient refusal; waiting 30 min >> artifacts\submit_v8.log
  timeout /t 1800 /nobreak >nul
  goto retry
)
echo [%date% %time%] unknown failure; stopping >> artifacts\submit_v8.log
goto done
:done
echo [%date% %time%] v8 terminal >> artifacts\submit_v8.log
