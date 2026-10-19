@echo off
cd /d "C:\Users\aacer\Documents\Virtual Cell Challenge"
curl.exe -sS -L -C - -o data\replogle\K562_gwps_raw_bulk_01.h5ad https://ndownloader.figshare.com/files/35774443
curl.exe -sS -L -C - -o data\replogle\K562_essential_raw_bulk_01.h5ad https://ndownloader.figshare.com/files/35773070
curl.exe -sS -L -C - -o data\replogle\rpe1_raw_bulk_01.h5ad https://ndownloader.figshare.com/files/35775581
echo DONE > artifacts\bulk_dl.done
