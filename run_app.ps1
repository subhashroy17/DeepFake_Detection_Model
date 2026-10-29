$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:OPENBLAS_NUM_THREADS = "1"
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting Deepfake Detection Web Application (Streamlit)..." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
python -m streamlit run app.py
