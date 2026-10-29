@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
set OPENBLAS_NUM_THREADS=1
echo ======================================================================
echo Starting Deepfake Detection Web Application (Streamlit)...
echo ======================================================================
python -m streamlit run app.py
pause
