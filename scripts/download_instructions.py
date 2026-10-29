"""
Dataset Download Instructions and Guide.

Provides detailed, step-by-step instructions for obtaining deepfake datasets:
1. KaggleHub programmatic download (with automatic D: drive cache redirection)
2. Kaggle Deepfake Detection Challenge (DFDC) sample videos
3. FaceForensics++ (for genuine unseen manipulation experiments)
4. Kaggle API credential setup without leaking keys
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def print_instructions():
    print("=" * 80)
    print("  DEEPFAKE DETECTION DATASET ACQUISITION GUIDE")
    print("=" * 80)
    print(
        """
Option 1: Programmatic Download via KaggleHub (Automated)
--------------------------------------------------------
Our script 'scripts/download_dataset.py' automatically handles disk space
detection. If C: drive has less than 25GB, it automatically redirects the
cache to D:\\kagglehub_cache where ample storage is available.

Run:
    python scripts/download_dataset.py

Requirements for KaggleHub:
1. An active Kaggle account (https://www.kaggle.com).
2. Download your API credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New Token" under the API section.
   - This downloads a file named 'kaggle.json'.
3. Place 'kaggle.json' in your user directory:
   - Windows: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json
   - Linux/macOS: ~/.kaggle/kaggle.json
   - (Antigravity's .gitignore is configured to NEVER commit this file).


Option 2: Kaggle DFDC Sample Dataset (Manual Web Download)
---------------------------------------------------------
1. Visit the Kaggle Deepfake Detection Challenge page:
   https://www.kaggle.com/competitions/deepfake-detection-challenge/data
2. Download 'train_sample_videos.zip' (approx. 4 GB).
3. Extract the contents directly into:
   data/raw/dfdc_sample/
   Expected contents:
     - 400 MP4 videos (approx. 323 fake, 77 real)
     - metadata.json (maps video filename to {"label": "FAKE"|"REAL", "split": "train", "original": "..."})
4. Run the frame extraction script:
   python scripts/extract_frames.py --input-dir data/raw/dfdc_sample
5. Run the dataset preparation and split script:
   python scripts/prepare_dataset.py


Option 3: FaceForensics++ for Unseen Manipulation Experiments
------------------------------------------------------------
To conduct genuine unseen manipulation research, the dataset must label the
exact generation technique used. FaceForensics++ provides four distinct methods:
  - Deepfakes (DF)
  - Face2Face (F2F)
  - FaceSwap (FS)
  - NeuralTextures (NT)

How to obtain FaceForensics++:
1. Submit the academic access request form at:
   https://github.com/ondyari/FaceForensics
2. Once access is granted, use the official download script:
   python download-FaceForensics.py <output_path> -d all -c c23 -t videos
3. Place videos into manipulation-labeled directories:
   data/raw/faceforensics/original/
   data/raw/faceforensics/Deepfakes/
   data/raw/faceforensics/Face2Face/
   data/raw/faceforensics/FaceSwap/
   data/raw/faceforensics/NeuralTextures/
4. Run:
   python scripts/prepare_dataset.py --dataset-type faceforensics --unseen-method NeuralTextures


Option 4: Synthetic Demo Data (For Immediate Code & Pipeline Testing)
---------------------------------------------------------------------
If you want to immediately test the model architecture, training loop, Streamlit
UI, and test suite on your laptop before waiting for large downloads:
Run:
    python scripts/create_demo_data.py
    python scripts/validate_dataset.py
    python train.py --epochs 2
    streamlit run app.py
"""
    )
    print("=" * 80)


if __name__ == "__main__":
    print_instructions()
