# Deepfake Detection Dataset Guide & Acquisition Protocol

This guide outlines the complete protocol for obtaining, verifying, extracting, and splitting deepfake datasets for training and genuine unseen-manipulation research.

---

## 1. Overview of Supported Datasets

| Dataset | Primary Modality | Total Size | Manipulation Labels | Unseen Generalization Suitable? |
| :--- | :--- | :--- | :--- | :--- |
| **KaggleHub DFD (Entire)** | Videos & Frames | ~22.5 GB | Binary (Real/Fake) | Limited (lacks manipulation family labels) |
| **Kaggle DFDC Sample** | 400 MP4 Videos | ~4.0 GB | Binary (Real/Fake) | Limited (lacks manipulation family labels) |
| **FaceForensics++ (FF++)** | 1,000 Sequences × 5 | ~40 GB (c23) | **DF, F2F, FS, NT** | **Yes (Gold Standard for Unseen Experiments)** |
| **Synthetic Demo Dataset** | JPG Face Crops | ~5 MB | Synthetic Watermarked | Testing only (pipeline & UI validation) |

---

## 2. Programmatic Acquisition via KaggleHub

We provide an automated, storage-aware downloader in `scripts/download_dataset.py`.

```bash
python scripts/download_dataset.py
```

### Storage Safety Feature
Kaggle datasets of 20+ GB can quickly fill up default system drives (`C:\`). Our downloader automatically:
1. Inspects free disk space on `C:\`.
2. If free space is below 25 GB, it checks secondary drives (`D:\`, `E:\`).
3. Automatically sets `KAGGLEHUB_CACHE` to `D:\kagglehub_cache` where ample space is available.
4. Prevents crashes and disk saturation.

### Setting up Kaggle Authentication
If Kaggle reports an authentication or permission error:
1. Log in to [Kaggle](https://www.kaggle.com).
2. Go to **Account Settings** -> **API** section.
3. Click **Create New Token**. A file named `kaggle.json` will download.
4. Move `kaggle.json` to:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`
5. Ensure permissions are private:
   - Linux/macOS: `chmod 600 ~/.kaggle/kaggle.json`

> [!CAUTION]
> Never commit `kaggle.json` to version control. The repository `.gitignore` is pre-configured to ignore `.kaggle/` and `kaggle.json`.

---

## 3. Kaggle DFDC Sample Dataset (Manual Web Download)

If you prefer downloading manually via browser:
1. Visit: [Kaggle Deepfake Detection Challenge Data](https://www.kaggle.com/competitions/deepfake-detection-challenge/data).
2. Download `train_sample_videos.zip` (approx 4.0 GB).
3. Extract the zip into:
   ```
   data/raw/dfdc_sample/
   ```
   Expected directory contents:
   ```
   data/raw/dfdc_sample/
   ├── metadata.json
   ├── aagfhgtpmv.mp4
   ├── abqwwclusr.mp4
   └── ... (400 videos)
   ```

### Validating Metadata
`metadata.json` has the following schema:
```json
{
  "aagfhgtpmv.mp4": {
    "label": "FAKE",
    "split": "train",
    "original": "vduuqbmlyq.mp4"
  },
  "vduuqbmlyq.mp4": {
    "label": "REAL",
    "split": "train"
  }
}
```

### Extracting Facial Frames
Run the configurable extractor:
```bash
python scripts/extract_frames.py --input-dir data/raw/dfdc_sample --max-frames 10 --interval 15
```
Parameters:
- `--max-frames 10`: Limits extraction to 10 frames per video to prevent laptop memory exhaustion.
- `--interval 15`: Samples every 15th frame to prevent near-identical facial expressions.

---

## 4. FaceForensics++ for Genuine Unseen Manipulation Research

To conduct scientifically valid unseen manipulation experiments, each fake video must be tagged with its specific manipulation family:
- **Deepfakes (DF)**: Autoencoder-based facial replacement.
- **Face2Face (F2F)**: Facial reenactment transferring source expressions to target video.
- **FaceSwap (FS)**: Traditional 3D graphics model facial alignment and blending.
- **NeuralTextures (NT)**: Neural rendering modifying facial mouth and expression textures.

### Access Request Protocol
1. Access request form: [FaceForensics GitHub](https://github.com/ondyari/FaceForensics).
2. Download official script: `download-FaceForensics.py`.
3. Download compressed sequences (c23 / lightly compressed):
   ```bash
   python download-FaceForensics.py data/raw/faceforensics -d all -c c23 -t videos
   ```
4. Place into the raw directory:
   ```
   data/raw/faceforensics/
   ├── original/
   ├── Deepfakes/
   ├── Face2Face/
   ├── FaceSwap/
   └── NeuralTextures/
   ```
5. Run dataset preparation with held-out method:
   ```bash
   python scripts/prepare_dataset.py --source-dir data/raw/faceforensics --unseen-method NeuralTextures
   ```

---

## 5. Preventing Data Leakage

A common methodological flaw in deepfake research is **frame-level random splitting**:
- If Frame 10 of Video A is in `train` and Frame 11 of Video A is in `test`, the model memorizes identity, lighting, and background rather than manipulation cues.

### Our Leakage-Free Protocol:
- Partitioning is performed strictly by **Source Video ID** or **Subject Identity**.
- All frames belonging to a source video reside exclusively within one split.
- `scripts/validate_dataset.py` mathematically audits sets for zero intersection:
  $$\text{Sources}_{\text{train}} \cap \text{Sources}_{\text{val}} = \emptyset$$
  $$\text{Sources}_{\text{train}} \cap \text{Sources}_{\text{test}} = \emptyset$$
  $$\text{Sources}_{\text{train}} \cap \text{Sources}_{\text{unseen}} = \emptyset$$

---

## 6. Synthetic Demo Dataset

To enable immediate testing of the entire ML lifecycle without waiting for 20 GB downloads:
```bash
# Generate 120 synthetic marked facial crops
python scripts/create_demo_data.py

# Audit splits
python scripts/validate_dataset.py

# Clean up synthetic data before real training
python scripts/create_demo_data.py --clean
```
All synthetic images contain explicit visual watermarks and manifest tags (`authentic_synthetic`, `synthetic_blend`) and must never be cited as real benchmark results.
