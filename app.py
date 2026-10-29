"""
Streamlit Web Application for Deepfake Detection System.

Provides an interactive user interface for:
1. Home: Overview, objectives, architecture, technology stack, and limitations.
2. Image Prediction: Upload facial images, OpenCV face detection, and model classification.
3. Model Evaluation: View actual training history curves, confusion matrices, and metrics.
4. About & Research: Deepfake mechanics, transfer learning theory, unseen manipulation challenges, and ethics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import config
from src.preprocessing import preprocess_for_inference
from src.utils import load_json

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="Deepfake Detection | MobileNetV2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished academic UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-real {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .badge-fake {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar():
    """Render sidebar navigation and project metadata."""
    st.sidebar.title("🛡️ Deepfake Detector")
    st.sidebar.markdown("**TensorFlow & MobileNetV2**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Image Prediction", "Model Evaluation", "About & Research"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")

    model_exists = config.BEST_MODEL_PATH.exists() or config.BACKUP_MODEL_PATH.exists()
    if model_exists:
        st.sidebar.success("✅ Model Available (.keras)")
    else:
        st.sidebar.warning("⚠️ Model Not Trained Yet")

    manifest_exists = config.DATASET_MANIFEST_PATH.exists()
    if manifest_exists:
        st.sidebar.success("✅ Dataset Manifest Ready")
    else:
        st.sidebar.info("ℹ️ Dataset Not Prepared")

    st.sidebar.markdown("---")
    st.sidebar.caption("Antigravity Academic ML Project | Version 1.0")
    return page


def page_home():
    """Render Home landing page."""
    st.markdown('<div class="main-header">Deep Learning-Based Deepfake Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evaluating Facial Manipulation Generalization Under Unseen Manipulation Techniques</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📌 Executive Summary")
        st.write(
            """
            This project investigates the resilience of convolutional neural networks in detecting
            deepfake facial manipulations. While modern deep learning models achieve high accuracy
            when evaluated on the manipulation techniques they were trained on (such as Face2Face or Deepfakes),
            their performance often degrades significantly when encountering **unseen manipulation techniques**
            (e.g., NeuralTextures, diffusion-based synthesis, or novel face swapping).

            Using **MobileNetV2** transfer learning pretrained on ImageNet, we construct an end-to-end
            detection system optimized for lightweight, CPU-friendly laptop execution.
            """
        )

        st.markdown("### 🎯 Core Objectives")
        st.markdown(
            """
            1. **Automated Facial Preprocessing:** Detect and extract facial bounding boxes from frames using OpenCV.
            2. **Transfer Learning Pipeline:** Leverage ImageNet representations with a frozen MobileNetV2 feature extractor and custom classification head.
            3. **Leakage-Free Splitting:** Enforce strict source-video independence across training, validation, and testing.
            4. **Dual Evaluation Framework:** Maintain strict separation between standard testing (known techniques) and unseen manipulation testing.
            5. **Practical Deployment:** Provide both command-line scripts and an accessible Streamlit graphical interface.
            """
        )

    with col2:
        st.markdown("### ⚙️ Technology Stack")
        st.markdown(
            """
            - **Framework:** TensorFlow 2.x & Keras 3.x
            - **Architecture:** MobileNetV2 (Pretrained on ImageNet)
            - **Computer Vision:** OpenCV (Haar Cascade detection) & Pillow
            - **Data Handling:** NumPy, Pandas, and tf.data pipelines
            - **Metrics:** Scikit-Learn (Accuracy, Precision, Recall, F1, ROC-AUC)
            - **Visualization:** Matplotlib
            - **UI Framework:** Streamlit
            - **Dataset Management:** KaggleHub
            """
        )

        st.info(
            "💡 **CPU Friendly:** Designed with configurable batch sizes (16) and frame limits (10/vid) "
            "to run comfortably on ordinary student laptops without GPU acceleration."
        )

    st.markdown("---")
    st.markdown("### 🔄 End-to-End System Pipeline")

    st.markdown(
        """
        ```
        [Source Video / Image] ──> [OpenCV Face Detector] ──> [Face Crop & Padding]
                                                                     │
                                                                     ▼
        [Predicted Probability] <── [Dense Head (Sigmoid)] <── [MobileNetV2 Backbone]
                 │
                 ▼
        [REAL vs FAKE + Confidence]
        ```
        """
    )


def page_predict():
    """Render Image Prediction interface."""
    st.markdown("## 🔍 Deepfake Image Prediction")
    st.write("Upload a facial photograph or video frame to assess whether it exhibits signs of manipulation.")

    model_path = config.BEST_MODEL_PATH if config.BEST_MODEL_PATH.exists() else config.BACKUP_MODEL_PATH
    if not model_path.exists():
        st.warning(
            "⚠️ **No trained model was found.**\n\n"
            "To train the model before running inference:\n"
            "1. Download or generate data: `python scripts/create_demo_data.py` (or `python scripts/download_dataset.py`)\n"
            "2. Train the model: `python train.py --epochs 5`\n\n"
            "Once training finishes, return to this tab to test images."
        )
        return

    # Load model
    @st.cache_resource
    def get_model():
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is not installed in the currently running Python environment.\n\n"
                "**Solution:** Run Streamlit with the Python interpreter where dependencies are installed:\n"
                "```powershell\n"
                "python -m streamlit run app.py\n"
                "```"
            )
        return tf.keras.models.load_model(model_path)

    try:
        model = get_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image...",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        use_crop = st.checkbox("Enable Automatic OpenCV Face Detection", value=True)
    with col_opt2:
        threshold = st.slider("Classification Threshold (FAKE decision boundary)", 0.1, 0.9, float(config.CLASSIFICATION_THRESHOLD), 0.05)

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            pil_image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Input Image")
                st.image(pil_image, use_container_width=True)

            # Preprocess
            batch_tensor, cropped_np, face_detected = preprocess_for_inference(
                image_bytes, crop_face=use_crop
            )

            with col2:
                st.markdown("#### Model Input (Cropped & Resized to 224×224)")
                st.image(cropped_np, use_container_width=True)
                if face_detected:
                    st.caption("✅ Face detected and cropped with margin padding.")
                else:
                    st.caption("ℹ️ No distinct face detected; full image resized.")

            # Predict
            with st.spinner("Analyzing image features..."):
                raw_score = float(model.predict(batch_tensor, verbose=0)[0][0])

            is_fake = raw_score >= threshold
            predicted_label = "FAKE" if is_fake else "REAL"
            confidence = raw_score if is_fake else (1.0 - raw_score)

            st.markdown("---")
            st.markdown("### 📊 Classification Result")

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                if is_fake:
                    st.markdown('<div class="badge-fake">🚨 PREDICTION: FAKE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-real">🛡️ PREDICTION: REAL</div>', unsafe_allow_html=True)

            with res_col2:
                st.metric("Model Confidence", f"{confidence * 100:.2f}%")

            with res_col3:
                st.metric("Raw Sigmoid Score", f"{raw_score:.4f}", help="Output range [0, 1]. Values >= threshold are classified as FAKE.")

            st.progress(raw_score)
            st.caption(f"Scale: 0.0 (Authentic REAL) ──────── [Threshold: {threshold:.2f}] ──────── 1.0 (Manipulated FAKE)")

            st.info(
                "📌 **Advisory Note:** Neural network output scores represent feature distance relative to training data, "
                "not calibrated forensic proof. A prediction should always be corroborated with contextual metadata and professional forensics."
            )

        except Exception as e:
            st.error(f"Error processing image: {e}")


def page_evaluation():
    """Render Evaluation Results & Plots."""
    st.markdown("## 📈 Model Performance & Evaluation")
    st.write("Review empirical validation and testing metrics generated from actual script execution.")

    tab1, tab2, tab3 = st.tabs(["Standard Test Set", "Unseen Manipulation Test", "Training Curves"])

    with tab1:
        st.markdown("### 📊 Standard Test Set Performance")
        if config.EVALUATION_REPORT_PATH.exists():
            report = load_json(config.EVALUATION_REPORT_PATH)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{report.get('accuracy', 0) * 100:.2f}%")
            m2.metric("Precision", f"{report.get('precision', 0) * 100:.2f}%")
            m3.metric("Recall", f"{report.get('recall', 0) * 100:.2f}%")
            m4.metric("F1-Score", f"{report.get('f1_score', 0):.4f}")
            m5.metric("ROC-AUC", f"{report.get('roc_auc', 'N/A')}")

            st.markdown("---")
            col_cm, col_roc = st.columns(2)
            with col_cm:
                cm_img_path = config.PLOTS_DIR / "confusion_matrix_standard_test.png"
                if cm_img_path.exists():
                    st.image(str(cm_img_path), caption="Confusion Matrix (Standard Test Set)")
                else:
                    st.info("Confusion matrix plot not found.")

            with col_roc:
                roc_img_path = config.PLOTS_DIR / "roc_curve_standard_test.png"
                if roc_img_path.exists():
                    st.image(str(roc_img_path), caption="ROC Curve (Standard Test Set)")
                else:
                    st.info("ROC Curve plot not found.")
        else:
            st.info(
                "ℹ️ Standard test evaluation has not been run yet.\n\n"
                "Execute the evaluation script via: `python evaluate.py`"
            )

    with tab2:
        st.markdown("### 🔬 Unseen Manipulation Generalization")
        if config.UNSEEN_REPORT_PATH.exists():
            unseen_rep = load_json(config.UNSEEN_REPORT_PATH)
            if unseen_rep.get("status") == "pending":
                st.warning(f"⚠️ **{unseen_rep.get('message')}**")
                st.markdown(
                    """
                    **Why is this pending?**
                    In honest academic research, unseen manipulation evaluation requires a dataset where specific
                    manipulation families (e.g. Deepfakes vs FaceSwap vs NeuralTextures) are explicitly identified.
                    Randomly splitting an unlabeled sample does not constitute a genuine unseen-manipulation experiment.
                    See the `DATASET_GUIDE.md` for instructions on loading FaceForensics++.
                    """
                )
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{unseen_rep.get('accuracy', 0) * 100:.2f}%")
                m2.metric("Precision", f"{unseen_rep.get('precision', 0) * 100:.2f}%")
                m3.metric("Recall", f"{unseen_rep.get('recall', 0) * 100:.2f}%")
                m4.metric("F1-Score", f"{unseen_rep.get('f1_score', 0):.4f}")

                cm_unseen = config.PLOTS_DIR / "confusion_matrix_unseen_test.png"
                if cm_unseen.exists():
                    st.image(str(cm_unseen), caption="Confusion Matrix (Unseen Manipulation)")
        else:
            st.info("ℹ️ Unseen test report not found. Run `python evaluate.py` to evaluate.")

    with tab3:
        st.markdown("### 📉 Training & Validation Curves")
        curves_path = config.PLOTS_DIR / "training_curves.png"
        if curves_path.exists():
            st.image(str(curves_path), caption="Accuracy and Binary Cross-Entropy Loss Over Epochs")
        else:
            st.info("Training curves plot not found. Run `python train.py` to train the model and generate curves.")


def page_about():
    """Render About & Academic Information."""
    st.markdown("## 📚 Academic Background & Theoretical Principles")

    st.markdown("### 🧠 1. Transfer Learning with MobileNetV2")
    st.write(
        """
        MobileNetV2 utilizes **inverted residual blocks** with linear bottlenecks and depthwise separable
        convolutions. By pretraining on ImageNet (1.4 million images across 1,000 object categories),
        the early and middle convolutional layers learn general visual primitives such as edge detectors,
        color gradients, and texture boundaries.

        In our implementation:
        1. We remove the 1000-class ImageNet classification head.
        2. We freeze the feature extractor to retain foundational weights without catastrophic forgetting.
        3. We attach a GlobalAveragePooling2D layer, a dropout layer (0.35 rate) for regularization, and a single sigmoid output unit.
        4. During Phase 2 fine-tuning, the upper 30 convolutional layers are unfrozen with a reduced learning rate ($10^{-5}$) to specialize on subtle facial manipulation artifacts.
        """
    )

    st.markdown("### 🔒 2. Preventing Data Leakage in Video Datasets")
    st.write(
        """
        A critical pitfall in published deepfake literature is **frame-level random splitting**.
        Because consecutive video frames share nearly identical lighting, facial expressions, background,
        and subject identity, assigning frame $t$ to the training set and frame $t+1$ to the test set
        causes acute data leakage. The model memorizes identity and lighting rather than manipulation artifacts.

        **Our Solution:** The pipeline strictly groups frames by **source video identifier** (and subject identity),
        guaranteeing that zero frames from any training video ever leak into validation or testing sets.
        """
    )

    st.markdown("### ⚠️ 3. Limitations of Deepfake Classifiers")
    st.write(
        """
        - **Domain Shift:** Models trained on compression level C23 often struggle on highly compressed social media videos (C40).
        - **Adversarial Robustness:** Simple post-processing (Gaussian blur, color jittering, re-encoding) can flip model predictions.
        - **Ethical Integrity:** Predictions generated by this system are heuristic estimates and must never be used as sole legal evidence.
        """
    )


def main():
    config.init_directories()
    page = render_sidebar()

    if page == "Home":
        page_home()
    elif page == "Image Prediction":
        page_predict()
    elif page == "Model Evaluation":
        page_evaluation()
    elif page == "About & Research":
        page_about()


if __name__ == "__main__":
    main()
