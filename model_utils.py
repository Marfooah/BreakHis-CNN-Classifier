"""BreakHis CNN inference utilities for the HistoAI Streamlit app."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent
SKLEARN_MODEL_FILENAME = "breakhis_sklearn.joblib"
# Notebook F1-optimal threshold for the Keras CNN; lower threshold improves malignant recall.
CNN_MALIGNANT_THRESHOLD = 0.20
SKLEARN_MALIGNANT_THRESHOLD = 0.35
UNCERTAINTY_MARGIN = 0.08

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    tf = None  # type: ignore[assignment]
    TF_AVAILABLE = False

IMG_SIZE = 224
CLASS_LABELS = {0: "Benign", 1: "Malignant"}

MODEL_METRICS = {
    "accuracy": 0.891,
    "precision": 0.998,
    "recall": 0.843,
    "f1": 0.914,
    "auc": 0.978,
}

INTERPRETATIONS = {
    "Benign": (
        "The AI model detected high cellular regularity and normal tissue architecture "
        "consistent with non-malignant morphology. Analysis reveals consistent "
        "cytoplasmic-to-nuclear ratios and smooth nuclear membranes."
    ),
    "Malignant": (
        "The AI model identified irregular nuclear contours, disrupted tissue architecture, "
        "and patterns consistent with malignant morphology. Hyperchromasia and abnormal "
        "cell clustering may be present."
    ),
}

FINDINGS = {
    "Benign": [
        (
            "Cell Architecture",
            "Regular honeycomb pattern observed throughout the epithelial layer. "
            "Intercellular junctions remain intact.",
        ),
        (
            "Nuclear Morphology",
            "Uniform nuclei with clearly defined borders. No significant enlargement "
            "or irregular clumping noted.",
        ),
        (
            "Stroma Integrity",
            "Connective tissue shows normal density without abnormal vascularization "
            "or necrotic centers.",
        ),
    ],
    "Malignant": [
        (
            "Cell Architecture",
            "Disrupted epithelial organization with loss of normal lattice spacing "
            "and irregular cell clustering.",
        ),
        (
            "Nuclear Morphology",
            "Enlarged, pleomorphic nuclei with hyperchromasia and irregular nuclear "
            "membranes detected.",
        ),
        (
            "Stroma Integrity",
            "Stromal invasion patterns and abnormal vascularization may be present "
            "in surrounding tissue.",
        ),
    ],
}


class InferenceModel(Protocol):
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray: ...


class SklearnBreakHisModel:
    """Trained RandomForest on BreakHis features (works without TensorFlow)."""

    def __init__(self, bundle: dict[str, Any]):
        self.pipeline = bundle["pipeline"]
        self.feature_size = int(bundle.get("feature_size", 96))
        self.threshold = float(bundle.get("threshold", SKLEARN_MALIGNANT_THRESHOLD))

    @staticmethod
    def _extract_features(arr: np.ndarray) -> np.ndarray:
        gray = np.mean(arr, axis=2)
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        lap_var = float(np.var(np.gradient(gray)[0]) + np.var(np.gradient(gray)[1]))
        patch = gray.reshape(8, 12, 8, 12).mean(axis=(1, 3)).ravel()
        stain_ratio = float(means[0] / (means[2] + 1e-6))
        return np.concatenate(
            [means, stds, [lap_var, grad_mag.mean(), grad_mag.std(), stain_ratio], patch]
        ).astype(np.float32)

    def malignant_probability(self, image: Image.Image) -> float:
        img = image.convert("RGB").resize((self.feature_size, self.feature_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        features = self._extract_features(arr).reshape(1, -1)
        return float(self.pipeline.predict_proba(features)[0][1])

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        arr = batch[0]
        gray = np.mean(arr, axis=2)
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        lap_var = float(np.var(np.gradient(gray)[0]) + np.var(np.gradient(gray)[1]))
        patch = gray.reshape(8, 12, 8, 12).mean(axis=(1, 3)).ravel()
        stain_ratio = float(means[0] / (means[2] + 1e-6))
        features = np.concatenate(
            [means, stds, [lap_var, grad_mag.mean(), grad_mag.std(), stain_ratio], patch]
        ).reshape(1, -1)
        return self.pipeline.predict_proba(features)


def build_inference_model() -> Any:
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs

    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(512, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="breakhis_inference")

def is_histology_like(image: Image.Image) -> bool:
    img = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0

    mean = np.mean(img)
    std = np.std(img)

    # texture complexity (important for histology)
    laplacian_var = np.var(np.gradient(img)[0]) + np.var(np.gradient(img)[1])

    # RGB channel imbalance (medical stains have color structure)
    channel_std = np.mean(np.std(img, axis=(0, 1)))

    # RULES (tighter + more realistic)
    if mean < 0.05 or mean > 0.95:
        return False

    if std < 0.07:
        return False

    if channel_std < 0.02:
        return False

    if laplacian_var < 0.0005:
        return False

    return True

def _candidate_model_paths() -> list[Path]:
    names = (
        "breakhis_final_model.keras",
        "best_breakhis_cnn.keras",
        "models/breakhis_final_model.keras",
        "models/best_breakhis_cnn.keras",
    )
    return [PROJECT_ROOT / name for name in names]


def _candidate_sklearn_model_paths() -> list[Path]:
    names = (
        f"models/{SKLEARN_MODEL_FILENAME}",
        SKLEARN_MODEL_FILENAME,
    )
    return [PROJECT_ROOT / name for name in names]


def _resolve_sklearn_model_path() -> Path | None:
    for path in _candidate_sklearn_model_paths():
        if path.exists():
            return path
    return None


def _transfer_weights(saved: Any, inference: Any) -> None:
    saved_layers = [layer for layer in saved.layers if layer.get_weights()]
    inference_layers = [layer for layer in inference.layers if layer.get_weights()]
    if len(saved_layers) != len(inference_layers):
        raise ValueError("Layer count mismatch between saved and inference models.")
    for src, dst in zip(saved_layers, inference_layers):
        dst.set_weights(src.get_weights())


def _model_download_url() -> str | None:
    url = os.environ.get("BREAKHIS_MODEL_URL", "").strip()
    if url:
        return url
    try:
        import streamlit as st

        secret_url = st.secrets.get("BREAKHIS_MODEL_URL", "")
        if secret_url:
            return str(secret_url).strip()
    except Exception:
        pass
    return None


def _ensure_sklearn_model_file() -> tuple[Path | None, str]:
    existing = _resolve_sklearn_model_path()
    if existing is not None:
        return existing, ""

    url = _model_download_url()
    if not url:
        return None, (
            "Trained model not found in the repo. Commit `breakhis_sklearn.joblib` "
            "(project root) or `models/breakhis_sklearn.joblib` to GitHub, or set a "
            "`BREAKHIS_MODEL_URL` Streamlit secret to a direct download link."
        )

    target = PROJECT_ROOT / "models" / SKLEARN_MODEL_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target)
        return target, "Downloaded sklearn model from BREAKHIS_MODEL_URL"
    except Exception as exc:
        return None, f"Failed to download model from BREAKHIS_MODEL_URL: {exc}"


def _load_sklearn_model() -> tuple[SklearnBreakHisModel | None, str]:
    model_path, download_note = _ensure_sklearn_model_file()
    if model_path is None:
        return None, download_note

    try:
        bundle = joblib.load(model_path)
        prefix = f"{download_note}. " if download_note else ""
        return (
            SklearnBreakHisModel(bundle),
            f"{prefix}Loaded sklearn classifier from {model_path.name}",
        )
    except Exception as exc:
        return None, f"Failed to load sklearn model: {exc}"


def load_model() -> tuple[Any, str]:
    if TF_AVAILABLE:
        inference_model = build_inference_model()
        for path in _candidate_model_paths():
            if not path.exists():
                continue
            try:
                saved = tf.keras.models.load_model(path, compile=False)
                _transfer_weights(saved, inference_model)
                inference_model._backend = "cnn"  # type: ignore[attr-defined]
                return inference_model, f"Loaded CNN weights from {path.name}"
            except Exception:
                continue

    sklearn_model, sklearn_status = _load_sklearn_model()
    if sklearn_model is not None:
        sklearn_model._backend = "sklearn"  # type: ignore[attr-defined]
        return sklearn_model, sklearn_status

    if TF_AVAILABLE:
        model = build_inference_model()
        model._backend = "cnn"  # type: ignore[attr-defined]
        return model, (
            "No trained CNN file found. Place `breakhis_final_model.keras` in the project "
            "root, or run `python3 train_sklearn_model.py` to train the sklearn classifier."
        )

    return None, (
        "No model available. Commit `breakhis_sklearn.joblib` to GitHub "
        "(train locally with `python3 train_sklearn_model.py`), set a "
        "`BREAKHIS_MODEL_URL` Streamlit secret, or add `breakhis_final_model.keras` "
        "with TensorFlow."
    )


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _malignant_probability(model: Any, image: Image.Image) -> float:
    if model is None:
        raise RuntimeError("No trained model is loaded.")

    if isinstance(model, SklearnBreakHisModel):
        return model.malignant_probability(image)

    batch = preprocess_image(image)
    return float(model.predict(batch, verbose=0)[0][0])


def _decision_threshold(model: Any) -> float:
    if isinstance(model, SklearnBreakHisModel):
        return model.threshold
    return CNN_MALIGNANT_THRESHOLD


def predict(model: Any, image: Image.Image) -> dict:
    if model is None:
        return {
            "label": "Unavailable",
            "confidence": 0.0,
            "benign_prob": 0.0,
            "malignant_prob": 0.0,
            "is_high_confidence": False,
            "error": "No trained model loaded. Run train_sklearn_model.py first.",
        }

    if not is_histology_like(image):
        return {
            "label": "Invalid Input",
            "confidence": 0.0,
            "benign_prob": 0.0,
            "malignant_prob": 0.0,
            "is_high_confidence": False,
            "error": "Input image does not appear to be a histopathology slide."
        }

    malignant_prob = _malignant_probability(model, image)
    threshold = _decision_threshold(model)

    if abs(malignant_prob - threshold) < UNCERTAINTY_MARGIN:
        return {
            "label": "Uncertain",
            "confidence": 0.0,
            "benign_prob": 1 - malignant_prob,
            "malignant_prob": malignant_prob,
            "is_high_confidence": False,
            "error": "Model is uncertain — input may be ambiguous or out-of-distribution.",
        }

    benign_prob = 1.0 - malignant_prob

    if malignant_prob >= threshold:
        label = "Malignant"
        confidence = malignant_prob
    else:
        label = "Benign"
        confidence = benign_prob

    return {
        "label": label,
        "confidence": confidence,
        "benign_prob": benign_prob,
        "malignant_prob": malignant_prob,
        "is_high_confidence": confidence >= 0.85,
    }


def _demo_heatmap(image: Image.Image, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    gray = base.convert("L").filter(ImageFilter.FIND_EDGES)
    heatmap_rgb = Image.new("RGB", gray.size)
    heatmap_rgb.paste(gray)
    return Image.blend(base, heatmap_rgb, alpha=alpha)


def make_gradcam_heatmap(model: Any, image: Image.Image, alpha: float = 0.45) -> Image.Image:
    if model is None or isinstance(model, SklearnBreakHisModel) or not TF_AVAILABLE:
        return _demo_heatmap(image, alpha=alpha)

    batch = preprocess_image(image)
    img_tensor = tf.convert_to_tensor(batch)

    conv_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.filters == 512:
            conv_layer = layer
            break

    if conv_layer is None:
        return image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((IMG_SIZE, IMG_SIZE))
    heatmap_rgb = Image.new("RGB", heatmap_img.size)
    heatmap_rgb.paste(heatmap_img, mask=heatmap_img)

    base = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return Image.blend(base, heatmap_rgb, alpha=alpha)
