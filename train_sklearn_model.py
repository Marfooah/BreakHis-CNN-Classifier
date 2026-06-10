"""Train a lightweight BreakHis benign/malignant classifier (no TensorFlow required)."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import kagglehub
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_SIZE = 96
MODEL_PATH = Path(__file__).resolve().parent / "models" / "breakhis_sklearn.joblib"
SEED = 42


def dataset_root() -> Path:
    root = kagglehub.dataset_download(
        "waseemalastal/breakhis-breast-cancer-histopathological-dataset"
    )
    return Path(root) / (
        "BreakHis - Breast Cancer Histopathological Database"
        "/dataset_cancer_v1/dataset_cancer_v1/classificacao_binaria"
    )


def collect_paths_labels(data_dir: Path) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            lower = root.lower()
            if "benign" in lower:
                label = 0
            elif "malignant" in lower:
                label = 1
            else:
                continue
            paths.append(os.path.join(root, name))
            labels.append(label)
    return paths, labels


def extract_features(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((FEATURE_SIZE, FEATURE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = np.mean(arr, axis=2)

    # Color statistics
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))

    # Texture / edge energy
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    lap_var = float(np.var(np.gradient(gray)[0]) + np.var(np.gradient(gray)[1]))

    # Downsampled patch statistics (captures tissue structure)
    patch = gray.reshape(8, 12, 8, 12).mean(axis=(1, 3)).ravel()

    # Purple/pink stain ratio common in H&E malignant regions
    stain_ratio = float(means[0] / (means[2] + 1e-6))

    return np.concatenate(
        [
            means,
            stds,
            [lap_var, grad_mag.mean(), grad_mag.std(), stain_ratio],
            patch,
        ]
    ).astype(np.float32)


def build_matrix(paths: list[str], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.stack([extract_features(p) for p in paths])
    ys = np.asarray(labels, dtype=np.int32)
    return xs, ys


def main() -> None:
    data_dir = dataset_root()
    paths, labels = collect_paths_labels(data_dir)
    print(f"Images: {len(paths)} (benign={labels.count(0)}, malignant={labels.count(1)})")

    x_train_paths, x_test_paths, y_train, y_test = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=SEED
    )

    print("Extracting training features…")
    x_train, y_train_arr = build_matrix(x_train_paths, y_train)
    print("Extracting test features…")
    x_test, y_test_arr = build_matrix(x_test_paths, y_test)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train_arr)
    y_pred = pipeline.predict(x_test)
    print(classification_report(y_test_arr, y_pred, target_names=["Benign", "Malignant"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "feature_size": FEATURE_SIZE, "threshold": 0.35},
        MODEL_PATH,
    )
    print(f"Saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
