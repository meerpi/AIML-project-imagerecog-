"""
Evaluate the YuNet + SFace pipeline on the AT&T Olivetti Faces dataset.
Olivetti is a classic 40-class face dataset, easy to test locally without downloading LFW.
"""
import numpy as np
import cv2
import os
import logging
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

SFACE_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "face_recognizer_fast.onnx")
THRESHOLD = 0.35
N_FOLDS = 5


def load_recognizer():
    if not os.path.exists(SFACE_MODEL):
        raise FileNotFoundError(f"SFace model not found at {SFACE_MODEL}")
    return cv2.FaceRecognizerSF.create(SFACE_MODEL, "")


def get_embedding(recognizer, face_img):
    """compute normalized sface embedding with CLAHE equalization"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    face_eq = cv2.merge([eq, eq, eq])
    resized = cv2.resize(face_eq, (112, 112))
    raw = recognizer.feature(resized).flatten()
    norm = max(np.linalg.norm(raw), 1e-10)
    return raw / norm


def run_olivetti_eval():
    log.info("loading AT&T Olivetti Faces dataset...")
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
    images = dataset.images  # (400, 64, 64) float [0,1]
    labels = dataset.target  # (400,) int

    log.info(f"dataset: {len(images)} images, {len(set(labels))} subjects")

    recognizer = load_recognizer()

    # convert to uint8 BGR for opencv
    imgs_bgr = []
    for img in images:
        gray8 = (img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        imgs_bgr.append(bgr)

    log.info("computing embeddings...")
    embeddings = np.array([get_embedding(recognizer, img) for img in imgs_bgr])

    # k-fold nearest neighbor evaluation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        train_embs = embeddings[train_idx]
        train_labels = labels[train_idx]
        test_embs = embeddings[test_idx]
        test_labels = labels[test_idx]

        preds = []
        for emb in test_embs:
            dists = np.linalg.norm(train_embs - emb, axis=1)
            best_idx = np.argmin(dists)
            if dists[best_idx] < THRESHOLD:
                preds.append(train_labels[best_idx])
            else:
                preds.append(-1)

        acc = accuracy_score(test_labels, preds)
        fold_accs.append(acc)
        log.info(f"fold {fold+1}: accuracy={acc:.4f}")

    mean_acc = np.mean(fold_accs)
    log.info(f"\nOlivetti mean accuracy ({N_FOLDS}-fold): {mean_acc:.4f}")
    return mean_acc


if __name__ == "__main__":
    run_olivetti_eval()
