"""
Evaluate YuNet + SFace pipeline on AT&T Olivetti Faces and LFW datasets.
Supports both datasets for benchmarking recognition accuracy.
"""
import numpy as np
import cv2
import os
import logging
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    face_eq = cv2.merge([eq, eq, eq])
    resized = cv2.resize(face_eq, (112, 112))
    raw = recognizer.feature(resized).flatten()
    norm = max(np.linalg.norm(raw), 1e-10)
    return raw / norm


def nn_accuracy(embeddings, labels, threshold=THRESHOLD, n_folds=N_FOLDS):
    """k-fold nearest neighbor leave-one-out evaluation"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        train_embs, train_labels = embeddings[train_idx], labels[train_idx]
        test_embs, test_labels = embeddings[test_idx], labels[test_idx]
        preds = []
        for emb in test_embs:
            dists = np.linalg.norm(train_embs - emb, axis=1)
            best = np.argmin(dists)
            preds.append(train_labels[best] if dists[best] < threshold else -1)
        acc = accuracy_score(test_labels, preds)
        fold_accs.append(acc)
        log.info(f"  fold {fold+1}: {acc:.4f}")
    return float(np.mean(fold_accs))


def run_olivetti():
    log.info("--- Olivetti Faces ---")
    ds = fetch_olivetti_faces(shuffle=True, random_state=42)
    images, labels = ds.images, ds.target
    recognizer = load_recognizer()
    imgs_bgr = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for img in images]
    log.info("computing embeddings...")
    embs = np.array([get_embedding(recognizer, img) for img in imgs_bgr])
    acc = nn_accuracy(embs, labels)
    log.info(f"Olivetti {N_FOLDS}-fold accuracy: {acc:.4f}")
    return acc


def run_lfw(min_faces=20):
    """LFW evaluation - people with >= min_faces images"""
    log.info(f"--- LFW (min_faces_per_person={min_faces}) ---")
    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        log.error("scikit-learn not installed, skipping LFW")
        return None
    log.info("fetching LFW dataset (downloads ~200MB on first run)...")
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, resize=1.0)
    images, labels = lfw.images, lfw.target
    log.info(f"LFW: {len(images)} images, {len(set(labels))} subjects")
    recognizer = load_recognizer()
    # convert grayscale LFW images to BGR uint8
    imgs_bgr = []
    for img in images:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs_bgr.append(img)
    log.info("computing LFW embeddings...")
    embs = np.array([get_embedding(recognizer, img) for img in imgs_bgr])
    acc = nn_accuracy(embs, labels)
    log.info(f"LFW {N_FOLDS}-fold accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Face recognition benchmark")
    parser.add_argument("--dataset", choices=["olivetti", "lfw", "both"], default="both")
    parser.add_argument("--min-faces", type=int, default=20)
    args = parser.parse_args()

    results = {}
    if args.dataset in ("olivetti", "both"):
        results["olivetti"] = run_olivetti()
    if args.dataset in ("lfw", "both"):
        results["lfw"] = run_lfw(args.min_faces)

    print("\n=== Results ===")
    for k, v in results.items():
        if v is not None:
            print(f"{k}: {v:.4f}")
