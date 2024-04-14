"""yunet + sface + faiss index for vector storage"""
import cv2
import numpy as np
import faiss
import logging
import os

logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognizer_fast.onnx"
CONF_THRESH = 0.6  # 0.6 catches more faces than 0.7, was missing detections indoors
EMBEDDING_DIM = 128


def is_blurry(image, threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


class Camera:
    def __init__(self, index_path="face_embedding.index"):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("cannot open webcam")
        self.face = None
        self.face_box = None
        self.last_frame = None
        self.detector = None
        self.face_recognizer = None
        self.embedding_dim = EMBEDDING_DIM
        self.index_path = index_path
        self.index = None
        self._load_detector()
        self._load_recognizer()
        self.initialize_faiss_index()

    def _load_detector(self):
        if not os.path.exists(YUNET_MODEL):
            log.warning(f"yunet not found: {YUNET_MODEL}")
            return
        try:
            self.detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 240))
            log.info("YuNet face detector loaded.")
        except Exception as e:
            log.error(f"yunet load failed: {e}")

    def _load_recognizer(self):
        if not os.path.exists(SFACE_MODEL):
            log.warning(f"sface not found: {SFACE_MODEL}")
            return
        try:
            self.face_recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
            log.info(f"SFace recognizer loaded ({EMBEDDING_DIM}-d embeddings).")
        except Exception as e:
            log.error(f"sface load failed: {e}")

    def initialize_faiss_index(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                log.info(f"Loading FAISS index from {self.index_path}")
            except Exception as e:
                log.error(f"FAISS load failed: {e}")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            log.info(f"Creating new FAISS index (dim={self.embedding_dim})")

    def store_embedding(self, embedding):
        if embedding is None:
            return False
        try:
            vec = embedding.astype(np.float32).reshape(1, -1)
            self.index.add(vec)
            faiss.write_index(self.index, self.index_path)
            return True
        except Exception as e:
            log.error(f"Failed to store embedding: {e}")
            return False

    def rebuild_index(self, embeddings):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if embeddings:
            mat = np.array(embeddings, dtype=np.float32)
            self.index.add(mat)
        try:
            faiss.write_index(self.index, self.index_path)
        except Exception as e:
            log.error(f"Failed to write index: {e}")

    def get_embedding(self, face_img):
        if self.face_recognizer is None:
            return None
        if is_blurry(face_img):
            log.warning("face crop is too blurry, skipping embedding")
            return None
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            eq = clahe.apply(gray)
            face_eq = cv2.merge([eq, eq, eq])
            resized = cv2.resize(face_eq, (112, 112))
            raw = self.face_recognizer.feature(resized).flatten()
            norm = max(np.linalg.norm(raw), 1e-10)
            return raw / norm
        except Exception as e:
            log.error(f"embedding failed: {e}")
            return None

    def process_frame(self, frame):
        if self.detector is None:
            return frame
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, dets = self.detector.detect(frame)
        self.face = None
        self.face_box = None
        self.last_frame = frame
        if dets is None or len(dets) == 0:
            return frame
        best = max(dets, key=lambda d: d[-1])
        if best[-1] < CONF_THRESH:
            return frame
        x, y, bw, bh = [int(v) for v in best[:4]]
        x2, y2 = min(w, x + bw), min(h, y + bh)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{best[-1]*100:.1f}%", (x, y - 8 if y > 20 else y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self.face = frame[y:y2, x:x2]
        self.face_box = best
        return frame

    def run(self):
        log.info("press q to quit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                log.error("frame read failed")
                break
            cv2.imshow("Video", self.process_frame(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        cam = Camera()
        cam.run()
    except IOError as e:
        log.error(e)
    finally:
        if "cam" in locals():
            cam.release_resources()
