"""face matching module - returns (student, dist) tuple now"""
import numpy as np
import faiss
import json
import os
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

STUDENT_DATA_FILE = "student_data.json"
EMBEDDING_DIM = 128
THRESHOLD = 0.35


class FaceChecker:
    def __init__(self, camera):
        self.camera = camera
        self.index = camera.index
        self.student_data = {}
        self._load_student_data()

    def _load_student_data(self):
        if not os.path.exists(STUDENT_DATA_FILE):
            log.info("no student data file yet")
            return
        try:
            with open(STUDENT_DATA_FILE, "r") as f:
                data = json.load(f)
            # json.load can return a list if file was written wrong
            if not isinstance(data, dict):
                log.error("student_data.json is not a dict, resetting")
                self.student_data = {}
                return
            self.student_data = data
            log.info(f"loaded {len(self.student_data)} students")
        except Exception as e:
            log.error(f"failed to load student data: {e}")

    def search_face(self, embedding, top_k=1):
        if embedding is None:
            return None, None
        if self.index.ntotal == 0:
            return None, None
        try:
            vec = embedding.astype(np.float32).reshape(1, -1)
            dists, idxs = self.index.search(vec, top_k)
            dist = dists[0][0]
            idx = idxs[0][0]
            if idx < 0:
                return None, None
            if dist > THRESHOLD:
                return None, dist
            keys = list(self.student_data.keys())
            if idx >= len(keys):
                return None, dist
            return self.student_data[keys[idx]], dist
        except Exception as e:
            log.error(f"face search failed: {e}")
            return None, None

    def check_face_from_camera(self):
        frame = self.camera.last_frame
        detection = self.camera.face_box
        if frame is None or detection is None:
            return None, None
        emb = self.camera.get_embedding_from_frame(frame, detection)
        return self.search_face(emb)
