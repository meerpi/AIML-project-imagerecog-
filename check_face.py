"""face matching module - searches faiss index for closest face"""
import numpy as np
import faiss
import json
import os
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

STUDENT_DATA_FILE = "student_data.json"
THRESHOLD = 0.35


class FaceChecker:
    def __init__(self, camera):
        self.camera = camera
        self.index = camera.index
        self.embedding_dim = camera.embedding_dim
        self.student_data = {}
        self._load_student_data()

    def _load_student_data(self):
        if not os.path.exists(STUDENT_DATA_FILE):
            log.info("no student data file yet")
            return
        with open(STUDENT_DATA_FILE, "r") as f:
            self.student_data = json.load(f)
        log.info(f"loaded {len(self.student_data)} students")

    def search_face(self, embedding, top_k=1):
        if embedding is None:
            return None
        if self.index.ntotal == 0:
            return None
        vec = embedding.astype(np.float32).reshape(1, -1)
        dists, idxs = self.index.search(vec, top_k)
        dist = dists[0][0]
        idx = idxs[0][0]
        if idx < 0 or dist > THRESHOLD:
            return None
        keys = list(self.student_data.keys())
        if idx >= len(keys):
            return None
        return self.student_data[keys[idx]]
