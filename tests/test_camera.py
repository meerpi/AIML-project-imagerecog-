"""unit tests for Camera class"""
import unittest
import numpy as np
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.index_path = os.path.join(self.tmp, "test.index")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("cv2.VideoCapture")
    def _make_camera(self, index_path, mock_cap):
        mock_cap.return_value.isOpened.return_value = True
        with patch("cam_on.Camera._load_detector"), \
             patch("cam_on.Camera._load_recognizer"), \
             patch("cam_on.Camera.initialize_faiss_index"):
            from cam_on import Camera
            cam = Camera(index_path=index_path)
            cam.face_recognizer = MagicMock()
            return cam

    def test_initialize_faiss_creates_new_index(self):
        cam = self._make_camera(self.index_path)
        cam.initialize_faiss_index()
        self.assertIsNotNone(cam.index)

    def test_store_embedding_adds_to_index(self):
        import faiss
        cam = self._make_camera(self.index_path)
        cam.initialize_faiss_index()
        emb = np.random.rand(128).astype(np.float32)
        emb /= np.linalg.norm(emb)
        result = cam.store_embedding(emb)
        self.assertTrue(result)
        self.assertEqual(cam.index.ntotal, 1)

    def test_store_embedding_none_returns_false(self):
        cam = self._make_camera(self.index_path)
        cam.initialize_faiss_index()
        self.assertFalse(cam.store_embedding(None))

    def test_get_embedding_no_recognizer(self):
        cam = self._make_camera(self.index_path)
        cam.face_recognizer = None
        import numpy as np
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        result = cam.get_embedding(face)
        self.assertIsNone(result)

    def test_rebuild_index_resets_ntotal(self):
        cam = self._make_camera(self.index_path)
        cam.initialize_faiss_index()
        emb = np.random.rand(128).astype(np.float32)
        cam.store_embedding(emb)
        cam.rebuild_index([])
        self.assertEqual(cam.index.ntotal, 0)


if __name__ == "__main__":
    unittest.main()
