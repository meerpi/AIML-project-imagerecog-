"""unit tests for FaceChecker class"""
import unittest
import numpy as np
import os
import sys
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch


def _make_mock_camera(index_path, n_students=0):
    import faiss
    cam = MagicMock()
    cam.index = faiss.IndexFlatL2(128)
    cam.last_frame = None
    cam.face_box = None
    return cam


class TestFaceChecker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_path = os.path.join(self.tmp, "student_data.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_checker(self, student_data=None):
        with patch("check_face.STUDENT_DATA_FILE", self.data_path):
            if student_data is not None:
                with open(self.data_path, "w") as f:
                    json.dump(student_data, f)
            cam = _make_mock_camera(self.tmp)
            from check_face import FaceChecker
            return FaceChecker(cam)

    def test_empty_index_returns_none(self):
        checker = self._make_checker()
        emb = np.random.rand(128).astype(np.float32)
        student, dist = checker.search_face(emb)
        self.assertIsNone(student)
        self.assertIsNone(dist)

    def test_search_finds_close_match(self):
        import faiss
        checker = self._make_checker({"22bcs001_alice": {"name": "Alice", "reg_no": "22BCS001"}})
        emb = np.random.rand(128).astype(np.float32)
        emb /= np.linalg.norm(emb)
        checker.index.add(emb.reshape(1, -1).astype(np.float32))
        # query with nearly same embedding
        noisy = emb + np.random.rand(128).astype(np.float32) * 0.001
        noisy /= np.linalg.norm(noisy)
        student, dist = checker.search_face(noisy)
        self.assertIsNotNone(student)
        self.assertEqual(student["name"], "Alice")

    def test_none_embedding_returns_none(self):
        checker = self._make_checker()
        student, dist = checker.search_face(None)
        self.assertIsNone(student)
        self.assertIsNone(dist)

    def test_corrupt_json_loads_empty(self):
        with open(self.data_path, "w") as f:
            f.write("not json {{{")
        cam = _make_mock_camera(self.tmp)
        with patch("check_face.STUDENT_DATA_FILE", self.data_path):
            from check_face import FaceChecker
            checker = FaceChecker(cam)
        self.assertEqual(checker.student_data, {})

    def test_non_dict_json_loads_empty(self):
        with open(self.data_path, "w") as f:
            json.dump([1, 2, 3], f)
        cam = _make_mock_camera(self.tmp)
        with patch("check_face.STUDENT_DATA_FILE", self.data_path):
            from check_face import FaceChecker
            checker = FaceChecker(cam)
        self.assertEqual(checker.student_data, {})


if __name__ == "__main__":
    unittest.main()
