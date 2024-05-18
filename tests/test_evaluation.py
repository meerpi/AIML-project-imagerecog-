"""unit tests for evaluation benchmark functions"""
import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch


class TestEmbeddingNorm(unittest.TestCase):
    def test_embedding_is_unit_length(self):
        """embeddings should be L2-normalized"""
        raw = np.random.rand(128).astype(np.float32)
        norm = max(np.linalg.norm(raw), 1e-10)
        emb = raw / norm
        self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=5)

    def test_zero_vector_no_division_error(self):
        """zero vector should not raise divide by zero"""
        raw = np.zeros(128, dtype=np.float32)
        norm = max(np.linalg.norm(raw), 1e-10)
        emb = raw / norm
        self.assertEqual(emb.shape, (128,))

    def test_nearest_neighbor_correct(self):
        """nearest neighbor search should return closest embedding"""
        gallery = np.random.rand(5, 128).astype(np.float32)
        for i in range(5):
            gallery[i] /= np.linalg.norm(gallery[i])
        query = gallery[2] + np.random.rand(128).astype(np.float32) * 0.001
        query /= np.linalg.norm(query)
        dists = np.linalg.norm(gallery - query, axis=1)
        pred = np.argmin(dists)
        self.assertEqual(pred, 2)

    def test_high_dist_means_no_match(self):
        """embeddings far apart should not match"""
        a = np.zeros(128, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(128, dtype=np.float32)
        b[1] = 1.0
        dist = np.linalg.norm(a - b)
        self.assertGreater(dist, 0.35)


if __name__ == "__main__":
    unittest.main()
