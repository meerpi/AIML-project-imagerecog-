# Model Evaluation Report — Face Recognition Pipeline

**Date:** May 18, 2024
**Pipeline:** YuNet (detection) + SFace (128-d cosine embeddings) + FAISS (L2 index)

---

## Dataset: AT&T Olivetti Faces

- 400 images, 40 subjects, 10 images each
- 64×64 grayscale, indoor controlled conditions
- Evaluation: 5-fold stratified nearest-neighbor

| Metric | Value |
|--------|-------|
| Mean Accuracy | 94.5% |
| Threshold | 0.35 |
| Embedding dim | 128 |

### Per-Fold Results

| Fold | Accuracy |
|------|----------|
| 1 | 92.5% |
| 2 | 95.0% |
| 3 | 96.25% |
| 4 | 95.0% |
| 5 | 93.75% |

### Notes

- CLAHE equalization helps significantly with the Olivetti grayscale images
- False rejections (dist > threshold) account for most errors
- Next: test on LFW for real-world accuracy
