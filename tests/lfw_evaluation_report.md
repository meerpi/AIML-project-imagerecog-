# Model Evaluation Report — Face Recognition Pipeline

**Last Updated:** August 9, 2024
**Pipeline:** YuNet (detection + 5-point alignment) + SFace (128-d normalized embeddings) + FAISS (L2 index)

---

## Dataset 1: AT&T Olivetti Faces

- 400 images, 40 subjects, 10 images each
- 64×64 grayscale, controlled indoor conditions
- Evaluation: 5-fold stratified nearest-neighbor, threshold=0.35

| Metric | Value |
|--------|-------|
| Mean Accuracy | 94.5% |

### Per-Fold Results

| Fold | Accuracy |
|------|----------|
| 1 | 92.5% |
| 2 | 95.0% |
| 3 | 96.25% |
| 4 | 95.0% |
| 5 | 93.75% |

---

## Dataset 2: LFW (Labeled Faces in the Wild)

- min_faces_per_person=20: ~2650 images, 62 subjects
- 125×94 RGB, real-world unconstrained conditions
- Evaluation: 5-fold stratified nearest-neighbor, threshold=0.35

| Metric | Value |
|--------|-------|
| Mean Accuracy | 96.1% |

### Per-Fold Results

| Fold | Accuracy |
|------|----------|
| 1 | 95.7% |
| 2 | 96.8% |
| 3 | 96.3% |
| 4 | 95.9% |
| 5 | 95.8% |

---

## Notes

- SFace with CLAHE equalization performs better than raw RGB on both datasets
- LFW accuracy is higher than Olivetti because subjects with 20+ photos tend to have well-lit, frontal images
- Main failure mode: false rejections (distance above threshold) not false acceptances
- alignCrop using yunet 5-point landmarks significantly reduces pose variation errors
