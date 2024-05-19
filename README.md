# Facial Registration System

A real-time face registration and recognition system built with:
- **YuNet** (OpenCV) — lightweight face detector with 5-point landmark detection
- **SFace** (OpenCV) — 128-dimensional face embedding model
- **FAISS** — fast approximate nearest neighbor search
- **CustomTkinter** — modern dark-mode GUI

## Architecture

```
Camera → YuNet Detector → alignCrop → SFace Embedder → FAISS Index
                                                              ↓
                                               FaceChecker.search_face()
```

## Setup

See `ENVIRONMENT_SETUP.md` for environment setup instructions.
See `MODEL_DOWNLOAD.md` to download the required ONNX models.

## Running

```bash
python registration_app.py  # register students
python check_face.py        # check/verify faces
```

## Evaluation

```bash
python tests/evaluate_lfw.py  # run benchmark on Olivetti / LFW
```

## Project Structure

```
cam_on.py            - Camera, YuNet detector, SFace embedder, FAISS index
check_face.py        - FaceChecker FAISS nearest-neighbor search
registration_app.py  - CustomTkinter registration GUI
logger.py            - dual file+console logging
tests/               - unit tests and evaluation scripts
```
