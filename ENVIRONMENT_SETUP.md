# Environment Setup

Tested on Python 3.11.4, Ubuntu 22.04.

## Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dependencies

- `opencv-python>=4.8.0` — YuNet and SFace are built into opencv-contrib from 4.5+
- `faiss-cpu>=1.7.4` — FAISS vector search
- `customtkinter>=5.2.0` — GUI
- `scikit-learn, matplotlib, tqdm` — evaluation only
