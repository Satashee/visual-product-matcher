# README.md

# Visual Product Matcher

Upload an image (or paste a URL) → extract CNN features (ResNet50/MobileNet) + HSV color → two-stage search (feature top-K → color-aware rerank) → clean UI with percent badges.

<p align="center">
  <img alt="App screenshot" src="static/preview.png" width="720">
</p>

<div align="center">
  
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000?logo=flask)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

</div>

---

## ✨ What it does

- **Backbones:** Torchvision **ResNet50** (default) or **MobileNetV3** (faster).
- **Color-aware rerank:** 24-dim HSV histogram blended into the score.
- **Two-stage search:** feature top-K (dynamic by dataset size), then color rerank.
- **Absolute scoring:** no per-query normalization → slider behaves consistently.
- **Image proxy & cache:** avoids hotlinking/CORS issues; flattens PNG transparency.
- **Multiple data sources:** quick online demo sources, or your own **folder** / **CSV**.

> Current version: **no background removal** (more robust for product photos with thin straps, glass, etc.).

---

## 🧱 Project structure
```
visual-product-matcher/
├─ app.py
├─ precompute.py
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ templates/
│  └─ index.html
├─ static/
│  ├─ cache/                     # runtime image cache (gitignored)
│  │  └─ .gitkeep
│  └─ images/                    # optional local images for offline mode
│     └─ .gitkeep
└─ data/
   └─ .gitkeep                   # keeps folder in git; products.json is generated
```

---

## 🚀 Quick start

### 1) Environment

```
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```
###2) Build a dataset (pick one)

#A) Mixed online demo sources (fastest to try):
```
python precompute.py --online mixed --items 600 --max-items 800 --min-size 128

```
#B) From a folder (category = parent folder name):
```
# catalog/
#   shoes/   img1.jpg ...
#   bags/    img2.jpg ...
python precompute.py --scan catalog --min-size 128
```

#C) From a CSV:
```
python precompute.py --csv catalog.csv --min-size 128
```
##3) Run the app
```
python app.py
# open http://127.0.0.1:5000
```
