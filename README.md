# Visual Product Matcher

Upload a photo (or paste a direct image URL) → extract CNN features (ResNet50/MobileNet) + HSV color → two-stage search (feature top-K → color-aware re-rank) → clean Bootstrap UI with percent badges, progress bars, and a loading overlay.

<p align="center">
  <img width="1919" height="817" alt="Screenshot 2025-08-31 150204" src="https://github.com/user-attachments/assets/f7c3fd08-9a63-406f-bde2-7eadbeb2270d" />
</p>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000?logo=flask)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

</div>

---

## ✨ What it does

- **Backbones:** Torchvision **ResNet50** (default) or **MobileNetV3** (faster/smaller).
- **Color-aware rerank:** 24-dim HSV histogram blended into the score (small weight).
- **Two-stage search:** feature-only dynamic **top-K** (scaled to dataset) → light color rerank.
- **Absolute scoring:** no per-query normalization → your slider is consistent.
- **Image proxy & cache:** `/img?src=...` avoids hotlink blocks/CORS, flattens transparency, and caches JPEGs in `static/cache`.
- **Multiple data sources:** use quick online demo sources or your own **folder** / **CSV**.
- **No background removal:** current version leaves backgrounds intact (more robust for straps, glass, thin objects).

---

## 🧠 How it works (quick)

## 1. **Precompute:** `precompute.py` downloads/builds a catalog and stores:
   - L2-normalized **CNN feature vectors** (ResNet50/MobileNetV3).
   - L2-normalized **HSV histograms** (24D).
   - JSON is written to `data/products.json` (gitignored).
## 2. **Query time:** `app.py` embeds the query image, picks dynamic **top-K** by feature cosine, then adds a small color dot-product to rerank. Results below `MIN_SHOW` are hidden.

---

## 📦 Project structure

```
visual-product-matcher/
├─ app.py # Flask app (no BG removal, proxy/cache, dynamic top-K)
├─ precompute.py # Build data/products.json from online/folder/CSV sources
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ templates/
│ └─ index.html # UI with score badges, bars, and loader overlay
├─ static/
│ ├─ placeholder.jpg
│ ├─ preview.png
│ ├─ cache/ # runtime image cache (gitignored)
│ │ └─ .gitkeep
│ └─ images/ # optional offline images (for --scan)
│ └─ .gitkeep
└─ data/
└─ .gitkeep # keeps folder in git; products.json is generated
```

---

## 🛠 Prerequisites

- Python **3.10+**
- pip / venv
- CPU is fine. (GPU speeds up precompute & query if you install CUDA PyTorch.)

---

## 🚀 Quick start

### 1) Create & activate a venv, install deps
```
python -m venv .venv

# Windows PowerShell

.venv\Scripts\Activate.ps1

# macOS/Linux

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```
### 2) Build a dataset (pick one)
## A) Mixed online demo sources (fastest to try)
```
python precompute.py --online mixed --items 600 --max-items 800 --min-size 128
```
## B) From a folder (category = parent folder name)
```
catalog/
  shoes/     nike1.jpg  adidas2.jpg  ...
  bags/      tote.png   backpack.jpg ...

python precompute.py --scan catalog --min-size 128
```
## C) From a CSV (columns: id,name,category,image_url)
```
python precompute.py --csv catalog.csv --min-size 128
```

data/products.json will be created. It should not be committed to git.

## 3) Run the app
```
python app.py
# visit http://127.0.0.1:5000
```
### ⚙️ Configuration & Tuning

In app.py:
```
W_FEAT / W_COLOR — feature vs. color blend (default 0.90 / 0.10).

MIN_SHOW — absolute score floor (default 0.50).

QUERY_MIN_SIDE — minimum query short side (default 256 px).
```
In precompute.py:
```
--online {mixed, escuelajs, fakestore, dummyjson} — pick a source.

--scan <folder> — build from your images; category is parent folder.

--csv <file.csv> — build from CSV (must include image_url).

--items, --max-items — control ingestion and compute budget.

--min-size — skip tiny images (default 128).

--backbone {resnet50, mobilenetv3} — feature extractor.
```
# Environment variables

Create a .env (or set in your host) if you change defaults:

# Optional; overrides the default user agent used for remote fetches
```
HTTP_USER_AGENT="Mozilla/5.0 (Visual Product Matcher)"
```
# Flask
```
FLASK_ENV=production
```

### 🧪 Tips for testing

Prefer direct image URLs that end in .jpg/.jpeg/.png/....

If a host blocks hotlinking, the built-in proxy /img?src=... + cache helps.

Uploading a local file is always the most reliable.

### 🐳 Deployment
# A) Render (simple)

Push this repo to GitHub.

Create Web Service on Render → connect repo.
```
Build: pip install -r requirements.txt
Start: gunicorn app:app --workers=2 --bind 0.0.0.0:$PORT
```
Set env FLASK_ENV=production.

Add a Procfile (optional but recommended):
```
web: gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT
```
# B) Docker
# Dockerfile
```
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["gunicorn","app:app","--workers","2","--bind","0.0.0.0:8000"]
```

Build & run:
```
docker build -t visual-product-matcher .
docker run -p 8000:8000 visual-product-matcher
```
# C) Bare metal / VM
```
pip install -r requirements.txt
gunicorn app:app --workers 2 --bind 0.0.0.0:8000
```

Note: Generate data/products.json on the server; don’t commit it.

### ❗ Troubleshooting
```
“data/products.json is empty / missing”
Run a precompute step again, e.g.:

python precompute.py --online mixed --items 600

```
“No results” for every query

Ensure your query image ≥ QUERY_MIN_SIDE (default 256 px short edge).

Lower MIN_SHOW (e.g., 0.45) or increase dataset size.

Remote image won’t load
Some hosts block hotlinking. The proxy /img?src=... helps; otherwise upload the file.

Git push is huge / times out
Confirm .gitignore excludes generated & cache files (see below).

Slow build time
Use fewer --items first (e.g., 400–800). Switch to MobileNetV3 for faster precompute.

### 🧾 .gitignore (important)

Make sure you don’t commit generated data or caches:
```
.venv/
__pycache__/
*.pyc
*.pyo
.DS_Store

# Hugging Face / Torch caches (if any)
.hf_cache/
torch_cache/

# Runtime caches & generated artifacts
static/cache/
data/products.json
```
### 🧭 Roadmap

Toggle backbones from UI (ResNet50 / MobileNet / CLIP/SigLIP).

Category filters & facets.

Vector DB (FAISS/Weaviate) for 50k+ images.

Multi-query voting (average embeddings across shots).

“Pin & Compare” UI, CSV export of results.

### 🤝 Contributing

PRs welcome!
Please keep the app functional without private keys and avoid committing generated assets (data/products.json, caches). For UI/ranking changes, include a short before/after summary.

## 🧩 FAQ

Q: I get “No results” for everything.
A: Check that data/products.json exists and isn’t empty. Re-run:
```
python precompute.py --online mixed --items 600
```

Also verify your uploaded/URL image is at least QUERY_MIN_SIDE pixels on the short edge, or reduce QUERY_MIN_SIDE in app.py.

Q: Scores feel too strict / too loose.

A: Lower or raise MIN_SHOW (e.g., 0.40 or 0.60). You can also tweak W_FEAT/W_COLOR.

Q: Remote images don’t load.

A: Some hosts block hotlinking or close connections. The /img proxy solves most cases. If it still fails, try another URL.

Q: CPU vs GPU?

A: This works on CPU. For GPU, install PyTorch with CUDA and it will run faster automatically.

Q: I don’t want any background removal.

A: It’s already disabled in the current version. We only square-crop + center to reduce border bias.
