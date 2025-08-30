# app.py — no background removal + resilient fetching (retries) + proxy cache
import io, os, json, hashlib
from pathlib import Path

import numpy as np
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from PIL import Image, ImageFile, ImageOps
from flask import Flask, render_template, request, send_file, abort

import torch, torchvision

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------- tuning knobs -------
W_FEAT   = 0.90          # main feature weight
W_COLOR  = 0.10          # small color influence
BACKBONE = "resnet50"    # or "mobilenetv3"
MIN_SHOW = 0.50          # hard floor to show results (absolute score)
QUERY_MIN_SIDE = 256     # reject tiny user images (< this on short side)
HEADERS = {"User-Agent": "Mozilla/5.0"}  # for remote fetches
MAX_IMAGE_BYTES = 10 * 1024 * 1024       # 10MB guardrail
# ---------------------------

app = Flask(__name__)

# ---------- requests Session with retries ----------
SESSION = requests.Session()
retries = Retry(
    total=4,
    backoff_factor=0.6,                      # 0.6s, 1.2s, 1.8s, ...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retries, pool_maxsize=64)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)

def fetch_bytes(url: str, timeout=25) -> bytes:
    """Stream download with retry + size guard."""
    r = SESSION.get(url, headers=HEADERS, timeout=timeout, stream=True)
    r.raise_for_status()
    declared = int(r.headers.get("content-length") or 0)
    if declared and declared > MAX_IMAGE_BYTES:
        raise ValueError(f"Remote image too large: {declared} bytes")
    chunk = r.raw.read(MAX_IMAGE_BYTES + 1, decode_content=True)
    if len(chunk) > MAX_IMAGE_BYTES:
        raise ValueError("Remote image too large")
    return chunk

# ---------- image proxy with transparency flatten ----------
CACHE_DIR = Path("static/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _flatten_to_rgb(data: bytes, bg=(255, 255, 255)) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)  # fix rotated images (EXIF)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        back = Image.new("RGBA", rgba.size, bg + (255,))
        im = Image.alpha_composite(back, rgba).convert("RGB")
    else:
        im = im.convert("RGB")
    return im

@app.route("/img")
def img_proxy():
    """
    Serve images through the app to avoid hotlink blocks.
    - Local paths are served directly.
    - Remote URLs are fetched once and cached as JPEG.
    """
    src = request.args.get("src", "")
    if not src:
        abort(400, description="Missing src")

    # Local path → serve
    if not (src.startswith("http://") or src.startswith("https://")):
        if os.path.exists(src):
            return send_file(src, conditional=True)
        abort(404, description="Local file not found")

    # Remote URL → cache file
    key = hashlib.sha1(src.encode("utf-8")).hexdigest() + ".jpg"
    cached = CACHE_DIR / key
    if not cached.exists():
        try:
            content = fetch_bytes(src, timeout=25)
            im = _flatten_to_rgb(content, bg=(255, 255, 255))
            im.thumbnail((1024, 1024), Image.LANCZOS)
            im.save(cached, "JPEG", quality=90, optimize=True)
        except Exception as e:
            abort(502, description=f"Proxy fetch failed: {e}")
    return send_file(cached, mimetype="image/jpeg", conditional=True)

# ---------- backbone ----------
def get_backbone(name="resnet50"):
    if name == "mobilenetv3":
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        m = torchvision.models.mobilenet_v3_small(weights=weights)
        m.classifier[-1] = torch.nn.Identity()
    else:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        m = torchvision.models.resnet50(weights=weights)
        m.fc = torch.nn.Identity()
    m.eval()
    preprocess = weights.transforms()
    return m, preprocess

model, preprocess = get_backbone(BACKBONE)

# ---------- catalog ----------
with open("data/products.json", "r", encoding="utf-8") as f:
    PRODUCTS = json.load(f)
if not PRODUCTS:
    raise RuntimeError("data/products.json is empty. Run precompute.py first.")

VECTORS = np.array([p["vector"] for p in PRODUCTS], dtype=np.float32)   # (N,D) L2-normalized
HISTS   = np.array([p.get("hist", [0]*24) for p in PRODUCTS], dtype=np.float32)
HISTS   = HISTS / (np.linalg.norm(HISTS, axis=1, keepdims=True) + 1e-9)
N = len(PRODUCTS)

# ---------- utilities ----------
def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    import base64 as _b64
    return "data:image/jpeg;base64," + _b64.b64encode(buf.getvalue()).decode("utf-8")

def center_square(pil_img: Image.Image) -> Image.Image:
    s = min(pil_img.width, pil_img.height)
    l = (pil_img.width - s) // 2
    t = (pil_img.height - s) // 2
    return pil_img.crop((l, t, l + s, t + s))

def crop_center_fraction(pil_img: Image.Image, frac=0.8) -> Image.Image:
    frac = max(0.3, min(1.0, float(frac)))
    s = int(min(pil_img.width, pil_img.height) * frac)
    l = (pil_img.width - s) // 2
    t = (pil_img.height - s) // 2
    return pil_img.crop((l, t, l + s, t + s))

def compute_hsv_hist(pil_img, bins=(8, 8, 8), ignore_bg=True):
    img = crop_center_fraction(pil_img, 0.8).convert("HSV")
    h_img, s_img, v_img = img.split()
    import numpy as _np
    h = _np.array(h_img); s = _np.array(s_img); v = _np.array(v_img)
    if ignore_bg:
        mask = (v > 30) & (v < 245)
        if mask.sum() > 0:
            h, s, v = h[mask], s[mask], v[mask]
        else:
            h, s, v = h.ravel(), s.ravel(), v.ravel()
    else:
        h, s, v = h.ravel(), s.ravel(), v.ravel()
    h_hist, _ = _np.histogram(h, bins=bins[0], range=(0, 255))
    s_hist, _ = _np.histogram(s, bins=bins[1], range=(0, 255))
    v_hist, _ = _np.histogram(v, bins=bins[2], range=(0, 255))
    hist = _np.concatenate([h_hist, s_hist, v_hist]).astype("float32")
    hist /= (hist.sum() + 1e-9)
    hist /= (_np.linalg.norm(hist) + 1e-9)
    return hist

def load_query_image(file, url):
    # file upload
    if file and getattr(file, "filename", ""):
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img, pil_to_data_url(img)
    # url / local path
    u = (url or "").strip()
    if not u:
        return None, None
    if os.path.exists(u):
        img = Image.open(u)
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img, pil_to_data_url(img)
    if not u.startswith("http"):
        u = "https://" + u
    content = fetch_bytes(u, timeout=25)
    img = _flatten_to_rgb(content, bg=(255, 255, 255))  # also handles PNG transparency
    return img, pil_to_data_url(img)

@torch.no_grad()
def embed(pil_img):
    x = preprocess(pil_img).unsqueeze(0)     # (1,3,224,224)
    feat = model(x).squeeze(0).cpu().numpy().astype("float32")
    feat /= (np.linalg.norm(feat) + 1e-9)
    return feat

def dynamic_topk(n_items: int) -> int:
    """Scale top-K with catalog size: 25–35% of N, clamped to [200, 600]."""
    k = int(0.30 * n_items)
    return max(200, min(600, k))

# ---------- route ----------
@app.route("/", methods=["GET", "POST"])
def index():
    results, err, preview = [], None, None
    min_score = float(request.form.get("minScore", MIN_SHOW)) if request.method == "POST" else MIN_SHOW
    try:
        if request.method == "POST":
            img_raw, preview = load_query_image(request.files.get("imageFile"), request.form.get("imageUrl"))
            if img_raw is None:
                raise ValueError("Please upload an image or paste an image URL or local path.")
            w, h = img_raw.size
            if min(w, h) < QUERY_MIN_SIDE:
                raise ValueError(f"Image is too small ({w}×{h}). Please use ≥ {QUERY_MIN_SIDE}px on the shorter side.")

            # No background removal — just center crop
            img_feat = center_square(img_raw)
            img_col  = img_feat

            q_vec  = embed(img_feat)
            q_hist = compute_hsv_hist(img_col, ignore_bg=True)

            # stage 1: feature-only top-K (dynamic)
            feat_sims = VECTORS @ q_vec
            k = dynamic_topk(N)
            cand_idx = np.argpartition(-feat_sims, k-1)[:k] if k < N else np.arange(N)

            # stage 2: rerank with small color term
            final_sub = W_FEAT * feat_sims[cand_idx] + W_COLOR * (HISTS[cand_idx] @ q_hist)

            order_local = np.argsort(-final_sub)[:100]
            threshold = max(min_score, MIN_SHOW)

            for j in order_local:
                i = int(cand_idx[j])
                score_abs = float(final_sub[j])
                if score_abs >= threshold:
                    item = dict(PRODUCTS[i])
                    item["score"] = round(score_abs, 3)  # absolute score
                    results.append(item)

    except Exception as e:
        err = str(e)

    ui_min = max(min_score, MIN_SHOW)
    return render_template("index.html", results=results, err=err, uploaded=preview, minScore=ui_min)

if __name__ == "__main__":
    app.run(debug=True)
