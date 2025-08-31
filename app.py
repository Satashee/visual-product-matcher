import io, os, json, hashlib, traceback
from pathlib import Path

import numpy as np
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from PIL import Image, ImageFile, ImageOps
from flask import Flask, render_template, request, send_file, abort

import torch, torchvision

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------- Config -------
W_FEAT   = 0.90
W_COLOR  = 0.10
BACKBONE = "resnet18"  # lightweight backbone
MIN_SHOW = 0.50
QUERY_MIN_SIDE = 256
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = Flask(__name__)

# ---------- Resilient fetcher ----------
SESSION = requests.Session()
retries = Retry(
    total=4,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retries, pool_maxsize=64)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)

def fetch_bytes(url: str, timeout=25) -> bytes:
    r = SESSION.get(url, headers=HEADERS, timeout=timeout, stream=True)
    r.raise_for_status()
    declared = int(r.headers.get("content-length") or 0)
    if declared and declared > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large: {declared} bytes")
    chunk = r.raw.read(MAX_IMAGE_BYTES + 1, decode_content=True)
    if len(chunk) > MAX_IMAGE_BYTES:
        raise ValueError("Image too large")
    return chunk

# ---------- Image proxy ----------
CACHE_DIR = Path("static/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _flatten_to_rgb(data: bytes, bg=(255, 255, 255)) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        back = Image.new("RGBA", rgba.size, bg + (255,))
        im = Image.alpha_composite(back, rgba).convert("RGB")
    else:
        im = im.convert("RGB")
    return im

@app.route("/img")
def img_proxy():
    src = request.args.get("src", "")
    if not src:
        abort(400, description="Missing src")
    if not (src.startswith("http://") or src.startswith("https://")):
        if os.path.exists(src):
            return send_file(src, conditional=True)
        abort(404, description="Local file not found")

    key = hashlib.sha1(src.encode("utf-8")).hexdigest() + ".jpg"
    cached = CACHE_DIR / key
    if not cached.exists():
        try:
            content = fetch_bytes(src, timeout=25)
            im = _flatten_to_rgb(content)
            im.thumbnail((1024, 1024), Image.LANCZOS)
            im.save(cached, "JPEG", quality=90, optimize=True)
        except Exception as e:
            abort(502, description=f"Proxy fetch failed: {e}")
    return send_file(cached, mimetype="image/jpeg", conditional=True)

# ---------- Model ----------
def get_backbone(name="resnet18"):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    m = torchvision.models.resnet18(weights=weights)
    m.fc = torch.nn.Identity()
    m.eval()
    preprocess = weights.transforms()
    return m, preprocess

model, preprocess = get_backbone(BACKBONE)

# ---------- Catalog ----------
try:
    with open("data/products.json", "r", encoding="utf-8") as f:
        PRODUCTS = json.load(f)
    if not PRODUCTS:
        raise RuntimeError("products.json is empty")
except Exception as e:
    print("❌ Error loading catalog:", e)
    PRODUCTS = []

VECTORS = np.array([p["vector"] for p in PRODUCTS], dtype=np.float32)
HISTS   = np.array([p.get("hist", [0]*24) for p in PRODUCTS], dtype=np.float32)
HISTS   = HISTS / (np.linalg.norm(HISTS, axis=1, keepdims=True) + 1e-9)
N = len(PRODUCTS)

# ---------- Utilities ----------
def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def center_square(img): return img.crop(((img.width - s := min(img.size)) // 2,)*2 + ((img.width + s) // 2, (img.height + s) // 2))

def compute_hsv_hist(pil_img, bins=(8, 8, 8), ignore_bg=True):
    img = center_square(pil_img).convert("HSV")
    h_img, s_img, v_img = img.split()
    h, s, v = np.array(h_img), np.array(s_img), np.array(v_img)
    mask = (v > 30) & (v < 245) if ignore_bg else np.ones_like(v, dtype=bool)
    h, s, v = h[mask], s[mask], v[mask]
    hist = np.concatenate([
        np.histogram(h, bins=bins[0], range=(0, 255))[0],
        np.histogram(s, bins=bins[1], range=(0, 255))[0],
        np.histogram(v, bins=bins[2], range=(0, 255))[0],
    ]).astype("float32")
    hist /= (hist.sum() + 1e-9)
    hist /= (np.linalg.norm(hist) + 1e-9)
    return hist

def load_query_image(file, url):
    if file and getattr(file, "filename", ""):
        img = Image.open(file.stream)
        return ImageOps.exif_transpose(img).convert("RGB"), pil_to_data_url(img)
    u = (url or "").strip()
    if not u:
        return None, None
    if os.path.exists(u):
        img = Image.open(u)
        return ImageOps.exif_transpose(img).convert("RGB"), pil_to_data_url(img)
    if not u.startswith("http"):
        u = "https://" + u
    content = fetch_bytes(u)
    img = _flatten_to_rgb(content)
    return img, pil_to_data_url(img)

@torch.no_grad()
def embed(pil_img):
    x = preprocess(pil_img).unsqueeze(0)
    feat = model(x).squeeze(0).cpu().numpy().astype("float32")
    feat /= (np.linalg.norm(feat) + 1e-9)
    return feat

def dynamic_topk(n): return max(200, min(600, int(0.3 * n)))

# ---------- Main Route ----------
@app.route("/", methods=["GET", "POST"])
def index():
    results, err, preview = [], None, None
    min_score = float(request.form.get("minScore", MIN_SHOW)) if request.method == "POST" else MIN_SHOW
    try:
        if request.method == "POST":
            img_raw, preview = load_query_image(request.files.get("imageFile"), request.form.get("imageUrl"))
            if img_raw is None:
                raise ValueError("Upload an image or provide a valid URL.")
            if min(img_raw.size) < QUERY_MIN_SIDE:
                raise ValueError("Image too small (min side < 256px)")

            img_feat = center_square(img_raw)
            q_vec  = embed(img_feat)
            q_hist = compute_hsv_hist(img_feat, ignore_bg=True)

            feat_sims = VECTORS @ q_vec
            k = dynamic_topk(N)
            cand_idx = np.argpartition(-feat_sims, k-1)[:k] if k < N else np.arange(N)

            final_sub = W_FEAT * feat_sims[cand_idx] + W_COLOR * (HISTS[cand_idx] @ q_hist)
            order_local = np.argsort(-final_sub)[:100]

            for j in order_local:
                i = int(cand_idx[j])
                score = float(final_sub[j])
                if score >= min_score:
                    item = dict(PRODUCTS[i])
                    item["score"] = round(score, 3)
                    results.append(item)

    except Exception as e:
        err = str(e)
        print("🔥 INTERNAL SERVER ERROR:", e)
        traceback.print_exc()

    return render_template("index.html", results=results, err=err, uploaded=preview, minScore=min_score)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
