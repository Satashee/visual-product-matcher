# precompute.py — mixed online → TTA features + HSV + light dedupe (NO background removal)
import io, os, json, pathlib, random, argparse, time
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFile, ImageOps
import torch, torchvision

ImageFile.LOAD_TRUNCATED_IMAGES = True

OUT_PATH = pathlib.Path("data/products.json")
IMG_DIR  = pathlib.Path("static/images")
HEADERS  = {"User-Agent": "Mozilla/5.0"}
MIN_ONLINE_ITEMS = 30

parser = argparse.ArgumentParser()
parser.add_argument("--offline", action="store_true", help="Generate synthetic images if online fails")
parser.add_argument("--online", choices=["dummyjson","escuelajs","fakestore","mixed"], default="mixed")
parser.add_argument("--items", type=int, default=600, help="Target items to fetch")
parser.add_argument("--min-size", type=int, default=128, help="Skip images smaller than this (short side)")
parser.add_argument("--max-items", type=int, default=800, help="Upper bound before embedding")
parser.add_argument("--cap-per-category", type=int, default=0, help="0 = no cap")
parser.add_argument("--backbone", choices=["resnet18","mobilenetv3"], default="resnet18")
args = parser.parse_args()

def get_backbone(name="mobilenetv3"):
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

# -------- data sources --------
def fetch_dummyjson(n):
    url = f"https://dummyjson.com/products?limit={n}"
    r = requests.get(url, timeout=30, headers=HEADERS); r.raise_for_status()
    items = r.json().get("products", [])
    out = []
    for it in items:
        img = (it.get("thumbnail") or (it.get("images") or [""])[0]).strip()
        if not img: continue
        out.append({"id": it.get("id"), "name": it.get("title","Item"), "category": it.get("category","misc"), "image_url": img})
    return out

def fetch_escuelajs(n):
    url = f"https://api.escuelajs.co/api/v1/products?offset=0&limit={n}"
    r = requests.get(url, timeout=30, headers=HEADERS); r.raise_for_status()
    out = []
    for it in r.json():
        images = it.get("images") or []
        img = next((im for im in images if isinstance(im,str) and im.startswith("http")), "")
        if not img: continue
        cat = it.get("category")
        cat_name = cat.get("name","misc") if isinstance(cat, dict) else (cat or "misc")
        out.append({"id": it.get("id"), "name": it.get("title","Item"), "category": cat_name, "image_url": img})
    return out

def fetch_fakestore(n):
    r = requests.get("https://fakestoreapi.com/products", timeout=30, headers=HEADERS); r.raise_for_status()
    items = r.json()[:n]
    out = []
    for it in items:
        img = (it.get("image") or "").strip()
        if not img: continue
        out.append({"id": it.get("id"), "name": it.get("title","Item"), "category": it.get("category","misc"), "image_url": img})
    return out

def fetch_online_mixed(total):
    targets = [("escuelajs", min(500, total)),
               ("dummyjson", min(300, total//2)),
               ("fakestore", min(100, total//6))]
    seen, merged = set(), []
    for name, want in targets:
        if want <= 0: continue
        try:
            batch = fetch_escuelajs(want) if name=="escuelajs" else fetch_dummyjson(want) if name=="dummyjson" else fetch_fakestore(want)
        except Exception as e:
            print(f"[warn] {name} fetch failed:", e); batch = []
        for x in batch:
            url = x["image_url"]
            if url in seen: continue
            seen.add(url); merged.append(x)
    print(f"Using MIXED online dataset: {len(merged)} items before trimming")
    return merged

# -------- offline synthetic --------
def make_swatch(w=320,h=320):
    img = Image.new("RGB",(w,h),(random.randint(30,220),random.randint(30,220),random.randint(30,220)))
    d = ImageDraw.Draw(img)
    for x in range(0,w,random.randint(20,60)):
        d.rectangle([x,0,x+10,h], fill=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    return img

def build_offline(n):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    cats = ["Footwear","Electronics","Accessories","Home","Apparel","Beauty","Outdoors","Toys","Kitchen","Books"]
    out=[]
    for i in range(1,n+1):
        img = make_swatch()
        path = IMG_DIR / f"prod_{i}.jpg"
        img.save(path,"JPEG",quality=90)
        out.append({"id": i, "name": f"Item {i:02d}", "category": random.choice(cats), "image_url": str(path).replace("\\","/")})
    print(f"Using OFFLINE dataset: {len(out)} images in {IMG_DIR}")
    return out

# -------- helpers --------
def _flatten_to_rgb(im: Image.Image, bg=(255,255,255)) -> Image.Image:
    """If image has alpha, composite onto white; else ensure RGB."""
    if im.mode in ("RGBA","LA") or (im.mode=="P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        back = Image.new("RGBA", rgba.size, bg + (255,))
        return Image.alpha_composite(back, rgba).convert("RGB")
    return im.convert("RGB")

def open_image(src):
    try:
        if src.startswith("http"):
            for attempt in range(2):
                try:
                    b = requests.get(src, timeout=25, headers=HEADERS).content
                    im = Image.open(io.BytesIO(b))
                    return _flatten_to_rgb(im, bg=(255,255,255))
                except Exception:
                    if attempt==1: raise
                    time.sleep(0.5)
        else:
            im = Image.open(src)
            return _flatten_to_rgb(im, bg=(255,255,255))
    except Exception as e:
        print("  [warn] bad image:", e)
    return None

def is_small(img, min_side): return min(img.size) < int(min_side)

def center_square(im: Image.Image) -> Image.Image:
    s = min(im.width, im.height); l=(im.width-s)//2; t=(im.height-s)//2
    return im.crop((l,t,l+s,t+s))

def compute_hsv_hist(im, bins=(8,8,8), ignore_bg=True):
    img = center_square(im).convert("HSV")
    h_img,s_img,v_img = img.split()
    h=np.array(h_img); s=np.array(s_img); v=np.array(v_img)
    if ignore_bg:
        m=(v>30)&(v<245)
        if m.sum()>0: h,s,v=h[m],s[m],v[m]
        else: h,s,v=h.ravel(),s.ravel(),v.ravel()
    else:
        h,s,v=h.ravel(),s.ravel(),v.ravel()
    hh,_=np.histogram(h,bins=bins[0],range=(0,255))
    sh,_=np.histogram(s,bins=bins[1],range=(0,255))
    vh,_=np.histogram(v,bins=bins[2],range=(0,255))
    hist=np.concatenate([hh,sh,vh]).astype("float32")
    hist/= (hist.sum()+1e-9); hist/= (np.linalg.norm(hist)+1e-9)
    return hist

@torch.no_grad()
def embed_tta(model, preprocess, im):
    views=[im, ImageOps.mirror(im)]
    vs=[]
    for v in views:
        x=preprocess(v).unsqueeze(0)
        f=model(x).squeeze(0).cpu().numpy().astype("float32")
        f/= (np.linalg.norm(f)+1e-9); vs.append(f)
    vec=np.mean(vs,axis=0); vec/= (np.linalg.norm(vec)+1e-9); return vec

def dedupe_by_feature(vecs, thr=0.985):
    keep, kept=[], []
    for i,v in enumerate(vecs):
        if not kept: keep.append(i); kept.append(v); continue
        sims = np.dot(np.stack(kept), v)
        if sims.max() < thr: keep.append(i); kept.append(v)
    return keep

# -------- main --------
def main():
    random.seed(7)
    products=None
    if not args.offline:
        try:
            if args.online=="dummyjson": products=fetch_dummyjson(args.items)
            elif args.online=="escuelajs": products=fetch_escuelajs(args.items)
            elif args.online=="fakestore": products=fetch_fakestore(args.items)
            else: products=fetch_online_mixed(args.items)
        except Exception as e:
            print("Online fetch failed:", e)
    if not products or len(products)<MIN_ONLINE_ITEMS:
        print("Online dataset too small; switching to OFFLINE.")
        products=build_offline(args.items)

    if args.max_items and len(products)>args.max_items:
        rng=random.Random(42); products=rng.sample(products, args.max_items)
    print(f"Ingesting {len(products)} items")

    model, preprocess = get_backbone(args.backbone)
    vecs, meta=[], []

    for idx,p in enumerate(products,1):
        img_raw=open_image(p["image_url"])
        if img_raw is None: continue
        if is_small(img_raw, args.min_size): 
            continue

        # NO background removal — just center crop
        img_sq = center_square(img_raw)

        try:
            v  = embed_tta(model, preprocess, img_sq)
            h  = compute_hsv_hist(img_sq, ignore_bg=True)
        except Exception as e:
            print("  [warn] embedding failed:", e); continue

        q=dict(p); q["vector"]=v.tolist(); q["hist"]=h.tolist()
        meta.append(q); vecs.append(v)
        if idx % 25 == 0: print(f"Processed {idx}/{len(products)}")

    print("De-duplicating…")
    if vecs:
        vecs=np.array(vecs,dtype=np.float32)
        keep=dedupe_by_feature(vecs, thr=0.985)
        out=[meta[i] for i in keep]
    else:
        out=[]

    if args.cap_per_category>0:
        buckets,balanced={},[]
        for item in out:
            c=item.get("category","misc")
            if buckets.get(c,0)<args.cap_per_category:
                balanced.append(item); buckets[c]=buckets.get(c,0)+1
        out=balanced

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH,"w",encoding="utf-8") as f: json.dump(out,f)
    print(f"Saved {OUT_PATH} with {len(out)} items.")

if __name__=="__main__":
    main()
