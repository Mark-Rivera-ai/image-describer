import os
import sys
import argparse
import json
import time
import logging
import re
import random
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

# ------------- Env & Client -------------
load_dotenv()

VISION_ENDPOINT = os.getenv("VISION_ENDPOINT", "").strip()
VISION_KEY = os.getenv("VISION_KEY", "").strip()

if not VISION_ENDPOINT or not VISION_KEY:
    print("ERROR: VISION_ENDPOINT and VISION_KEY must be set (see .env).", file=sys.stderr)
    sys.exit(2)

client = ImageAnalysisClient(
    endpoint=VISION_ENDPOINT,
    credential=AzureKeyCredential(VISION_KEY)
)

# ------------- Helpers -------------
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def iter_images_from_dir(directory: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            yield p

def iter_urls_from_file(file: Path) -> Iterable[str]:
    for line in file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        yield s

def sanitize_filename(url: str) -> str:
    filename = re.sub(r'[\\/*?:"<>|]', '_', url)
    return filename[:100]

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/120.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36',
]

def fetch_image_url_with_headers(url: str, session: requests.Session) -> str:
    headers = {
        'User-Agent': random.choice(user_agents),
        'Referer': url
    }
    response = session.get(url, headers=headers, allow_redirects=True)
    if 'image' in response.headers.get('Content-Type', ''):
        return response.url
    raise Exception(f"URL did not resolve to a valid image: {url}")


# ------------- Core Vision Calls -------------
def analyze_one(source: str, session: requests.Session, retries: int = 4, timeout: int = 30) -> Dict[str, Any]:
    attempt = 0
    backoff = 1.5

    while True:
        time.sleep(random.uniform(2, 6))
        try:
            if source.startswith("http://") or source.startswith("https://"):
                source = fetch_image_url_with_headers(source, session=session)
                result = client.analyze_from_url(
                    image_url=source,
                    visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS]
                )
            else:
                with open(source, "rb") as f:
                    result = client.analyze(
                        image_data=f,
                        visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS]
                    )

            caption_text = getattr(result.caption, "text", None) or getattr(result.caption, "content", None)
            caption_conf = getattr(result.caption, "confidence", None)

            tags_list = []
            tags_obj = getattr(result, "tags", None)
            if tags_obj is not None:
                raw_tags = []
                if hasattr(tags_obj, "values"):
                    v = getattr(tags_obj, "values")
                    raw_tags = v() if callable(v) else v
                elif hasattr(tags_obj, "list"):
                    l = getattr(tags_obj, "list")
                    raw_tags = l() if callable(l) else l
                elif isinstance(tags_obj, (list, tuple)):
                    raw_tags = tags_obj

                for t in raw_tags:
                    name = getattr(t, "name", None) or getattr(t, "text", None)
                    conf = getattr(t, "confidence", None) or getattr(t, "score", None)
                    if name is not None:
                        tags_list.append({
                            "name": name,
                            "confidence": float(conf) if conf is not None else None
                        })

            return {
                "source": source,
                "success": True,
                "error": None,
                "caption": {"text": caption_text, "confidence": caption_conf},
                "tags": sorted(tags_list, key=lambda x: (x["confidence"] is not None, x["confidence"]), reverse=True),
                "raw": result.as_dict() if hasattr(result, "as_dict") else None
            }

        except (HttpResponseError, ServiceRequestError, ServiceResponseError) as e:
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            retriable = status in (429, 500, 502, 503, 504) or isinstance(e, (ServiceRequestError, ServiceResponseError))
            if attempt < retries and retriable:
                time.sleep(backoff ** (attempt + 1))
                attempt += 1
                continue
            return {"source": source, "success": False, "error": f"{type(e).__name__}: {str(e)}", "caption": None, "tags": [], "raw": None}
        except Exception as e:
            return {"source": source, "success": False, "error": f"{type(e).__name__}: {str(e)}", "caption": None, "tags": [], "raw": None}

# ------------- CLI & Main Execution -------------
def print_result(entry: Dict[str, Any], top_k: int = 5, threshold: Optional[float] = None) -> None:
    print(f"\n== {entry['source']}")
    if entry.get("success"):
        cap = entry.get("caption") or {}
        if cap.get("text"):
            conf = cap.get("confidence")
            conf_s = f" ({conf:.2f})" if isinstance(conf, (int, float)) else ""
            print(f"Caption: {cap['text']}{conf_s}")
        tags = entry.get("tags", [])
        if threshold is not None:
            tags = [t for t in tags if t.get("confidence") is None or t.get("confidence", 0) >= threshold]
        for t in tags[:top_k]:
            cnf = t.get("confidence")
            cnf_s = f" ({cnf:.2f})" if isinstance(cnf, (int, float)) else ""
            print(f"- {t.get('name')}{cnf_s}")
    else:
        print(f"ERROR: {entry.get('error')}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Describe images (caption + tags) using Azure AI Vision")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Path to a single image file")
    g.add_argument("--dir", type=str, help="Directory of images (recursively)")
    g.add_argument("--urls", type=str, help="Text file with one image URL per line")

    p.add_argument("--top-k", type=int, default=5, help="Top-K tags to show")
    p.add_argument("--threshold", type=float, default=None, help="Minimum confidence for tags (e.g., 0.30)")
    p.add_argument("--out", type=str, default="out/results.jsonl", help="Path to JSONL output")
    p.add_argument("--csv", action="store_true", help="Also write a CSV summary")
    p.add_argument("--no-raw", action="store_true", help="Skip per-image raw JSON files")
    return p.parse_args()


def main():
    args = parse_args()

    sources: List[str] = []
    if args.image:
        sources = [args.image]
    elif args.dir:
        sources = [str(p) for p in iter_images_from_dir(Path(args.dir))]
    elif args.urls:
        sources = list(iter_urls_from_file(Path(args.urls)))

    if not sources:
        print("No images found. Check your inputs.", file=sys.stderr)
        sys.exit(1)

    jsonl_path = Path(args.out)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    with requests.Session() as session:
        for src in tqdm(sources, desc="Analyzing", unit="img"):
            entry = analyze_one(src, session=session)
            print_result(entry, top_k=args.top_k, threshold=args.threshold)
            results.append(entry)
            if not args.no_raw:
                slug = sanitize_filename(entry["source"])
                with open(OUT_DIR / f"{slug}.json", "w", encoding="utf-8") as f:
                    json.dump(entry, f, ensure_ascii=False, indent=2)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if args.csv:
        from csv import writer
        csv_path = Path("out/results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = writer(f)
            header = ["source", "caption", "caption_confidence"] + [f"tag_{i+1}" for i in range(args.top_k)]
            w.writerow(header)
            for e in results:
                if not e.get("success"):
                    w.writerow([e.get("source"), "", ""] + [""] * args.top_k)
                    continue
                cap = e.get("caption") or {}
                tags = [t.get("name", "") for t in e.get("tags", [])[:args.top_k]]
                row = [e.get("source"), cap.get("text", ""), cap.get("confidence", "")] + tags
                w.writerow(row[:len(header)])

    print(f"\nDone. {sum(1 for r in results if r.get('success'))}/{len(results)} succeeded. JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()

