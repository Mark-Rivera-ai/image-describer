# Image Describer (Azure AI Vision)

Small, showcase-ready CLI that takes images (file path or URL), calls **Azure AI Vision** for a **caption** + **tags**, and prints tidy results to the console — with batch mode, retries, and JSONL/CSV export.

## Features

* Single image → caption + top-N tags (with confidence)
* Batch mode (folder or list of URLs)
* Output: console + `out/results.jsonl` (one JSON object per line)
* Optional CSV export to `out/results.csv`
* Graceful errors (bad file/URL, 401/415, timeouts) with retries & backoff
* Optional confidence filtering via `--threshold`
* Option to skip per-image JSON export with `--no-raw`

## Architecture

```
[CLI (Python)]
    │  paths/URLs
    ▼
[Azure AI Vision endpoint]
    │  caption, tags, confidences (JSON)
    ▼
[Console output + optional JSON/CSV log]
```

## Quickstart

1. **Create Azure resource**: Azure AI Services (or Azure AI Vision).
   Grab `VISION_ENDPOINT` and `VISION_KEY` from the Keys & Endpoint blade.

2. **Setup** (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your endpoint + key
```

3. **Run**

```bash
# Single local file
python describe.py --image samples/dog.jpg

# Folder of images
python describe.py --dir samples/ --top-k 7

# URLs from a text file (one per line)
python describe.py --urls urls.txt --out out/results.jsonl --csv

# With confidence filtering or skipping raw JSON
python describe.py --dir samples --threshold 0.3 --no-raw
```

## Additional CLI Options

* `--threshold <float>`: Hide tags below a given confidence (e.g., `--threshold 0.3`).
* `--no-raw`: Skip writing per-image JSON files (useful for large batches).

## Performance & Stability

* Introduces a random delay (2–6 seconds) between API calls to respect rate limits.
* Rotates HTTP User-Agent headers when fetching images from URLs.
* Includes exponential backoff retries for transient 429/5xx responses.
* Displays progress with a `tqdm` progress bar.

## Output Schema (CSV)

| Column             | Description           |
| ------------------ | --------------------- |
| source             | File path or URL      |
| caption            | Generated description |
| caption_confidence | Confidence score      |
| tag_1..tag_N       | Top N predicted tags  |

## Sample set

Put a few test images in `samples/` (e.g., `.jpg`, `.png`). A tiny set of 5–10 is perfect for demos.

## Notes

* Uses SDK `azure-ai-vision-imageanalysis` (`ImageAnalysisClient`).
* Feature flags: caption + tags.
* Retries on 429/5xx with exponential backoff.
* JSON Lines file is append-only; you can safely re-run to grow the log.
* Filenames are sanitized automatically for safe storage.
* Prints clear `ERROR:` messages inline for failed files.

## Environment

Create `.env` from `.env.example`:

```
VISION_ENDPOINT=...
VISION_KEY=...
```

## Outputs

* Raw per-image JSON: `out/<slug>.json`
* Aggregated JSONL: `out/results.jsonl`
* Optional CSV: `out/results.csv`

## LinkedIn-ready bullets

* Built a Python Image Describer with Azure AI Vision captions & tags.
* Supports batch mode, JSONL/CSV logs, retries & rate-limit handling.
* Useful for content tagging, accessibility alt-text drafts, and quick dataset labeling.
