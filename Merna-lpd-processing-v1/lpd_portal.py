import argparse
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import quote

from flask import Flask, abort, render_template_string, send_file


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
app = Flask(__name__)
RESULTS_ROOT = None


def default_results_root():
    for path in (Path("images/preprocessing_results"), Path("preprocessing_results")):
        if path.is_dir():
            return path.resolve()
    return Path("images/preprocessing_results").resolve()


def set_results_root(path_value):
    global RESULTS_ROOT
    RESULTS_ROOT = Path(path_value).resolve()


def root():
    global RESULTS_ROOT
    if RESULTS_ROOT is None:
        RESULTS_ROOT = default_results_root()
    return RESULTS_ROOT


def read_result_json(folder):
    path = folder / "result.json"
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def media_url(path):
    if path is None:
        return None
    try:
        rel = path.resolve().relative_to(root())
    except ValueError:
        return None
    return "/media/" + quote(rel.as_posix())


def first_match(folder, stem):
    matches = sorted(path for path in folder.glob(stem + ".*") if is_image(path))
    return matches[0] if matches else None


def first_lpd_crop(folder):
    candidates = []
    for path in sorted(folder.iterdir()):
        if not is_image(path):
            continue
        if path.stem == "lpd" or (path.stem.startswith("lpd_") and path.stem[4:].isdigit()):
            candidates.append(path)
    return candidates[0] if candidates else None


def scan_items():
    base = root()
    if not base.is_dir():
        return []

    items = []
    for folder in sorted(path for path in base.iterdir() if path.is_dir()):
        data = read_result_json(folder)
        items.append({
            "name": folder.name,
            "folder": str(folder),
            "original": media_url(first_match(folder, "original_image")),
            "lpd": media_url(first_lpd_crop(folder)),
            "lpd_after": media_url(first_match(folder, "lpd_after_preprocessing")),
            "final": media_url(first_match(folder, "image_after_preprocessing")),
            "has_plate": bool(data.get("has_plate")) if data else False,
            "count": data.get("detected_plate_count", 0),
            "pasted": data.get("pasted_filtered_lpd_count", 0),
            "error": data.get("error") if data else None,
        })
    return items


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LPD Results Portal</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --line: #d7e1ee;
      --line-strong: #b8c7d9;
      --text: #101828;
      --muted: #667085;
      --ok: #067647;
      --warn: #b54708;
      --bad: #b42318;
      --shadow: 0 12px 30px rgba(16, 24, 40, .08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(244, 247, 251, .96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }

    .bar {
      max-width: 1600px;
      margin: 0 auto;
      padding: 16px 22px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }

    h1 { margin: 0; font-size: 24px; line-height: 1.2; }
    .path { margin-top: 5px; color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }

    .tools { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    input {
      min-width: 260px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      padding: 9px 12px;
      font-size: 14px;
      background: #fff;
    }

    .pill {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 700;
    }

    main { max-width: 1600px; margin: 0 auto; padding: 18px 22px 40px; }

    .result {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      margin-bottom: 96px;
      overflow: hidden;
    }

    .head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfdff;
      flex-wrap: wrap;
    }

    h2 { margin: 0; font-size: 20px; line-height: 1.2; overflow-wrap: anywhere; }

    .badges { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .badge {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 800;
      background: #f8fafc;
    }
    .ok { color: var(--ok); background: #ecfdf3; border-color: #abefc6; }
    .warn { color: var(--warn); background: #fffaeb; border-color: #fedf89; }
    .bad { color: var(--bad); background: #fef3f2; border-color: #fecdca; }

    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      column-gap: 24px;
      row-gap: 48px;
      padding: 16px;
    }

    .tile {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
      min-width: 0;
    }

    .tile h3 {
      margin: 0;
      padding: 11px 12px;
      font-size: 15px;
      line-height: 1.2;
      border-bottom: 1px solid var(--line);
    }

    .imgbox {
      height: 280px;
      background: #eef3f8;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .imgbox img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .missing { color: var(--muted); padding: 18px; text-align: center; }

    @media (max-width: 1200px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 700px) {
      .bar, main { padding-left: 12px; padding-right: 12px; }
      .grid { grid-template-columns: 1fr; padding: 12px; }
      input { min-width: 100%; }
      .imgbox { height: 230px; }
      .badges { justify-content: flex-start; }
    }
  </style>
</head>
<body>
<header>
  <div class="bar">
    <div>
      <h1>LPD Results Portal</h1>
      <div class="path">{{ root_path }}</div>
    </div>
    <div class="tools">
      <input id="search" type="search" placeholder="Search image folder">
      <div class="pill">Images: {{ items|length }}</div>
      <div class="pill">With Plate: {{ with_plate }}</div>
    </div>
  </div>
</header>
<main>
  {% if not items %}
    <section class="result"><div class="head"><h2>No result folders found</h2></div></section>
  {% endif %}

  {% for item in items %}
  <section class="result" data-name="{{ item.name|lower }}">
    <div class="head">
      <h2>{{ item.name }}</h2>
      <div class="badges">
        {% if item.error %}
          <span class="badge bad">ERROR</span>
        {% elif item.has_plate %}
          <span class="badge ok">PLATE FOUND</span>
        {% else %}
          <span class="badge warn">NO PLATE</span>
        {% endif %}
        <span class="badge">LPD: {{ item.count }}</span>
        <span class="badge">PASTED: {{ item.pasted }}</span>
      </div>
    </div>

    <div class="grid">
      <article class="tile">
        <h3>Original Image</h3>
        <div class="imgbox">{% if item.original %}<img src="{{ item.original }}" alt="Original image for {{ item.name }}">{% else %}<div class="missing">Missing</div>{% endif %}</div>
      </article>
      <article class="tile">
        <h3>LPD Cropped</h3>
        <div class="imgbox">{% if item.lpd %}<img src="{{ item.lpd }}" alt="LPD crop for {{ item.name }}">{% else %}<div class="missing">Missing</div>{% endif %}</div>
      </article>
      <article class="tile">
        <h3>LPD After Preprocessing</h3>
        <div class="imgbox">{% if item.lpd_after %}<img src="{{ item.lpd_after }}" alt="LPD after preprocessing for {{ item.name }}">{% else %}<div class="missing">Missing</div>{% endif %}</div>
      </article>
      <article class="tile">
        <h3>Final Image</h3>
        <div class="imgbox">{% if item.final %}<img src="{{ item.final }}" alt="Final image for {{ item.name }}">{% else %}<div class="missing">Missing</div>{% endif %}</div>
      </article>
    </div>
  </section>
  {% endfor %}
</main>
<script>
  const search = document.getElementById('search');
  const cards = Array.from(document.querySelectorAll('.result'));
  search.addEventListener('input', () => {
    const value = search.value.trim().toLowerCase();
    cards.forEach(card => {
      card.style.display = card.dataset.name.includes(value) ? '' : 'none';
    });
  });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    items = scan_items()
    return render_template_string(
        HTML,
        items=items,
        root_path=str(root()),
        with_plate=sum(1 for item in items if item["has_plate"]),
    )


@app.route("/media/<path:relpath>")
def media(relpath):
    base = root()
    path = (base / relpath).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        abort(404)
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype=mimetypes.guess_type(str(path))[0])


def parse_args():
    parser = argparse.ArgumentParser(description="View LPD preprocessing result folders in Flask.")
    parser.add_argument("--results-root", default=os.getenv("RESULTS_ROOT"), help="Result root folder.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.results_root:
        set_results_root(args.results_root)
    else:
        root()
    app.run(host=args.host, port=args.port, debug=args.debug)




