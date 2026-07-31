"""Auto-scale Swin2SR: pick x2 or x4 from an image's resolution and type.

Given an image and its kind ("car" or "plate"), choose the smallest upscale that
lifts both sides to the kind's target resolution, then run the matching local
Swin2SR checkpoint (models/sr/swin2sr_{2x,4x}.bin). One 4x pass cannot lift a very
small crop to target, so the result is re-measured and fed back through until it
clears target. An image already at/above target is returned unchanged.

Run: uv run test/1_swin2sr.py <input_folder> <output_folder> --type plate
"""

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import fire
import numpy as np

# ==========================================================================
# Inlined from helpers/, deblurring/ and denoising/ so this file stands alone.
# ==========================================================================

# ===========================================================================
# Files & CLI
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def resolve_dir(arg: str) -> Path:
    """Resolve a folder argument against the current directory (absolute paths kept)."""
    return Path(arg).expanduser().resolve()


def pick_device(device: str | None = None) -> str:
    """The requested device, or cuda when one is available. Torch is imported lazily."""
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_io(input_folder: str, output_folder: str) -> tuple[Path, Path]:
    """Resolve a script's two folder arguments, failing fast if the input is missing."""
    input_dir = resolve_dir(input_folder)
    if not input_dir.is_dir():
        raise SystemExit(f"input folder not found: {input_dir}")
    return input_dir, resolve_dir(output_folder)


def iter_images(folder: Path) -> list[Path]:
    """All image files under *folder* (recursive), sorted."""
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def atomic_save(out_path: Path, save: Callable[[Path], object]) -> None:
    """Write via a temp file, then atomically replace *out_path*.

    A run killed mid-write leaves at most a stray temp file, never a partial
    output — so `out_path.exists()` reliably means "fully done" and a re-run can
    safely skip it. The temp keeps the real suffix so format-by-extension encoders
    (e.g. cv2.imencode) still pick the right format.
    """
    tmp = out_path.with_name(f".{out_path.stem}.part{out_path.suffix}")
    try:
        save(tmp)
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def copy_original(path: Path, out_dir: Path) -> Path:
    """Copy *path* unchanged into *out_dir* (same filename), atomically; return dest."""
    dest = out_dir / path.name
    atomic_save(dest, lambda tmp: shutil.copyfile(path, tmp))
    return dest


# ===========================================================================
# Image I/O (cv2)
# ===========================================================================
# read_image / write_image use np.fromfile + cv2.imdecode/imencode rather than
# cv2.imread/imwrite so non-ASCII paths work on Windows (a known cv2 gotcha). In
# memory images are RGB (matching what the models and skimage expect); the BGR
# order cv2 uses on disk is confined to these two functions.


def read_image(path: Path, *, gray: bool = False) -> np.ndarray:
    """Decode *path* as a uint8 image: (H, W) grayscale or (H, W, 3) RGB.

    Raises if the file cannot be decoded.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path: Path, arr: np.ndarray) -> None:
    """Atomically encode a uint8 image (grayscale H,W or RGB H,W,3) to *path*."""
    out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
    ext = path.suffix or ".png"

    def _save(tmp: Path) -> None:
        ok, buf = cv2.imencode(ext, out)
        if not ok:
            raise ValueError(f"Failed to encode image: {path}")
        buf.tofile(str(tmp))

    atomic_save(path, _save)


# ===========================================================================
# Batch runner
# ===========================================================================


def process_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    output_name: Callable[[Path], str],
    process: Callable[[Path, Path], str],
    copy_reason: Callable[[Path], str | None] = lambda _src: None,
    skip_existing: bool = True,
    check_collisions: bool = True,
) -> None:
    """Run *process* over every image in *input_dir* -> *output_dir*.

    - ``output_name(src)`` -> the produced file's name.
    - ``process(src, dst)`` -> do the work and write *dst* (via atomic_save);
      return a short status string for the log. Called only for non-skipped,
      non-copied files, so any lazy model load inside it never runs on a copy-only
      pass.
    - ``copy_reason(src)`` -> a reason string to copy the source unchanged instead
      of processing it (e.g. too large / won't fit VRAM), or None to process.
    - ``skip_existing`` -> when True a source whose output (or an as-is copy under
      ``src.name``) already exists is skipped, so a re-run resumes. Worth it only
      for slow scripts (models, BM3D); fast scripts pass False and just re-process,
      avoiding the surprise of a stale skip.
    - ``check_collisions`` -> when True, abort up front if two inputs map to the
      same output name (e.g. foo.jpg and foo.png -> foo_x.png), naming the clashing
      files instead of silently overwriting one. Pass False for pipeline scripts.
    """
    files = iter_images(input_dir)
    if not files:
        raise SystemExit(f"no images in {input_dir}")

    if check_collisions:
        by_name: dict[str, list[Path]] = {}
        for src in files:
            by_name.setdefault(output_name(src), []).append(src)
        clashes = {name: srcs for name, srcs in by_name.items() if len(srcs) > 1}
        if clashes:
            lines = [
                f"  {name} <- " + ", ".join(str(s.relative_to(input_dir)) for s in srcs)
                for name, srcs in sorted(clashes.items())
            ]
            raise SystemExit("multiple inputs map to the same output name:\n" + "\n".join(lines))

    def done(src: Path) -> bool:
        return skip_existing and (
            (output_dir / output_name(src)).exists() or (output_dir / src.name).exists()
        )

    if skip_existing and all(done(f) for f in files):
        print(f"All {len(files)} outputs already exist in {output_dir}; nothing to do.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files, 1):
        if done(src):
            print(f"[{i}/{len(files)}] {src.name} (skip, exists)")
            continue
        reason = copy_reason(src)
        if reason is not None:
            copy_original(src, output_dir)
            print(f"[{i}/{len(files)}] {src.name} COPIED as-is ({reason})")
            continue
        status = process(src, output_dir / output_name(src))
        print(f"[{i}/{len(files)}] {src.name} {status}")

    print(f"Done. {len(files)} images -> {output_dir}")


# scale -> (checkpoint under models/sr/, Swin2SRConfig fields that differ from the
# defaults). x2 is classical (caidas/swin2SR-classical-sr-x2-64); x4 is real-world
# (…-realworld-sr-x4-…). Every other tensor matches Swin2SR's defaults (embed_dim
# 180, depths/heads [6]x6, window 8), so only upscale + reconstruction tail are
# overridden; load_swin2sr_local's strict load then asserts the architecture matches.
MODELS_DIR = PROJECT_ROOT / "wetransfer_swin2sr_2x-bin_2026-07-28_0652"


SWIN2SR_SCALES = {
    2: ("swin2sr_2x.bin", {"upscale": 2, "upsampler": "pixelshuffle"}),
    4: ("swin2sr_4x.bin", {"upscale": 4, "upsampler": "nearest+conv"}),
}


def load_swin2sr_local(weights: Path, config: dict, device: str):
    """Rebuild a Swin2SR model offline and strict-load a local checkpoint.

    *config* overrides the Swin2SRConfig fields that differ from the defaults
    (e.g. {"upscale": 4, "upsampler": "nearest+conv"}); every other tensor must
    match, so the strict load asserts the architecture fits the file exactly.
    Returns (model.eval() on *device*, Swin2SRImageProcessor()).
    """
    import torch
    from transformers import (
        AutoModelForImageToImage,
        Swin2SRConfig,
        Swin2SRImageProcessor,
    )

    weights = Path(weights)
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")
    model = AutoModelForImageToImage.from_config(Swin2SRConfig(**config))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"weight/architecture mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected (e.g. {(missing or unexpected)[:3]})"
        )
    return model.to(device).eval(), Swin2SRImageProcessor()


def to_array(out) -> np.ndarray:
    """(1, C, H, W) tensor in [0, 1] -> uint8 image (H,W) grayscale or (H,W,3) RGB."""
    arr = out.squeeze(0).clamp_(0, 1).cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr[:, :, 0] if arr.shape[2] == 1 else arr


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets of `tile`-wide windows covering [0, length) with `overlap`
    between neighbours; the last window is clamped to end exactly at `length`."""
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)  # cover the remainder strip
    return starts


def _blend_ramp(n: int, ramp: int) -> np.ndarray:
    """1-D feather weight of length *n*: linear ramp up over the first *ramp*
    samples, flat 1 in the middle, ramp down over the last *ramp*. Strictly > 0
    everywhere so a normalised blend never divides by zero, even on a border tile
    whose outer ramp overlaps no neighbour."""
    w = np.ones(n, dtype=np.float32)
    if ramp > 0:
        edge = (np.arange(min(ramp, n), dtype=np.float32) + 1.0) / (ramp + 1.0)
        w[: edge.size] = np.minimum(w[: edge.size], edge)
        w[-edge.size :] = np.minimum(w[-edge.size :], edge[::-1])
    return w


def swin2sr_forward(
    model,
    processor,
    image: np.ndarray,
    scale: int,
    device: str,
    *,
    tile: int = 0,
    overlap: int = 16,
) -> np.ndarray:
    """Upscale a uint8 *image* by *scale* with a loaded Swin2SR (model, processor).

    ``tile <= 0`` (or an image already within one tile) runs a single whole-image
    forward pass — byte-identical to calling the model directly. ``tile > 0`` cuts
    the image into ``tile x tile`` patches with an *overlap* px halo, upscales each
    independently, and feather-blends the results, so peak VRAM tracks the tile size
    rather than the image size and OOM is bounded regardless of input resolution.
    The blended output is visually equivalent to the single-pass result but not
    bit-identical: window partitioning and edge padding differ at tile seams.

    The processor pads up to the next window multiple, so each reconstruction runs a
    few pixels past input*scale; that bottom/right padding is cropped before use.
    """
    import torch

    h, w = image.shape[:2]

    def _forward(patch: np.ndarray) -> np.ndarray:
        inputs = processor(np.ascontiguousarray(patch), return_tensors="pt").to(device)
        with torch.no_grad():
            recon = model(**inputs).reconstruction
        out = to_array(recon)
        del recon, inputs
        if device == "cuda":
            torch.cuda.empty_cache()  # release this patch's VRAM before the next
        ph, pw = patch.shape[:2]
        return out[: ph * scale, : pw * scale]  # drop the window-multiple padding

    if tile <= 0 or (h <= tile and w <= tile):
        return _forward(image)

    if not 0 <= overlap < tile:
        raise SystemExit(f"overlap must be in [0, tile); got overlap={overlap}, tile={tile}")

    gray = image.ndim == 2
    channels = 1 if gray else image.shape[2]
    acc = np.zeros((h * scale, w * scale, channels), dtype=np.float32)
    wsum = np.zeros((h * scale, w * scale), dtype=np.float32)
    ramp = overlap * scale
    for y in _tile_starts(h, tile, overlap):
        for x in _tile_starts(w, tile, overlap):
            th = min(tile, h - y)
            tw = min(tile, w - x)
            patch_out = _forward(image[y : y + th, x : x + tw])
            if patch_out.ndim == 2:
                patch_out = patch_out[:, :, None]
            ph, pw = patch_out.shape[:2]
            weight = np.outer(_blend_ramp(ph, ramp), _blend_ramp(pw, ramp))
            oy, ox = y * scale, x * scale
            acc[oy : oy + ph, ox : ox + pw] += patch_out.astype(np.float32) * weight[:, :, None]
            wsum[oy : oy + ph, ox : ox + pw] += weight
    blended = (acc / wsum[:, :, None]).round().clip(0, 255).astype(np.uint8)
    return blended[:, :, 0] if gray else blended


# ==========================================================================
# 1_swin2sr.py
# ==========================================================================

# Per kind: the (width, height) we try to reach. A side already at/above its
# target needs no upscaling.
THRESHOLDS = {"car": (1080, 1080), "plate": (1054, 256)}


def scale_for(width: int, height: int, kind: str) -> int:
    """Smallest available upscale (2 or 4) that lifts both sides to the kind's
    target resolution; 1 if the image is already big enough. Capped at 4 (the
    largest checkpoint) even when 4x still falls short.

    Equivalent to: upscale if either side is below target, choosing 2x when
    doubling clears every deficient side and 4x otherwise.
    """
    if width <= 0 or height <= 0:
        raise SystemExit(f"width and height must be positive, got {width}x{height}")
    try:
        target_w, target_h = THRESHOLDS[kind]
    except KeyError:
        raise SystemExit(f"kind must be one of {sorted(THRESHOLDS)}, got {kind!r}") from None
    need = max(target_w / width, target_h / height)
    if need <= 1.0:
        return 1
    return 2 if need <= 2.0 else 4


class SuperResolver:
    """Loads the local Swin2SR x2/x4 checkpoints on demand and applies them. Each
    scale's model is loaded once, on first use, then reused — a 2x-only or
    plate-only workload never loads the other checkpoint.
    """

    def __init__(self, tile: int = 512, overlap: int = 16):
        self.device = pick_device()
        self.tile = tile  # 0 = whole-image forward; >0 = tile x tile patches
        self.overlap = overlap
        self._cache: dict[int, tuple] = {}

    def _model(self, scale: int):
        if scale not in self._cache:
            checkpoint, config = SWIN2SR_SCALES[scale]
            self._cache[scale] = load_swin2sr_local(MODELS_DIR / checkpoint, config, self.device)
        return self._cache[scale]

    def upscale(self, image: np.ndarray, scale: int) -> np.ndarray:
        """Upscale RGB *image* by an explicit *scale* of 2 or 4 with the matching
        model. Output is exactly (w*scale, h*scale). With --tile > 0 the forward is
        run over overlapping patches to cap VRAM (see swin2sr_forward)."""
        model, processor = self._model(scale)
        return swin2sr_forward(
            model, processor, image, scale, self.device, tile=self.tile, overlap=self.overlap
        )

    def enhance(self, image: np.ndarray, kind: str) -> np.ndarray:
        """Enhance RGB *image* to *kind*'s target resolution (car or plate),
        unchanged when already big enough. Re-measures after every pass and repeats
        while a side is still short — one 4x pass cannot lift a very small crop to
        target. Terminates: every pass lifts by >=2x, so the shortfall halves each
        time. Binds the scale decision to this image so the two can't drift apart —
        the primary entry point."""
        while (scale := scale_for(image.shape[1], image.shape[0], kind)) != 1:
            image = self.upscale(image, scale)
        return image


def run(
    input_folder: str,
    output_folder: str,
    type: str,  # noqa: A002 - CLI flag is --type
    tile: int = 512,
    overlap: int = 16,
) -> None:
    """Auto-scale every image in <input_folder> into <output_folder> for --type (car/plate).

    --tile 0 (default) runs each forward on the whole image; --tile N caps peak VRAM
    by upscaling N x N patches with --overlap px halos and feather-blending them,
    preventing OOM on large intermediates at the cost of a not-bit-exact seam blend.
    """
    kind = type  # image type, applied to the whole input folder
    if kind not in THRESHOLDS:
        raise SystemExit(f"--type must be one of {sorted(THRESHOLDS)}, got {kind!r}")

    input_dir, output_dir = resolve_io(input_folder, output_folder)

    # resolver is created on first real upscale, so an all-copy run loads nothing.
    resolver: SuperResolver | None = None
    decoded: np.ndarray | None = None  # process_folder calls copy_reason then process

    def copy_reason(src: Path) -> str | None:
        nonlocal decoded
        decoded = read_image(src)  # reused by process() so we decode each file once
        h, w = decoded.shape[:2]
        # already at/above target: keep coverage under the original name
        return f"{w}x{h} >= target" if scale_for(w, h, kind) == 1 else None

    def process(src: Path, dst: Path) -> str:
        nonlocal resolver
        if resolver is None:
            resolver = SuperResolver(tile=tile, overlap=overlap)
            tiling = f"tile {tile} overlap {overlap}" if tile > 0 else "no tiling"
            print(f"device: {resolver.device} | type: {kind} | {tiling}")
        image = decoded
        h, w = image.shape[:2]
        out = resolver.enhance(image, kind)
        write_image(dst, out)
        return f"{w}x{h} -> {out.shape[1]}x{out.shape[0]} {dst.name}"

    process_folder(
        input_dir,
        output_dir,
        output_name=lambda src: f"{src.stem}_p_1_swin2sr.png",
        process=process,
        copy_reason=copy_reason,
        check_collisions=False,
    )


if __name__ == "__main__":
    fire.Fire(run)
