"""Unified plate preprocessing: measure once, then denoise, deblur, contrast, sharpen.

One grayscale pass replacing the separate denoise -> deblur -> preprocess chain.
Each image is measured once (Immerkaer sigma, variance-of-Laplacian sharpness) and
those two numbers both gate the steps and set their strength: denoise only when the
image is noisy, at the strength its own sigma implies (BM3D; the nlmeans path is
fixed-strength and only its gate is measured); deconvolve only when it is blurry;
unsharp only when it is *still* soft after CLAHE. The chain applies its denoisers
and sharpeners unconditionally and twice over (BM3D then NlMeans, deconv then
unsharp), which smooths character strokes and re-amplifies what was just removed;
running each at most once, in noise -> blur -> contrast -> sharpen order, is the
whole point of this script. Work stays in float through the denoise and the
deconvolution and is quantised once, for CLAHE, so the chain's three uint8
round-trips become one.

Grayscale (H, W) uint8 output, the same size as the input. Resolution is not this
stage's job: reach the target with 1_swin2sr.py or 4_resize.py first.

Run:
    uv run python test/2_preprocess.py <input_folder> <output_folder>
    uv run python test/2_preprocess.py <input_folder> <output_folder> --denoiser nlmeans
    uv run python test/2_preprocess.py <input_folder> <output_folder> --clahe-clip 3.0
"""

import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import bm3d as bm3d_lib
import cv2
import fire
import numpy as np
from skimage.restoration import richardson_lucy

# ==========================================================================
# Inlined from helpers/, deblurring/ and denoising/ so this file stands alone.
# ==========================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def resolve_dir(arg: str) -> Path:
    """Resolve a folder argument against the current directory (absolute paths kept)."""
    return Path(arg).expanduser().resolve()


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


def to_float(arr: np.ndarray) -> np.ndarray:
    """uint8 image -> float64 array in [0, 1]."""
    return arr.astype(np.float64) / 255.0


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Float image in [0, 1] -> uint8 (clipped, rounded)."""
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


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


def laplacian_sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian divided by the image's own variance; higher = sharper.

    Normalising by the image variance cancels contrast and resolution, so one
    threshold transfers between datasets — unlike the raw variance, whose scale
    moves with the content.
    """
    f = image.astype(np.float64)
    return float(cv2.Laplacian(f, cv2.CV_64F).var() / (f.var() + 1e-9))


# ===========================================================================
# Deblur — PSF construction
# ===========================================================================


def _odd_positive(value: int, name: str) -> None:
    if value <= 0 or value % 2 == 0:
        raise SystemExit(f"{name} must be a positive odd integer, got {value}")


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Normalized 2D Gaussian PSF for soft, out-of-focus blur."""
    _odd_positive(size, "--kernel-size")
    if sigma <= 0.0:
        raise SystemExit(f"--kernel-sigma must be > 0, got {sigma}")
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def motion_kernel(size: int, angle_degrees: float) -> np.ndarray:
    """Normalized linear-motion PSF: a length-``size`` streak at ``angle``."""
    _odd_positive(size, "--kernel-size")
    kernel = np.zeros((size, size), dtype=np.float64)
    center = (size - 1) / 2.0
    radius = size // 2
    theta = math.radians(angle_degrees)
    dx, dy = math.cos(theta), math.sin(theta)
    # Sample the line densely and splat each sample bilinearly onto the grid.
    for t in np.linspace(-radius, radius, max(size * 8, 64)):
        x, y = center + dx * float(t), center + dy * float(t)
        x0, y0 = math.floor(x), math.floor(y)
        for yy in (y0, y0 + 1):
            if not 0 <= yy < size:
                continue
            wy = 1.0 - abs(y - yy)
            if wy <= 0.0:
                continue
            for xx in (x0, x0 + 1):
                if 0 <= xx < size:
                    wx = 1.0 - abs(x - xx)
                    if wx > 0.0:
                        kernel[yy, xx] += wx * wy
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def build_psf(
    kernel: str, kernel_size: int, kernel_sigma: float, motion_angle: float
) -> np.ndarray:
    if kernel == "gaussian":
        return gaussian_kernel(kernel_size, kernel_sigma)
    if kernel == "motion":
        return motion_kernel(kernel_size, motion_angle)
    raise SystemExit(f"--kernel must be one of [gaussian, motion], got {kernel!r}")


# Laplacian-of-differences mask: convolving with it cancels image content up to
# first order, so the response is dominated by noise (Immerkaer 1996).
_IMMERKAER_MASK = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])


def sigma_immerkaer(img: np.ndarray) -> float:
    """Fast noise sigma via a Laplacian mask that cancels structure up to first
    order (Immerkaer 1996). Grayscale uint8 in, sigma in gray levels out."""
    from scipy.signal import convolve2d

    f = img.astype(np.float64)
    h, w = f.shape
    if h < 3 or w < 3:
        return 0.0
    response = np.abs(convolve2d(f, _IMMERKAER_MASK, mode="valid")).sum()
    return float(np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * response)


def apply_channels(img: np.ndarray, fn) -> np.ndarray:
    """Run a 2D-only operation on each channel independently, restacking to H,W,C."""
    if img.ndim == 2:
        return fn(img)
    return np.stack([fn(img[..., c]) for c in range(img.shape[-1])], axis=-1)


def reflect_pad(img: np.ndarray, edge_pad: int) -> np.ndarray:
    """Reflect-pad H/W by ``edge_pad`` (0 disables it).

    Deconvolution assumes a circular (wrap-around) blur; reflect-padding the
    border first keeps that wrap from ringing the real image edges. Undo it with
    ``crop_pad(out, edge_pad)``.
    """
    if edge_pad <= 0:
        return img
    pad_width = ((edge_pad, edge_pad), (edge_pad, edge_pad)) + (((0, 0),) if img.ndim == 3 else ())
    return np.pad(img, pad_width, mode="reflect")


def crop_pad(arr: np.ndarray, pad: int) -> np.ndarray:
    """Undo ``reflect_pad`` by cropping ``pad`` pixels off each side."""
    return arr[pad:-pad, pad:-pad] if pad else arr


def rl_deblur(
    img: np.ndarray, psf: np.ndarray, *, num_iter: int = 30, edge_pad: int = 16
) -> np.ndarray:
    """Richardson-Lucy deconvolve a float image in [0, 1] with the assumed ``psf``.

    Runs on a reflect-padded copy (cropped back afterwards) so the circular FFT
    boundary does not ring the real edges.
    """
    work = reflect_pad(img, edge_pad)
    out = apply_channels(work, lambda c: richardson_lucy(c, psf, num_iter))
    return crop_pad(np.clip(out, 0.0, 1.0), edge_pad)


def bm3d_denoise(arr: np.ndarray, sigma: float, gray: bool) -> np.ndarray:
    """BM3D-denoise a float image in [0, 1]; grayscale (H,W) or colour (H,W,3).

    A colour image with identical channels takes the grayscale path: bm3d_rgb
    divides by zero-variance chroma and returns NaN for the whole array.
    """
    if not gray and arr.ndim == 3 and (arr == arr[:, :, :1]).all():
        return np.clip(
            np.repeat(bm3d_lib.bm3d(arr[:, :, 0], sigma)[:, :, None], 3, axis=2), 0.0, 1.0
        )
    out = bm3d_lib.bm3d(arr, sigma) if gray else bm3d_lib.bm3d_rgb(arr, sigma)
    return np.clip(out, 0.0, 1.0)


# ==========================================================================
# 2_preprocess.py
# ==========================================================================

DENOISERS = ("bm3d", "nlmeans")

# Gates, both fitted on post-SR plates and carried over unchanged from the two
# stages this script replaces. NOISE_THRESHOLD is an Immerkaer sigma
# gray levels; the estimator under-reads, so SIGMA_SCALE corrects it before BM3D.
# BLUR_THRESHOLD is variance-of-Laplacian normalised by image variance.
NOISE_THRESHOLD = 0.195
SIGMA_SCALE = 2.0
BLUR_THRESHOLD = 0.0098

NLMEANS_H = 5  # only when --denoiser nlmeans; low so character strokes survive
KERNEL_SIZE = 7
KERNEL_SIGMA = 0.8
NUM_ITER = 15
CLAHE_CLIP = 2.5
CLAHE_TILE = (8, 8)
UNSHARP_SIGMA = 1.2
UNSHARP_AMOUNT = 0.8  # sharp = (1 + amount) * img - amount * blur


def denoise(gray_float: np.ndarray, sigma: float, denoiser: str, sigma_scale: float) -> np.ndarray:
    """Denoise a float image; BM3D is matched to the strength *sigma* implies.

    The nlmeans path runs at a fixed h (merna's tuned value, chosen to keep
    character strokes) and ignores *sigma* - there only the gate is measured.
    Falls back to the input when BM3D returns something unusable (non-finite, or a
    black frame from a non-black input) - both were observed on real plates.
    """
    if denoiser == "nlmeans":
        return to_float(cv2.fastNlMeansDenoising(to_uint8(gray_float), None, h=NLMEANS_H))
    out = bm3d_denoise(gray_float, sigma * sigma_scale / 255.0, True)
    if not np.isfinite(out).all() or (np.any(gray_float) and not np.any(to_uint8(out))):
        return gray_float
    return out


def enhance(
    image: np.ndarray,
    *,
    denoiser: str = "bm3d",
    noise_threshold: float = NOISE_THRESHOLD,
    sigma_scale: float = SIGMA_SCALE,
    blur_threshold: float = BLUR_THRESHOLD,
    psf: np.ndarray | None = None,
    num_iter: int = NUM_ITER,
    clahe_clip: float = CLAHE_CLIP,
) -> tuple[np.ndarray, str]:
    """Preprocess one plate image; return grayscale uint8 (H, W) and a status string.

    Accepts RGB (H, W, 3) or grayscale (H, W). Each measurement gates the step that
    follows it and is taken on the image that step will actually see: sharpness
    after denoising, since noise inflates variance-of-Laplacian and a noisy blurry
    plate would otherwise score as sharp and skip the deconvolution; and again
    after CLAHE, since local contrast changes what "still soft" means.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    sigma = sigma_immerkaer(gray)
    notes = []

    work = to_float(gray)
    denoised = bool(np.isfinite(sigma) and sigma > noise_threshold)
    if denoised:
        work = denoise(work, sigma, denoiser, sigma_scale)
        notes.append(f"denoised sigma={sigma:.3f}")

    score = laplacian_sharpness(to_uint8(work) if denoised else gray)
    if np.isfinite(score) and score < blur_threshold:
        if psf is None:
            psf = build_psf("gaussian", KERNEL_SIZE, KERNEL_SIGMA, 0.0)
        work = rl_deblur(work, psf, num_iter=num_iter)
        notes.append(f"deblurred sharpness={score:.4f}")

    out = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=CLAHE_TILE).apply(to_uint8(work))
    final = laplacian_sharpness(out)
    if np.isfinite(final) and final < blur_threshold:
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=UNSHARP_SIGMA)
        out = cv2.addWeighted(out, 1.0 + UNSHARP_AMOUNT, blurred, -UNSHARP_AMOUNT, 0)
        notes.append(f"unsharped sharpness={final:.4f}")
    return out, ", ".join(notes) if notes else "clean, contrast only"


def run(
    input_folder: str,
    output_folder: str,
    denoiser: str = "bm3d",
    noise_threshold: float = NOISE_THRESHOLD,
    sigma_scale: float = SIGMA_SCALE,
    blur_threshold: float = BLUR_THRESHOLD,
    num_iter: int = NUM_ITER,
    kernel_size: int = KERNEL_SIZE,
    kernel_sigma: float = KERNEL_SIGMA,
    clahe_clip: float = CLAHE_CLIP,
) -> None:
    """Preprocess every plate image in <input_folder> into <output_folder> (grayscale)."""
    if denoiser not in DENOISERS:
        raise SystemExit(f"--denoiser must be one of {list(DENOISERS)}, got {denoiser!r}")
    psf = build_psf("gaussian", int(kernel_size), float(kernel_sigma), 0.0)

    input_dir, output_dir = resolve_io(input_folder, output_folder)

    print(f"preprocess | denoiser={denoiser} | noise>{noise_threshold:g} | blur<{blur_threshold:g}")

    def process(src: Path, dst: Path) -> str:
        out, notes = enhance(
            read_image(src),
            denoiser=denoiser,
            noise_threshold=noise_threshold,
            sigma_scale=sigma_scale,
            blur_threshold=blur_threshold,
            psf=psf,
            num_iter=num_iter,
            clahe_clip=clahe_clip,
        )
        write_image(dst, out)
        h, w = out.shape[:2]
        return f"{w}x{h} | {notes}"

    process_folder(
        input_dir,
        output_dir,
        output_name=lambda src: f"{src.stem}_p_2_preprocess.png",
        process=process,
        skip_existing=False,
        check_collisions=False,
    )


if __name__ == "__main__":
    fire.Fire(run)
