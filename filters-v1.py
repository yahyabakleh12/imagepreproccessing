#!/usr/bin/env python3
"""
Remove smooth shadow from all images in a folder or one image, reduce blur, enhance tones, and save results.

Usage examples:
	python filters.py
	python filters.py images
	python filters.py --image images/test.jpg
	python filters.py --input images --output out-images
	python filters.py --image images/test.jpg --output out-images
	python filters.py --input images --output out-images --deshadow-kernel 51
	python filters.py images --no-enhance
	python filters.py images --deblur-strength 1.2
	python filters.py images --black-threshold 80 --white-threshold 220
"""
import argparse
import glob
import os
from typing import List

import cv2
import numpy as np

# Edit these defaults to change input/output folders quickly
DEFAULT_INPUT = "images"
DEFAULT_OUTPUT = "out-images"
DEFAULT_OUTPUT_EXT = "png"


def parse_args():
	p = argparse.ArgumentParser(description="Remove smooth shadow, reduce blur, and enhance image tones in a folder or one image")
	p.add_argument("input_folder", nargs="?", help=f"Input images folder (default: {DEFAULT_INPUT})")
	p.add_argument("--input", "-i", default=None, help=f"Input images folder (default: {DEFAULT_INPUT})")
	p.add_argument("--image", "-m", default=None, help="Single image to process instead of a folder")
	p.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"Output folder to save results (default: {DEFAULT_OUTPUT})")
	p.add_argument("--output-ext", default=DEFAULT_OUTPUT_EXT,
		   help=f"Output image extension. PNG preserves gray tones well (default: {DEFAULT_OUTPUT_EXT})")
	p.add_argument("--deshadow-kernel", type=int, default=51,
		   help="Kernel size for shadow background estimate (odd integer, default 51)")
	p.add_argument("--no-enhance", action="store_true", help="Skip quality enhancement after deshadowing")
	p.add_argument("--clahe-clip", type=float, default=2.0,
		   help="Local contrast strength for enhancement (default: 2.0)")
	p.add_argument("--denoise-strength", type=float, default=7.0,
		   help="Noise reduction strength for enhancement (default: 7.0)")
	p.add_argument("--no-deblur", action="store_true", help="Skip blur-removal filter during enhancement")
	p.add_argument("--deblur-strength", type=float, default=0.8,
		   help="Blur-removal strength for crisper edges (default: 0.8)")
	p.add_argument("--deblur-radius", type=float, default=1.0,
		   help="Blur radius estimate for blur-removal filter (default: 1.0)")
	p.add_argument("--sharpen-amount", type=float, default=0.0,
		   help="Extra final sharpening strength after deblurring (default: 0.0)")
	p.add_argument("--no-tone-cleanup", action="store_true",
		   help="Skip tone cleanup that darkens digits and whitens the background")
	p.add_argument("--black-threshold", type=int, default=80,
		   help="Pixels at or below this gray value become black (default: 80)")
	p.add_argument("--white-threshold", type=int, default=220,
		   help="Pixels at or above this gray value become white (default: 220)")
	p.add_argument("--ext", "-e", default="png,jpg,jpeg,tif,tiff,bmp",
		   help="Comma-separated image extensions to process (default: common formats)")
	args = p.parse_args()
	if args.image:
		if args.input is not None or args.input_folder:
			p.error("Use either --image or an input folder, not both.")
	elif args.input is None:
		args.input = args.input_folder or DEFAULT_INPUT
	elif args.input_folder:
		p.error("Use either positional input_folder or --input, not both.")
	return args


def gather_files(input_folder: str, exts: List[str]) -> List[str]:
	files = []
	for ext in exts:
		pattern = os.path.join(input_folder, f"*.{ext}")
		files.extend(sorted(glob.glob(pattern)))
	return files


def make_output_path(output_folder: str, input_path: str, output_ext: str) -> str:
	base_name = os.path.splitext(os.path.basename(input_path))[0]
	ext = output_ext.strip().lstrip(".") or DEFAULT_OUTPUT_EXT
	return os.path.join(output_folder, f"{base_name}.{ext}")


def deshadow(img: np.ndarray, deshadow_kernel: int = 51) -> np.ndarray:
	k = int(deshadow_kernel)
	if k <= 1:
		k = 51
	if k % 2 == 0:
		k += 1

	bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
	background = cv2.morphologyEx(img, cv2.MORPH_CLOSE, bg_kernel)
	background = background.astype("float32")
	background[background == 0] = 1.0

	result = cv2.divide(img.astype("float32"), background, scale=255.0)
	return np.clip(result, 0, 255).astype("uint8")


def enhance_quality(
	img: np.ndarray,
	clahe_clip: float = 2.0,
	denoise_strength: float = 7.0,
	deblur_strength: float = 0.8,
	deblur_radius: float = 1.0,
	apply_deblur: bool = True,
	sharpen_amount: float = 0.0,
	apply_tone_cleanup: bool = True,
	black_threshold: int = 80,
	white_threshold: int = 220,
) -> np.ndarray:
	clip = max(float(clahe_clip), 0.1)
	denoise = max(float(denoise_strength), 0.0)
	sharpen = max(float(sharpen_amount), 0.0)

	clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
	enhanced = clahe.apply(img)

	if denoise > 0:
		enhanced = cv2.fastNlMeansDenoising(enhanced, None, denoise, 7, 21)

	if apply_deblur:
		enhanced = remove_blur(enhanced, deblur_strength, deblur_radius)

	if sharpen > 0:
		blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
		enhanced = cv2.addWeighted(enhanced, 1.0 + sharpen, blurred, -sharpen, 0)

	if apply_tone_cleanup:
		enhanced = enhance_tones(enhanced, black_threshold, white_threshold)

	return np.clip(enhanced, 0, 255).astype("uint8")


def enhance_tones(
	img: np.ndarray,
	black_threshold: int = 80,
	white_threshold: int = 220,
) -> np.ndarray:
	black = int(np.clip(black_threshold, 0, 254))
	white = int(np.clip(white_threshold, 1, 255))
	if white <= black:
		white = min(black + 1, 255)

	result = (img.astype("float32") - black) * (255.0 / (white - black))
	return np.clip(result, 0, 255).astype("uint8")


def remove_blur(
	img: np.ndarray,
	deblur_strength: float = 0.8,
	deblur_radius: float = 1.0,
) -> np.ndarray:
	strength = max(float(deblur_strength), 0.0)
	radius = max(float(deblur_radius), 0.1)
	if strength == 0:
		return img

	base = img.astype("float32")
	blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=radius, sigmaY=radius)
	deblurred = base + strength * (base - blurred)
	return np.clip(deblurred, 0, 255).astype("uint8")


def process_image(in_path: str, out_path: str, args) -> bool:
	img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
	if img is None:
		print(f"Warning: failed to read {in_path}")
		return False

	img = deshadow(img, args.deshadow_kernel)
	if not args.no_enhance:
		img = enhance_quality(
			img,
			args.clahe_clip,
			args.denoise_strength,
			args.deblur_strength,
			args.deblur_radius,
			not args.no_deblur,
			args.sharpen_amount,
			not args.no_tone_cleanup,
			args.black_threshold,
			args.white_threshold,
		)

	ok = cv2.imwrite(out_path, img)
	if not ok:
		print(f"Warning: failed to write {out_path}")
	return ok


def main():
	args = parse_args()
	exts = [e.strip().lstrip(".") for e in args.ext.split(",") if e.strip()]

	if args.image:
		if not os.path.isfile(args.image):
			print(f"Input image does not exist: {args.image}")
			return

		os.makedirs(args.output, exist_ok=True)
		dst = make_output_path(args.output, args.image, args.output_ext)
		ok = process_image(args.image, dst, args)
		if ok:
			print(f"Saved: {dst}")
		return

	if not os.path.isdir(args.input):
		print(f"Input folder does not exist: {args.input}")
		return

	os.makedirs(args.output, exist_ok=True)

	files = gather_files(args.input, exts)
	if not files:
		print("No images found with given extensions in the input folder.")
		return

	for src in files:
		dst = make_output_path(args.output, src, args.output_ext)
		ok = process_image(src, dst, args)
		if ok:
			print(f"Saved: {dst}")


if __name__ == "__main__":
	main()

