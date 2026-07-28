from pathlib import Path

import cv2
import numpy as np


def enhance_plate(
    input_path: str,
    output_directory: str = "enhanced_plate",
) -> None:
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Convert to grayscale
    # Perspective correction should ideally be performed before
    # this step using the four plate corner points.
    # ---------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # 2. Upscale 4x
    # ---------------------------------------------------------
    upscaled = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_LANCZOS4,
    )

    # ---------------------------------------------------------
    # 3. Light denoising
    # Keep h low so character strokes are not removed.
    # ---------------------------------------------------------
    denoised = cv2.fastNlMeansDenoising(
        upscaled,
        None,
        h=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # ---------------------------------------------------------
    # 4. Local contrast enhancement
    # ---------------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8),
    )
    contrasted = clahe.apply(denoised)

    # ---------------------------------------------------------
    # 5. Mild unsharp mask
    # sharp = original + amount * (original - blurred)
    # ---------------------------------------------------------
    gaussian = cv2.GaussianBlur(
        contrasted,
        ksize=(0, 0),
        sigmaX=1.2,
    )

    sharpened = cv2.addWeighted(
        contrasted,
        1.8,       # 1 + sharpening amount
        gaussian,
        -0.8,      # negative blurred component
        0,
    )

    # ---------------------------------------------------------
    # Save each stage through the sharpened result.
    # ---------------------------------------------------------
    outputs = {
        "01_upscaled.png": upscaled,
        "02_denoised.png": denoised,
        "03_clahe.png": contrasted,
        "04_sharpened.png": sharpened,
    }

    for filename, result in outputs.items():
        destination = output_path / filename

        if not cv2.imwrite(str(destination), result):
            raise RuntimeError(f"Could not save: {destination}")

        print(f"Saved: {destination}")


if __name__ == "__main__":
    enhance_plate(
        input_path="33.jpg",
        output_directory="enhanced_plate",
    )
