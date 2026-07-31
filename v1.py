"""v1 pipeline, standalone copy: compose the self-contained stages in test/.

Mirrors pipelines/v1.py but depends on nothing outside this folder: it loads the
sibling test/1_swin2sr.py and test/2_preprocess.py by path (their names start
with a digit, so a normal import statement cannot name them) and reuses the
helpers those files already carry inlined.

--order is a string of stage digits applied left to right: 1 = Swin2SR
super-resolution, 2 = grayscale plate preprocessing (gated denoise,
deconvolution, CLAHE, unsharp). So --order 12 upscales then preprocesses,
--order 21 reverses them, and --order 2 preprocesses alone. Output name carries
the order, so different orders written to one folder do not collide. The resize
stage (digit 4 in pipelines/v1.py) is not cloned here.

The Swin2SR stage caps peak VRAM via --tile (px patch size, default 512; 0 =
whole-image forward) and --overlap (px halo between patches, default 16).

Run:
    uv run python test/v1.py <input_folder> <output_folder>
    uv run python test/v1.py <input_folder> <output_folder> --order 21
    uv run python test/v1.py <input_folder> <output_folder> --order 12 --type car
"""

from collections.abc import Callable
from importlib import util
from pathlib import Path

import cv2
import fire
import numpy as np


def _load_stage(filename: str):
    """Import a sibling stage file whose name starts with a digit.

    `1_swin2sr` is not a valid identifier, so it cannot be imported by statement;
    load it from its path instead and give the module a legal name.
    """
    path = Path(__file__).resolve().parent / f"{filename}.py"
    spec = util.spec_from_file_location(f"stage_{filename}", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sr = _load_stage("1_swin2sr")
preprocess = _load_stage("2_preprocess")


STAGE_LABELS = {
    "1": "Swin2SR Super Resolution",
    "2": "Adaptive Preprocessing",
}


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Promote a grayscale image to 3-channel RGB (Swin2SR expects 3 channels; a
    preprocess stage earlier in the order emits grayscale)."""
    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def build_stages(
    order: str, kind: str, tile: int, overlap: int
) -> list[Callable[[np.ndarray], np.ndarray]]:
    """Map each digit in *order* to an in-memory image -> image stage.

    Every stage takes and returns an in-memory image, so the composition never
    touches disk between steps. Raises on an unknown digit.
    """
    resolver = sr.SuperResolver(tile=tile, overlap=overlap) if "1" in order else None
    stages: list[Callable[[np.ndarray], np.ndarray]] = []
    for digit in order:
        if digit == "1":
            stages.append(lambda img: resolver.enhance(_ensure_rgb(img), kind))
        elif digit == "2":
            stages.append(lambda img: preprocess.enhance(img)[0])  # (image, notes)
        else:
            raise SystemExit(f"--order digits must be one of 1, 2, got {digit!r}")
    return stages


class EnhancementPipelineV1:
    """Reusable v1 enhancement pipeline with lazily loaded Swin2SR models."""

    def __init__(
        self,
        order: str = "12",
        kind: str = "plate",
        tile: int = 512,
        overlap: int = 16,
    ) -> None:
        if kind not in sr.THRESHOLDS:
            raise ValueError(f"kind must be one of {sorted(sr.THRESHOLDS)}, got {kind!r}")
        self.order = str(order)
        if not self.order:
            raise ValueError("order must contain at least one stage digit")
        self.stages = build_stages(self.order, kind, tile, overlap)

    def apply_with_stages(self, image: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Apply v1 and return a named image after every processing stage."""
        if image is None or image.size == 0:
            raise ValueError("Plate image is empty.")

        # The v1 stages use RGB internally; OpenCV detector crops are BGR.
        result = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.ndim == 3
            else image.copy()
        )
        outputs: list[tuple[str, np.ndarray]] = []
        for digit, stage in zip(self.order, self.stages):
            result = stage(result)
            outputs.append((STAGE_LABELS[digit], result.copy()))
        return outputs

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply v1 to an OpenCV BGR/grayscale image and return its result."""
        return self.apply_with_stages(image)[-1][1]


def run(
    input_folder: str,
    output_folder: str,
    order: str = "12",
    tile: int = 512,
    overlap: int = 16,
    type: str = "plate",  # noqa: A002 - CLI flag is --type
) -> None:
    """Enhance every image in <input_folder> into <output_folder> along --order."""
    kind = type  # image type, applied to the whole input folder
    if kind not in sr.THRESHOLDS:
        raise SystemExit(f"--type must be one of {sorted(sr.THRESHOLDS)}, got {kind!r}")

    order = str(order)  # fire parses a bare 12 as an int
    if not order:
        raise SystemExit("--order must contain at least one stage digit")

    input_dir, output_dir = sr.resolve_io(input_folder, output_folder)
    stages = build_stages(order, kind, tile, overlap)

    def process(src: Path, dst: Path) -> str:
        image = sr.read_image(src)
        for stage in stages:
            image = stage(image)
        sr.write_image(dst, image)
        h, w = image.shape[:2]
        return f"{w}x{h} -> {dst.name}"

    sr.process_folder(
        input_dir,
        output_dir,
        output_name=lambda src: f"{src.stem}_p_v1_{order}.png",
        process=process,
    )


if __name__ == "__main__":
    fire.Fire(run)
