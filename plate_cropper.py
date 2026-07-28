import os
import cv2
import numpy as np


def order_plate_points(pts):
    """
    Order 4 points as:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def crop_and_save_lpd(image_path, plate_data, output_name="lpd.jpg", lpd_folder_output_name=None):
    """
    Crop the detected plate from image using the plate coordinates
    and save it in the same folder as the source image.

    Args:
        image_path (str): Full path to the image
        plate_data (dict): Plate object from API response
        output_name (str): Output file name beside the source image
        lpd_folder_output_name (str | None): Optional output file name inside lpd folder

    Returns:
        str | None: Output path if success, otherwise None
    """
    try:
        if not image_path or not os.path.isfile(image_path):
            print(f"[LPD] Image not found: {image_path}")
            return None

        required_keys = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        for key in required_keys:
            if key not in plate_data:
                print(f"[LPD] Missing key in plate data: {key}")
                return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"[LPD] Failed to read image: {image_path}")
            return None

        pts = [
            [plate_data["x1"], plate_data["y1"]],
            [plate_data["x2"], plate_data["y2"]],
            [plate_data["x3"], plate_data["y3"]],
            [plate_data["x4"], plate_data["y4"]],
        ]

        pts = order_plate_points(pts)

        width_top = np.linalg.norm(pts[1] - pts[0])
        width_bottom = np.linalg.norm(pts[2] - pts[3])
        max_width = int(max(width_top, width_bottom))

        height_right = np.linalg.norm(pts[2] - pts[1])
        height_left = np.linalg.norm(pts[3] - pts[0])
        max_height = int(max(height_right, height_left))

        if max_width <= 0 or max_height <= 0:
            print("[LPD] Invalid plate dimensions.")
            return None

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, output_name)

        ok = cv2.imwrite(output_path, warped)
        if not ok:
            print(f"[LPD] Failed to save image: {output_path}")
            return None

        print(f"[LPD] Saved cropped plate: {output_path}")

        if lpd_folder_output_name:
            lpd_dir = os.path.join(output_dir, "lpd")
            os.makedirs(lpd_dir, exist_ok=True)
            lpd_folder_path = os.path.join(lpd_dir, lpd_folder_output_name)
            if cv2.imwrite(lpd_folder_path, warped):
                print(f"[LPD] Saved cropped plate in lpd folder: {lpd_folder_path}")
            else:
                print(f"[LPD] Failed to save image in lpd folder: {lpd_folder_path}")

        return output_path

    except Exception as e:
        print(f"[LPD] Error while cropping plate: {e}")
        return None


def crop_and_save_lpd_from_response(image_path, response_data, output_name="lpd.jpg", lpd_folder_output_name=None):
    """
    Extract first plate from full API response and save cropped plate.

    Args:
        image_path (str): Full path to the image
        response_data (dict | str): Full API response
        output_name (str): Output file name beside the source image
        lpd_folder_output_name (str | None): Optional output file name inside lpd folder

    Returns:
        str | None: Output path if success, otherwise None
    """
    try:
        if isinstance(response_data, str):
            import json
            response_data = json.loads(response_data)

        plates = response_data.get("plate_", [])
        if not plates:
            print("[LPD] No plate_ found in response.")
            return None

        first_plate = plates[0]
        return crop_and_save_lpd(
            image_path,
            first_plate,
            output_name=output_name,
            lpd_folder_output_name=lpd_folder_output_name,
        )

    except Exception as e:
        print(f"[LPD] Error while extracting plate from response: {e}")
        return None
