"""Visual attributes extraction, color analysis, and appearance Re-Identification.

Analyzes cropped detections to identify upper/lower clothing colors, vehicle subtypes
(sedan, SUV, hatchback), associated accessories (backpack, suitcase, trolley bag),
and appearance vectors for identity re-identification across occlusions.
"""

import cv2
import numpy as np


def compute_appearance_vector(crop_bgr: np.ndarray) -> np.ndarray:
    """Computes a multi-zone normalized HSV color appearance signature for ReID."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(96, dtype=np.float32)

    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return np.zeros(96, dtype=np.float32)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mid_y = max(1, h // 2)

    upper = hsv[:mid_y, :]
    lower = hsv[mid_y:, :]

    # 12 Hue bins, 4 Saturation bins = 48 bins per zone -> 96 total
    hist_up = cv2.calcHist([upper], [0, 1], None, [12, 4], [0, 180, 0, 256])
    hist_lo = cv2.calcHist([lower], [0, 1], None, [12, 4], [0, 180, 0, 256])

    cv2.normalize(hist_up, hist_up)
    cv2.normalize(hist_lo, hist_lo)

    vec = np.concatenate([hist_up.flatten(), hist_lo.flatten()]).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def compare_appearance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two appearance vectors (0.0 to 1.0)."""
    if v1 is None or v2 is None or len(v1) == 0 or len(v2) == 0:
        return 0.0
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return max(0.0, min(1.0, dot / (norm + 1e-6)))


def detect_dominant_color(bgr_region: np.ndarray) -> str:
    """Maps pixel values in an image region to standard human-named colors with chromatic priority."""
    if bgr_region is None or bgr_region.size == 0:
        return "unknown"

    h, w = bgr_region.shape[:2]
    if h < 2 or w < 2:
        return "unknown"

    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    n_pixels = float(h * w)

    # 1. Chromatic colors (colored shirts/clothing: yellow, blue, red, green, etc.)
    # Prioritized over neutral background/shadows
    chromatic_masks = {
        "red": ((h_ch <= 9) | (h_ch >= 168)) & (s_ch >= 35) & (v_ch >= 35),
        "orange": (h_ch >= 10) & (h_ch < 22) & (s_ch >= 35) & (v_ch >= 45),
        "yellow": (h_ch >= 22) & (h_ch <= 38) & (s_ch >= 32) & (v_ch >= 45),
        "green": (h_ch >= 39) & (h_ch <= 85) & (s_ch >= 30) & (v_ch >= 35),
        "cyan": (h_ch >= 86) & (h_ch <= 95) & (s_ch >= 30) & (v_ch >= 35),
        "blue": (h_ch >= 96) & (h_ch <= 135) & (s_ch >= 30) & (v_ch >= 35),
        "purple": (h_ch >= 136) & (h_ch <= 155) & (s_ch >= 30) & (v_ch >= 35),
        "pink": (h_ch >= 156) & (h_ch < 168) & (s_ch >= 30) & (v_ch >= 45),
        "brown": (h_ch >= 10) & (h_ch <= 25) & (s_ch >= 35) & (v_ch >= 30) & (v_ch <= 130),
    }

    best_chrom = None
    max_chrom_ratio = 0.0

    for name, mask in chromatic_masks.items():
        ratio = float(np.count_nonzero(mask) / n_pixels)
        if ratio > max_chrom_ratio and ratio >= 0.12:
            max_chrom_ratio = ratio
            best_chrom = name

    if best_chrom is not None:
        return best_chrom

    # 2. Achromatic fallback (neutral clothing: black, white, gray)
    achromatic_masks = {
        "black": v_ch < 55,
        "white": (s_ch < 35) & (v_ch >= 155),
        "gray": (s_ch < 40) & (v_ch >= 55) & (v_ch < 155),
    }

    best_achrom = "unknown"
    max_achrom_ratio = 0.0

    for name, mask in achromatic_masks.items():
        ratio = float(np.count_nonzero(mask) / n_pixels)
        if ratio > max_achrom_ratio and ratio >= 0.18:
            max_achrom_ratio = ratio
            best_achrom = name

    return best_achrom


def classify_car_subtype(bbox: list[float]) -> str:
    """Classifies vehicle subtype (sedan, SUV, hatchback) based on bounding geometry."""
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    aspect = w / h

    if aspect >= 2.15:
        return "sedan"
    elif aspect < 1.90:
        return "SUV"
    else:
        return "hatchback"


def find_accessories_for_person(person_bbox: list[float], frame_dets: list[dict]) -> list[str]:
    """Finds accessories (backpack, suitcase, handbag) carried by or attached to a person."""
    px1, py1, px2, py2 = person_bbox
    accessories = []

    for d in frame_dets:
        cls = d.get("cls", "").lower()
        if cls not in ("backpack", "suitcase", "handbag"):
            continue

        bx = d.get("bbox", [0, 0, 0, 0])
        cx = (bx[0] + bx[2]) / 2.0
        cy = (bx[1] + bx[3]) / 2.0

        # Check if center of accessory lies inside or heavily overlaps person box
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            acc_name = "trolley bag" if cls == "suitcase" else cls
            color = d.get("color", "")
            desc = f"{color} {acc_name}".strip() if color else acc_name
            if desc not in accessories:
                accessories.append(desc)

    return accessories


def extract_attributes(crop_bgr: np.ndarray, cls_name: str, bbox: list[float], other_dets: list[dict] | None = None) -> dict:
    """Extracts rich attributes including upper/lower colors, subtype, and accessories."""
    attrs = {
        "cls": cls_name,
        "upper_color": None,
        "lower_color": None,
        "primary_color": None,
        "all_colors": [],
        "subtype": None,
        "accessories": [],
        "description": cls_name,
    }

    if crop_bgr is None or crop_bgr.size == 0:
        return attrs

    h, w = crop_bgr.shape[:2]
    primary_color = detect_dominant_color(crop_bgr)
    attrs["primary_color"] = primary_color if primary_color != "unknown" else None

    if cls_name == "person":
        # Torso crop: target 20% to 58% height, 18% to 82% width
        # This isolates the shirt, top, dress, or jacket from head/hair above and background on sides
        y_top = int(round(h * 0.20))
        y_bot = int(round(h * 0.58))
        x_left = int(round(w * 0.18))
        x_right = int(round(w * 0.82))

        torso_crop = (
            crop_bgr[y_top:y_bot, x_left:x_right]
            if (y_bot > y_top and x_right > x_left)
            else crop_bgr[: max(1, h // 2), :]
        )

        # Lower body crop: target 60% to 90% height, 18% to 82% width
        # This isolates the pants, trousers, jeans, or skirt
        y_lo_top = int(round(h * 0.60))
        y_lo_bot = int(round(h * 0.90))
        legs_crop = (
            crop_bgr[y_lo_top:y_lo_bot, x_left:x_right]
            if (y_lo_bot > y_lo_top and x_right > x_left)
            else crop_bgr[max(1, h // 2) :, :]
        )

        up_col = detect_dominant_color(torso_crop)
        lo_col = detect_dominant_color(legs_crop)

        attrs["upper_color"] = up_col if up_col != "unknown" else None
        attrs["lower_color"] = lo_col if lo_col != "unknown" else None

        all_cols = []
        if attrs["upper_color"]:
            all_cols.append(attrs["upper_color"])
        if attrs["lower_color"] and attrs["lower_color"] not in all_cols:
            all_cols.append(attrs["lower_color"])
        if attrs["primary_color"] and attrs["primary_color"] not in all_cols:
            all_cols.append(attrs["primary_color"])
        attrs["all_colors"] = all_cols

        # Check carried accessories from frame detections
        if other_dets:
            accessories = find_accessories_for_person(bbox, other_dets)
            attrs["accessories"] = accessories

        # Generate descriptive sentence
        desc_parts = ["person"]
        if attrs["upper_color"]:
            desc_parts.append(f"wearing {attrs['upper_color']} shirt/top/dress")
        if attrs["lower_color"]:
            desc_parts.append(f"and {attrs['lower_color']} pants")
        if attrs["accessories"]:
            desc_parts.append(f"carrying {', '.join(attrs['accessories'])}")

        attrs["description"] = " ".join(desc_parts)

    elif cls_name in ("car", "truck", "bus"):
        subtype = classify_car_subtype(bbox) if cls_name == "car" else cls_name
        attrs["subtype"] = subtype
        col_str = f"{attrs['primary_color']} " if attrs["primary_color"] else ""
        attrs["description"] = f"{col_str}{subtype}"

    elif cls_name in ("dog", "cat", "horse", "bird", "elephant", "bear", "cow", "sheep"):
        col_str = f"{attrs['primary_color']} " if attrs["primary_color"] else ""
        attrs["description"] = f"{col_str}{cls_name}"

    elif cls_name in ("backpack", "suitcase"):
        subtype = "trolley bag" if cls_name == "suitcase" else cls_name
        attrs["subtype"] = subtype
        col_str = f"{attrs['primary_color']} " if attrs["primary_color"] else ""
        attrs["description"] = f"{col_str}{subtype}"

    else:
        col_str = f"{attrs['primary_color']} " if attrs["primary_color"] else ""
        attrs["description"] = f"{col_str}{cls_name}"

    return attrs
