import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from vidsense import config
from vidsense.video_io import save_jpeg

_CACHED_CLASS_NAMES: list[str] | None = None


def create_model() -> YOLO:
    num_threads = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(num_threads)
    return YOLO(config.YOLO_WEIGHTS)


def class_names() -> list[str]:
    global _CACHED_CLASS_NAMES
    if _CACHED_CLASS_NAMES is None:
        model = create_model()
        # Ultralytics model.names is a dict {id: name}
        names_dict = getattr(model, "names", {})
        if isinstance(names_dict, dict):
            _CACHED_CLASS_NAMES = [names_dict[i] for i in range(len(names_dict))]
        elif isinstance(names_dict, list):
            _CACHED_CLASS_NAMES = list(names_dict)
        else:
            _CACHED_CLASS_NAMES = []
    return _CACHED_CLASS_NAMES


def track_frame(model: YOLO, frame_bgr):
    results = model.track(
        frame_bgr,
        persist=True,
        tracker=config.TRACKER,
        conf=config.YOLO_CONF,
        imgsz=config.YOLO_IMGSZ,
        device="cpu",
        verbose=False,
    )
    return results[0]


def draw_detections(frame_bgr, detections: list[dict]):
    annotated = frame_bgr.copy()
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cls_name = det.get("cls", "")
        track_id = det.get("track_id", "")
        label = f"{cls_name} #{track_id}"

        # Draw thin bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw background label box
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_text = max(y1 - 4, h + 2)
        cv2.rectangle(
            annotated,
            (x1, y_text - h - baseline),
            (x1 + w, y_text + baseline),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


from vidsense import attributes


class TrackAggregator:
    def __init__(self, source_id: str):
        self.source_id = source_id
        # Master tracks keyed by master_id
        self._tracks: dict[int, dict] = {}
        # Mapping from raw ByteTrack ID to persistent master track ID
        self._raw_to_master: dict[int, int] = {}
        # Appearance vectors for master tracks
        self._appearance_gallery: dict[int, np.ndarray] = {}
        # Next master ID counter
        self._next_master_id: int = 1
        # Multi-frame color observation voting: master_id -> dict of counts
        self._color_votes: dict[int, dict] = {}

    def update(self, ts: float, frame_bgr, result) -> list[dict]:
        current_detections = []
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return current_detections

        boxes = result.boxes
        has_id = boxes.id is not None
        if not has_id:
            return current_detections

        ids = boxes.id.cpu().numpy().astype(int)
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        names = result.names or {}

        h_frame, w_frame = frame_bgr.shape[:2]

        # First pass: collect basic frame detections for spatial accessory linking
        raw_frame_dets = []
        for i in range(len(ids)):
            raw_id = int(ids[i])
            if raw_id < 0:
                continue
            cls_id = int(classes[i])
            cls_name = names.get(cls_id, str(cls_id))
            bbox = [float(coords[i][0]), float(coords[i][1]), float(coords[i][2]), float(coords[i][3])]

            det_color = ""
            if cls_name in ("backpack", "suitcase", "handbag"):
                x1_c = max(0, int(round(bbox[0])))
                y1_c = max(0, int(round(bbox[1])))
                x2_c = min(w_frame, int(round(bbox[2])))
                y2_c = min(h_frame, int(round(bbox[3])))
                if x2_c > x1_c and y2_c > y1_c:
                    acc_crop = frame_bgr[y1_c:y2_c, x1_c:x2_c]
                    col = attributes.detect_dominant_color(acc_crop)
                    if col != "unknown":
                        det_color = col

            raw_frame_dets.append({"raw_id": raw_id, "cls": cls_name, "bbox": bbox, "color": det_color})

        # Second pass: process each detection with ReID & attribute extraction
        for i in range(len(ids)):
            raw_id = int(ids[i])
            if raw_id < 0:
                continue

            cls_id = int(classes[i])
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(confs[i])
            bbox = [float(coords[i][0]), float(coords[i][1]), float(coords[i][2]), float(coords[i][3])]

            # Extract cropped object image with slight padding
            pad_x = max(2, int((bbox[2] - bbox[0]) * 0.05))
            pad_y = max(2, int((bbox[3] - bbox[1]) * 0.05))
            x1 = max(0, int(round(bbox[0])) - pad_x)
            y1 = max(0, int(round(bbox[1])) - pad_y)
            x2 = min(w_frame, int(round(bbox[2])) + pad_x)
            y2 = min(h_frame, int(round(bbox[3])) + pad_y)

            crop = frame_bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

            # Compute visual appearance signature
            app_vec = attributes.compute_appearance_vector(crop)
            # Extract attributes (upper/lower clothing, car subtype, accessories)
            attrs = attributes.extract_attributes(crop, cls_name, bbox, raw_frame_dets)

            # Determine persistent master track ID
            master_id = self._resolve_master_id(raw_id, cls_name, ts, app_vec)

            # Build detection info for live overlay
            det_label = cls_name
            if attrs.get("subtype"):
                det_label = attrs["subtype"]
            elif attrs.get("upper_color"):
                det_label = f"{attrs['upper_color']} {cls_name}"

            current_detections.append({
                "track_id": master_id,
                "cls": cls_name,
                "conf": conf,
                "bbox": bbox,
                "label": det_label,
                "description": attrs.get("description", cls_name),
            })

            thumb_rel = f"{self.source_id}/{master_id}.jpg"
            full_thumb_path = os.path.join(config.DATA_DIR, "frames", self.source_id, f"{master_id}.jpg")

            if master_id not in self._tracks:
                # Save initial high quality crop
                if crop is not None and crop.size > 0:
                    save_jpeg(crop, full_thumb_path)
                else:
                    save_jpeg(frame_bgr, full_thumb_path)

                self._tracks[master_id] = {
                    "track_id": master_id,
                    "cls": cls_name,
                    "first_ts": ts,
                    "last_ts": ts,
                    "n_frames": 1,
                    "max_conf": conf,
                    "best_ts": ts,
                    "thumb_path": thumb_rel,
                    "attributes": attrs,
                }
                self._appearance_gallery[master_id] = app_vec
                self._color_votes[master_id] = {
                    "upper": {},
                    "lower": {},
                    "primary": {},
                    "all": set(),
                }
                if attrs.get("upper_color"):
                    self._color_votes[master_id]["upper"][attrs["upper_color"]] = 1.0
                    self._color_votes[master_id]["all"].add(attrs["upper_color"])
                if attrs.get("lower_color"):
                    self._color_votes[master_id]["lower"][attrs["lower_color"]] = 1.0
                    self._color_votes[master_id]["all"].add(attrs["lower_color"])
                if attrs.get("primary_color"):
                    self._color_votes[master_id]["primary"][attrs["primary_color"]] = 1.0
                    self._color_votes[master_id]["all"].add(attrs["primary_color"])
            else:
                trk = self._tracks[master_id]
                trk["last_ts"] = ts
                trk["n_frames"] += 1

                # Update appearance gallery with moving average
                existing_vec = self._appearance_gallery.get(master_id)
                if existing_vec is not None and app_vec is not None:
                    updated_vec = 0.75 * existing_vec + 0.25 * app_vec
                    norm = np.linalg.norm(updated_vec)
                    if norm > 0:
                        updated_vec /= norm
                    self._appearance_gallery[master_id] = updated_vec

                # Save best thumbnail on higher confidence
                if conf > trk["max_conf"]:
                    trk["max_conf"] = conf
                    trk["best_ts"] = ts
                    if crop is not None and crop.size > 0:
                        save_jpeg(crop, full_thumb_path)
                    trk["thumb_path"] = thumb_rel

                # Multi-frame color accumulation and voting across observations
                votes = self._color_votes.setdefault(
                    master_id, {"upper": {}, "lower": {}, "primary": {}, "all": set()}
                )

                def _get_weight(c_name: str) -> float:
                    return 1.0 if c_name in ("black", "white", "gray", "unknown") else 2.5

                if attrs.get("upper_color"):
                    c = attrs["upper_color"]
                    votes["upper"][c] = votes["upper"].get(c, 0.0) + _get_weight(c)
                    votes["all"].add(c)
                if attrs.get("lower_color"):
                    c = attrs["lower_color"]
                    votes["lower"][c] = votes["lower"].get(c, 0.0) + _get_weight(c)
                    votes["all"].add(c)
                if attrs.get("primary_color"):
                    c = attrs["primary_color"]
                    votes["primary"][c] = votes["primary"].get(c, 0.0) + _get_weight(c)
                    votes["all"].add(c)

                # Determine winning colors from votes across all frames
                cur_attrs = trk.get("attributes", {})
                if votes["upper"]:
                    top_up = max(votes["upper"].items(), key=lambda x: x[1])[0]
                    cur_attrs["upper_color"] = top_up
                if votes["lower"]:
                    top_lo = max(votes["lower"].items(), key=lambda x: x[1])[0]
                    cur_attrs["lower_color"] = top_lo
                if votes["primary"]:
                    top_pri = max(votes["primary"].items(), key=lambda x: x[1])[0]
                    cur_attrs["primary_color"] = top_pri

                if attrs.get("subtype"):
                    cur_attrs["subtype"] = attrs["subtype"]
                if attrs.get("accessories"):
                    for acc in attrs["accessories"]:
                        if acc not in cur_attrs.setdefault("accessories", []):
                            cur_attrs["accessories"].append(acc)

                cur_attrs["all_colors"] = list(votes["all"])

                # Update description
                if trk["cls"] == "person":
                    desc_parts = ["person"]
                    if cur_attrs.get("upper_color"):
                        desc_parts.append(f"wearing {cur_attrs['upper_color']} shirt/top/dress")
                    if cur_attrs.get("lower_color"):
                        desc_parts.append(f"and {cur_attrs['lower_color']} pants")
                    if cur_attrs.get("accessories"):
                        desc_parts.append(f"carrying {', '.join(cur_attrs['accessories'])}")
                    cur_attrs["description"] = " ".join(desc_parts)
                else:
                    col_str = f"{cur_attrs.get('primary_color')} " if cur_attrs.get("primary_color") else ""
                    sub_str = cur_attrs.get("subtype") or trk["cls"]
                    cur_attrs["description"] = f"{col_str}{sub_str}".strip()

                trk["attributes"] = cur_attrs

        return current_detections

    def _resolve_master_id(self, raw_id: int, cls_name: str, ts: float, app_vec: np.ndarray) -> int:
        """Resolves raw tracker ID to master track ID via appearance ReID."""
        # If raw ID already mapped, return persistent master ID
        if raw_id in self._raw_to_master:
            return self._raw_to_master[raw_id]

        # Check against existing master tracks for ReID match
        best_sim = 0.0
        best_master_id = None

        for m_id, m_trk in self._tracks.items():
            if m_trk["cls"] != cls_name:
                continue

            time_gap = ts - m_trk["last_ts"]
            # IMPORTANT: Two people in the SAME frame (time_gap < 0.4s) must NEVER be merged!
            # ReID only applies to re-identifying an identity after occlusion or reappearance (> 0.4s gap)
            if 0.4 <= time_gap <= 45.0:
                m_vec = self._appearance_gallery.get(m_id)
                if m_vec is not None:
                    sim = attributes.compare_appearance(app_vec, m_vec)
                    # High similarity threshold for re-identifying after reappearance
                    if sim > best_sim and sim >= 0.72:
                        best_sim = sim
                        best_master_id = m_id

        if best_master_id is not None:
            # Re-identified as existing object
            self._raw_to_master[raw_id] = best_master_id
            return best_master_id

        # New unique object
        new_master_id = self._next_master_id
        self._next_master_id += 1
        self._raw_to_master[raw_id] = new_master_id
        return new_master_id

    def tracks(self) -> list[dict]:
        return list(self._tracks.values())


def draw_styled_tag_box(
    img: np.ndarray,
    bbox: list[float],
    title: str,
    tags: list[str] | None = None,
    color: tuple[int, int, int] = (129, 185, 16),
    is_target: bool = True,
) -> None:
    """Draws a modern AI bounding box with corner accents and floating translucent label pill."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    th = 2 if is_target else 1
    # Main rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)

    # Stylish corner brackets
    corner_len = max(6, min(22, (x2 - x1) // 4, (y2 - y1) // 4))
    cth = th + 2
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, cth)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, cth)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, cth)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, cth)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, cth)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, cth)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, cth)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, cth)

    # Label text with badges
    label_text = title
    if tags:
        label_text += " | " + " | ".join(tags)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    (tw, th_box), base = cv2.getTextSize(label_text, font, scale, 1)

    bx1 = x1
    by1 = max(0, y1 - th_box - base - 8)
    bx2 = min(w, x1 + tw + 14)
    by2 = by1 + th_box + base + 8

    # Translucent dark background for label badge
    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (12, 16, 26), -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 1)
    cv2.putText(img, label_text, (bx1 + 6, by2 - base - 3), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def generate_annotated_keyframe(
    source_id: str,
    ts: float,
    track_id: int | None = None,
    target_cls: str | None = None,
) -> np.ndarray | None:
    """Extracts a frame at timestamp ts, runs detection, and renders highlighted boxes & tags."""
    from vidsense import db
    src = db.get_source(source_id)
    if not src:
        return None

    video_path = src.get("path") or ""
    abs_path = os.path.abspath(video_path) if video_path else ""
    if not (abs_path and os.path.exists(abs_path)):
        live_fin = os.path.abspath(os.path.join(config.DATA_DIR, "live", f"{source_id}.mp4"))
        live_raw = os.path.abspath(os.path.join(config.DATA_DIR, "live", f"{source_id}_raw.mp4"))
        if os.path.exists(live_fin):
            abs_path = live_fin
        elif os.path.exists(live_raw):
            abs_path = live_raw
        else:
            return None

    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(ts)) * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            return None
    finally:
        cap.release()

    h_frame, w_frame = frame.shape[:2]
    model = create_model()
    res = model.predict(
        frame,
        conf=max(0.20, config.YOLO_CONF - 0.15),
        imgsz=config.YOLO_IMGSZ,
        device="cpu",
        verbose=False,
    )[0]

    annotated = frame.copy()
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return annotated

    boxes = res.boxes
    coords = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    names = res.names or {}

    # Pass 1: gather frame detections with color analysis
    frame_dets = []
    for i in range(len(boxes)):
        c_id = int(classes[i])
        c_name = names.get(c_id, str(c_id))
        bx = [float(coords[i][0]), float(coords[i][1]), float(coords[i][2]), float(coords[i][3])]

        x1_c = max(0, int(round(bx[0])))
        y1_c = max(0, int(round(bx[1])))
        x2_c = min(w_frame, int(round(bx[2])))
        y2_c = min(h_frame, int(round(bx[3])))
        acc_col = ""
        if x2_c > x1_c and y2_c > y1_c:
            crop = frame[y1_c:y2_c, x1_c:x2_c]
            col = attributes.detect_dominant_color(crop)
            if col != "unknown":
                acc_col = col

        frame_dets.append({
            "cls": c_name,
            "bbox": bx,
            "color": acc_col,
            "conf": float(confs[i]),
        })

    # Find target match index
    target_idx = None
    if target_cls:
        best_c = -1.0
        for i, d in enumerate(frame_dets):
            if d["cls"].lower() == target_cls.lower() and d["conf"] > best_c:
                best_c = d["conf"]
                target_idx = i

    for i, d in enumerate(frame_dets):
        c_name = d["cls"]
        bx = d["bbox"]
        x1, y1, x2, y2 = [int(round(v)) for v in bx]
        is_target = (i == target_idx) if target_idx is not None else True

        if c_name == "person":
            crop = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
            attrs = attributes.extract_attributes(crop, c_name, bx, frame_dets)
            tags = []
            if attrs.get("upper_color"):
                tags.append(f"{attrs['upper_color']} shirt")
            if attrs.get("lower_color"):
                tags.append(f"{attrs['lower_color']} pants")
            if attrs.get("accessories"):
                for a in attrs["accessories"]:
                    tags.append(a)

            title = f"person #{track_id}" if (is_target and track_id is not None) else "person"
            draw_styled_tag_box(
                annotated,
                bx,
                title,
                tags=tags,
                color=(129, 185, 16),  # Emerald
                is_target=is_target,
            )

        elif c_name in ("backpack", "handbag", "suitcase"):
            acc_name = "trolley bag" if c_name == "suitcase" else ("bag" if c_name == "handbag" else "backpack")
            color = d.get("color", "")
            bag_label = f"{color} {acc_name}".strip() if color else acc_name
            draw_styled_tag_box(
                annotated,
                bx,
                bag_label,
                tags=[color] if color else None,
                color=(212, 182, 6),  # Cyan
                is_target=True,
            )

        elif c_name in ("car", "truck", "bus"):
            crop = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
            attrs = attributes.extract_attributes(crop, c_name, bx)
            tags = []
            if attrs.get("primary_color"):
                tags.append(attrs["primary_color"])
            if attrs.get("subtype"):
                tags.append(attrs["subtype"])
            draw_styled_tag_box(
                annotated,
                bx,
                c_name,
                tags=tags,
                color=(246, 130, 59),  # Blue
                is_target=is_target,
            )

        elif is_target or (target_idx is None and d["conf"] >= 0.4):
            crop = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
            attrs = attributes.extract_attributes(crop, c_name, bx)
            tags = [attrs["primary_color"]] if attrs.get("primary_color") else []
            draw_styled_tag_box(
                annotated,
                bx,
                c_name,
                tags=tags,
                color=(11, 158, 245),  # Amber
                is_target=is_target,
            )

    return annotated


