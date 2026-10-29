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
            raw_frame_dets.append({"raw_id": raw_id, "cls": cls_name, "bbox": bbox})

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

