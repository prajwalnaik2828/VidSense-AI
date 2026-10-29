import re
import json
from vidsense import config
from vidsense import llm
from vidsense import db
from vidsense import index
from vidsense import detect
from vidsense.video_io import fmt_ts

PARSE_PROMPT = """You convert a question about a video into JSON. Return ONLY valid JSON, no markdown, with exactly these keys:
{{"intent": "find" | "count" | "describe", "cls": string or null, "start_ts": number or null, "end_ts": number or null, "semantic_text": string or null}}
Rules:
- intent "count" for how-many questions, "find" for when/where/show-me questions, "describe" for what-is-happening questions.
- cls must be exactly one of these object classes, or null: {classes}. Map synonyms (man, woman, pedestrian -> person; bike -> bicycle; motorbike -> motorcycle). If unsure, use null.
- start_ts and end_ts are seconds from the start of the video. Convert phrases like "in the first minute" to 0 and 60. Use null if no time is mentioned.
- semantic_text is the descriptive part that object classes cannot capture (e.g. "wearing a red jacket", "carrying a box"), otherwise null.
Question: {question}"""

SEMANTIC_PROMPT = """You answer questions about a video using ONLY the timestamped scene descriptions below.
Rules:
- If the descriptions do not clearly support an answer, reply with exactly: {NOT_FOUND_MESSAGE}
- Be concise. Cite timestamps as (t=mm:ss).
Descriptions:
{context}
Question: {question}"""


def _rule_based_fallback(question: str, known_classes: list[str]) -> dict:
    """Robust rule-based parser used when LLM API quota is exhausted or unreachable."""
    q_lower = question.lower()

    synonyms = {
        "people": "person",
        "person": "person",
        "man": "person",
        "woman": "person",
        "men": "person",
        "women": "person",
        "pedestrian": "person",
        "pedestrians": "person",
        "human": "person",
        "humans": "person",
        "guy": "person",
        "girl": "person",
        "boy": "person",
        "cars": "car",
        "car": "car",
        "automobile": "car",
        "automobiles": "car",
        "vehicle": "car",
        "vehicles": "car",
        "sedan": "car",
        "sedans": "car",
        "suv": "car",
        "suvs": "car",
        "hatchback": "car",
        "hatchbacks": "car",
        "saloon": "car",
        "bikes": "bicycle",
        "bike": "bicycle",
        "bicycle": "bicycle",
        "bicycles": "bicycle",
        "motorbikes": "motorcycle",
        "motorbike": "motorcycle",
        "motorcycle": "motorcycle",
        "motorcycles": "motorcycle",
        "phones": "cell phone",
        "phone": "cell phone",
        "cellphone": "cell phone",
        "cellphones": "cell phone",
        "mobile": "cell phone",
        "chairs": "chair",
        "chair": "chair",
        "seats": "chair",
        "seat": "chair",
        "laptops": "laptop",
        "laptop": "laptop",
        "computer": "laptop",
        "computers": "laptop",
        "screens": "tv",
        "screen": "tv",
        "monitor": "tv",
        "monitors": "tv",
        "tv": "tv",
        "tvs": "tv",
        "television": "tv",
        "bottles": "bottle",
        "bottle": "bottle",
        "cups": "cup",
        "cup": "cup",
        "cats": "cat",
        "cat": "cat",
        "dogs": "dog",
        "dog": "dog",
        "backpacks": "backpack",
        "backpack": "backpack",
        "bag": "backpack",
        "bags": "backpack",
        "suitcase": "suitcase",
        "suitcases": "suitcase",
        "trolley": "suitcase",
        "trolly": "suitcase",
        "luggage": "suitcase",
        "books": "book",
        "book": "book",
        "bird": "bird",
        "birds": "bird",
        "horse": "horse",
        "horses": "horse",
        "cow": "cow",
        "cows": "cow",
        "sheep": "sheep",
    }

    matched_cls = None
    for word, target_cls in synonyms.items():
        if re.search(r"\b" + re.escape(word) + r"\b", q_lower):
            matched_cls = target_cls
            break

    if not matched_cls:
        for c in known_classes:
            if re.search(r"\b" + re.escape(c.lower()) + r"\b", q_lower):
                matched_cls = c
                break

    is_count = bool(re.search(r"\b(how many|count|number of|how much)\b", q_lower))
    is_find = bool(re.search(r"\b(when|where|show me|find|detect|any|first|seen|appear|look for|who was|which)\b", q_lower))

    if matched_cls:
        if is_count:
            return {
                "intent": "count",
                "cls": matched_cls,
                "start_ts": None,
                "end_ts": None,
                "semantic_text": None,
            }
        elif is_find or not is_count:
            return {
                "intent": "find",
                "cls": matched_cls,
                "start_ts": None,
                "end_ts": None,
                "semantic_text": None,
            }

    return {
        "intent": "describe",
        "cls": None,
        "start_ts": None,
        "end_ts": None,
        "semantic_text": question,
    }


def parse_query(question: str, known_classes: list[str]) -> dict:
    try:
        classes_str = ", ".join(known_classes)
        prompt = PARSE_PROMPT.format(classes=classes_str, question=question)
        raw = llm.generate(prompt).strip()

        # Strip code fences
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        data = json.loads(raw)
        if not isinstance(data, dict):
            return _rule_based_fallback(question, known_classes)

        intent = data.get("intent")
        if intent not in ("find", "count", "describe"):
            intent = "describe"

        cls = data.get("cls")
        if cls not in known_classes:
            cls = None

        start_ts = data.get("start_ts")
        if start_ts is not None:
            try:
                start_ts = float(start_ts)
            except (ValueError, TypeError):
                start_ts = None

        end_ts = data.get("end_ts")
        if end_ts is not None:
            try:
                end_ts = float(end_ts)
            except (ValueError, TypeError):
                end_ts = None

        semantic_text = data.get("semantic_text")
        if semantic_text is not None:
            semantic_text = str(semantic_text).strip()
            if not semantic_text:
                semantic_text = None

        # If LLM classified as count or find with a known cls, ensure semantic_text doesn't override it
        if intent in ("count", "find") and cls:
            if semantic_text and semantic_text.lower().strip() in (cls.lower(), question.lower()):
                semantic_text = None

        return {
            "intent": intent,
            "cls": cls,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "semantic_text": semantic_text,
        }
    except Exception:
        return _rule_based_fallback(question, known_classes)


def _parse_query_attributes(question: str) -> dict:
    """Extracts fine-grained visual attribute filters from the question."""
    q_lower = question.lower()

    colors = [
        "yellow", "red", "blue", "green", "black", "white",
        "gray", "grey", "orange", "pink", "purple", "brown", "cyan"
    ]
    detected_colors = [c for c in colors if re.search(r"\b" + c + r"\b", q_lower)]

    # Upper body clothing / dress / wearing
    upper_color = None
    m_upper = re.search(
        r"\b(" + "|".join(colors) + r")\s+(shirt|tshirt|t-shirt|top|jacket|hoodie|sweater|coat|tee|dress|suit|outfit|clothing|clothes|wear|attire)\b",
        q_lower,
    )
    if not m_upper:
        # Also match phrases like "wearing yellow dress", "wearing yellow", "in red", "dressed in blue"
        m_upper = re.search(
            r"\b(?:wearing|in|wears|dressed in)\s+(?:a\s+|an\s+)?(" + "|".join(colors) + r")(?:\s+(?:shirt|tshirt|t-shirt|top|jacket|hoodie|sweater|coat|tee|dress|suit|outfit|clothing|clothes|attire))?\b",
            q_lower,
        )
    if m_upper:
        upper_color = m_upper.group(1)

    # Lower body clothing
    lower_color = None
    m_lower = re.search(
        r"\b(" + "|".join(colors) + r")\s+(pants|jeans|trousers|shorts|bottom|skirt)\b",
        q_lower,
    )
    if m_lower:
        lower_color = m_lower.group(1)

    # Car subtypes
    subtype = None
    for st in ["sedan", "suv", "hatchback"]:
        if re.search(r"\b" + st + r"\b", q_lower):
            subtype = st
            break

    # Car color
    m_car_color = re.search(
        r"\b(" + "|".join(colors) + r")\s+(car|sedan|suv|hatchback|automobile|vehicle)\b",
        q_lower,
    )
    car_color = m_car_color.group(1) if m_car_color else None

    # Animal color
    m_animal_color = re.search(
        r"\b(" + "|".join(colors) + r")\s+(dog|cat|bird|horse|cow|sheep)\b",
        q_lower,
    )
    animal_color = m_animal_color.group(1) if m_animal_color else None

    # Accessories
    accessories = []
    for acc in ["backpack", "bag", "suitcase", "trolley", "luggage", "handbag"]:
        if re.search(r"\b" + acc + r"\b", q_lower):
            accessories.append(acc)

    return {
        "colors": detected_colors,
        "upper_color": upper_color,
        "lower_color": lower_color,
        "subtype": subtype,
        "car_color": car_color,
        "animal_color": animal_color,
        "accessories": accessories,
    }


def _score_track(track: dict, query_attrs: dict, question: str) -> tuple[float, list[str]]:
    """Calculates attribute match score and reasons for a given track."""
    attrs_val = track.get("attributes", "")
    if isinstance(attrs_val, str) and attrs_val.strip():
        try:
            attrs = json.loads(attrs_val)
        except Exception:
            attrs = {}
    elif isinstance(attrs_val, dict):
        attrs = attrs_val
    else:
        attrs = {}

    score = 0.0
    matched_aspects = []

    q_colors = query_attrs.get("colors", [])
    q_upper = query_attrs.get("upper_color")
    q_lower = query_attrs.get("lower_color")
    q_subtype = query_attrs.get("subtype")
    q_accs = query_attrs.get("accessories", [])
    q_car_color = query_attrs.get("car_color")
    q_animal_color = query_attrs.get("animal_color")

    trk_upper = (attrs.get("upper_color") or "").lower()
    trk_lower = (attrs.get("lower_color") or "").lower()
    trk_primary = (attrs.get("primary_color") or "").lower()
    trk_all = [str(c).lower() for c in attrs.get("all_colors", [])]
    trk_subtype = (attrs.get("subtype") or "").lower()
    trk_accs = [str(a).lower() for a in attrs.get("accessories", [])]
    trk_desc = (attrs.get("description") or "").lower()

    # Upper body / dress color matching
    if q_upper:
        if trk_upper == q_upper:
            score += 4.0
            matched_aspects.append(f"{q_upper} top/dress")
        elif q_upper in trk_all or q_upper == trk_primary:
            score += 3.0
            matched_aspects.append(f"{q_upper} clothing")
        elif q_upper in trk_desc:
            score += 2.0
            matched_aspects.append(f"{q_upper} clothing")
        else:
            score -= 1.0

    # Lower body color matching
    if q_lower:
        if trk_lower == q_lower:
            score += 3.5
            matched_aspects.append(f"{q_lower} bottom")
        elif q_lower in trk_lower or q_lower in trk_desc:
            score += 2.0
            matched_aspects.append(f"{q_lower} bottom")
        else:
            score -= 1.0

    # Car color matching
    if q_car_color:
        if q_car_color in trk_upper or q_car_color in trk_lower or q_car_color in trk_desc or q_car_color == trk_primary:
            score += 3.5
            matched_aspects.append(f"{q_car_color} vehicle")

    # Animal color matching
    if q_animal_color:
        if q_animal_color in trk_upper or q_animal_color in trk_lower or q_animal_color in trk_desc or q_animal_color == trk_primary:
            score += 3.5
            matched_aspects.append(f"{q_animal_color}")

    # General colors if not tied specifically
    for c in q_colors:
        if c not in (q_upper, q_lower, q_car_color, q_animal_color):
            if c == trk_upper:
                score += 3.0
                matched_aspects.append(f"{c} top/dress")
            elif c == trk_lower:
                score += 2.5
                matched_aspects.append(f"{c} bottom")
            elif c in trk_all or c == trk_primary:
                score += 2.5
                matched_aspects.append(f"{c} clothing")
            elif c in trk_desc:
                score += 1.5
                matched_aspects.append(c)

    # Car subtype matching
    if q_subtype:
        if trk_subtype == q_subtype:
            score += 4.0
            matched_aspects.append(q_subtype)
        elif q_subtype in trk_desc:
            score += 2.5
            matched_aspects.append(q_subtype)
        else:
            score -= 1.5

    # Accessories matching
    for acc in q_accs:
        if acc in trk_accs or any(acc in a for a in trk_accs):
            score += 3.5
            matched_aspects.append(acc)
        elif acc in ("bag", "backpack") and any("pack" in a or "bag" in a for a in trk_accs):
            score += 3.0
            matched_aspects.append(acc)
        elif acc in trk_desc:
            score += 2.0
            matched_aspects.append(acc)

    return score, matched_aspects


def answer_query(question: str, source_id: str | None) -> dict:
    if source_id is not None:
        src = db.get_source(source_id)
        if src is None:
            raise ValueError("Unknown source")

    classes = detect.class_names()
    parsed = parse_query(question, classes)
    query_attrs = _parse_query_attributes(question)

    intent = parsed["intent"]
    cls = parsed["cls"]
    start_ts = parsed["start_ts"]
    end_ts = parsed["end_ts"]
    semantic_text = parsed["semantic_text"]

    # Deduce class if not explicitly resolved but hinted by attributes
    if cls is None:
        if query_attrs.get("subtype") in ("sedan", "suv", "hatchback") or query_attrs.get("car_color"):
            cls = "car"
        elif query_attrs.get("upper_color") or query_attrs.get("lower_color"):
            cls = "person"
        elif query_attrs.get("animal_color"):
            # check animal
            for a in ["dog", "cat", "bird", "horse", "cow", "sheep"]:
                if a in question.lower():
                    cls = a
                    break

    has_attr_filter = bool(
        query_attrs["colors"]
        or query_attrs["upper_color"]
        or query_attrs["lower_color"]
        or query_attrs["subtype"]
        or query_attrs["car_color"]
        or query_attrs["animal_color"]
        or query_attrs["accessories"]
    )

    # 1. Attribute & Track Search
    if cls is not None or has_attr_filter:
        tracks = db.query_tracks(source_id, cls, start_ts, end_ts)
        if tracks:
            scored_tracks = []
            for t in tracks:
                score, matched_aspects = _score_track(t, query_attrs, question)
                scored_tracks.append((score, matched_aspects, t))

            if has_attr_filter:
                # Keep tracks with positive matching attributes
                filtered = [item for item in scored_tracks if item[0] >= 1.5]
                # If no strict threshold match, try items with score > 0
                if not filtered:
                    filtered = [item for item in scored_tracks if item[0] > 0]
            else:
                # No specific attribute filter requested, keep all tracks of this class
                filtered = [(1.0, [], t) for t in tracks]

            if filtered:
                # Sort by match score descending, then by track duration (n_frames) descending, then start_ts
                filtered.sort(key=lambda x: (x[0], x[2]["n_frames"], -x[2]["first_ts"]), reverse=True)
                matched_tracks = [item[2] for item in filtered]
                aspect_map = {item[2]["track_id"]: item[1] for item in filtered}

                # Format human-friendly response
                n_matches = len(matched_tracks)
                first_trk = matched_tracks[0]
                first_aspects = aspect_map.get(first_trk["track_id"], [])
                aspect_phrase = f" ({', '.join(first_aspects)})" if first_aspects else ""

                if intent == "count":
                    answer = (
                        f"I found {n_matches} distinct {cls or 'object'}{aspect_phrase} track(s) "
                        f"(tracker-based appearance matching ensures distinct identities)."
                    )
                else:
                    first_time = fmt_ts(first_trk["first_ts"])
                    last_time = fmt_ts(first_trk["last_ts"])
                    answer = (
                        f"Pinpointed {cls or 'object'} #{first_trk['track_id']}{aspect_phrase} "
                        f"in continuous video from {first_time} to {last_time}."
                    )

                results = []
                for t in matched_tracks:
                    t_aspects = aspect_map.get(t["track_id"], [])
                    t_desc = f" ({', '.join(t_aspects)})" if t_aspects else ""
                    results.append({
                        "source_id": t["source_id"],
                        "source_name": t.get("source_name") or "",
                        "kind": "track",
                        "ts_start": float(t["first_ts"]),
                        "ts_end": float(t["last_ts"]),
                        "label": f"{t['cls']} #{t['track_id']}{t_desc}",
                        "thumbnail": t.get("thumb_path") or None,
                    })

                results.sort(key=lambda x: x["ts_start"])
                results = results[: config.MAX_RESULTS]
                return {"answer": answer, "results": results, "found": True}

    # 2. Semantic Scene Search Fallback
    search_text = semantic_text if semantic_text else question
    semantic_res = _answer_semantic(question, search_text, source_id)
    if semantic_res.get("found"):
        return semantic_res

    # 3. Not Found
    return {"answer": config.NOT_FOUND_MESSAGE, "results": [], "found": False}


def _answer_semantic(question: str, search_text: str, source_id: str | None) -> dict:
    captions = index.search_captions(search_text, source_id=source_id)
    if not captions:
        return {"answer": config.NOT_FOUND_MESSAGE, "results": [], "found": False}

    context_lines = []
    for c in captions:
        src = db.get_source(c["source_id"])
        s_name = src["name"] if src else ""
        context_lines.append(f"[{s_name} t={fmt_ts(c['ts'])}] {c['text']}")

    context_str = "\n".join(context_lines)
    prompt = SEMANTIC_PROMPT.format(
        NOT_FOUND_MESSAGE=config.NOT_FOUND_MESSAGE,
        context=context_str,
        question=question,
    )
    try:
        reply = llm.generate(prompt).strip()
    except Exception:
        top_c = captions[0]
        reply = f"Detected in video at (t={fmt_ts(top_c['ts'])}): {top_c['text']}"

    if config.NOT_FOUND_MESSAGE in reply:
        return {"answer": config.NOT_FOUND_MESSAGE, "results": [], "found": False}

    results = []
    for c in captions:
        src = db.get_source(c["source_id"])
        s_name = src["name"] if src else ""
        results.append({
            "source_id": c["source_id"],
            "source_name": s_name,
            "kind": "caption",
            "ts_start": float(c["ts"]),
            "ts_end": float(c["ts"] + config.CAPTION_INTERVAL_S),
            "label": c["text"],
            "thumbnail": None,
        })

    results.sort(key=lambda x: x["ts_start"])
    results = results[: config.MAX_RESULTS]
    return {"answer": reply, "results": results, "found": True}
