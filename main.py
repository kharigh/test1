r"""
How to call this file (from PowerShell in project root):

1) Extract lines from pic.jpeg and print JSON array of lines:
    .\.venv\Scripts\python.exe ocr_lines.py --mode lines --image pic.jpeg

2) Extract numeric array data:
    .\.venv\Scripts\python.exe ocr_lines.py --mode array --image pic.jpeg

3) Save output to CSV:
    .\.venv\Scripts\python.exe ocr_lines.py --mode lines --image pic.jpeg --csv lines.csv
    .\.venv\Scripts\python.exe ocr_lines.py --mode array --image pic.jpeg --csv array.csv

Optional:
- Use --cols N to force/pad array rows to N columns in array mode.
- Use --model-root to force local PaddleOCR model folder (offline mode).
"""

import argparse
import csv
import json
import re
import os
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR


def _default_model_root() -> Path:
    env_root = os.environ.get("PADDLE_OCR_MODEL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path.home() / ".paddlex" / "official_models").resolve()


def _pick_model_dir(root: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _validate_model_dir(path: Path, model_kind: str) -> None:
    required = ["inference.json", "inference.pdiparams"]
    missing = [filename for filename in required if not (path / filename).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"{model_kind} model is incomplete at {path}. Missing: {missing_text}")


def _resolve_local_model_dirs(model_root: Optional[Path] = None) -> Dict[str, str]:
    root = model_root.resolve() if model_root else _default_model_root()
    if not root.exists():
        raise FileNotFoundError(
            f"Local model root not found: {root}. "
            "Provide --model-root or set PADDLE_OCR_MODEL_ROOT."
        )

    det_dir = _pick_model_dir(root, ["PP-OCRv5_server_det", "PP-OCRv5_mobile_det"])
    rec_dir = _pick_model_dir(root, ["en_PP-OCRv5_mobile_rec", "en_PP-OCRv5_server_rec"])
    textline_dir = _pick_model_dir(root, ["PP-LCNet_x1_0_textline_ori"])

    if not det_dir or not rec_dir or not textline_dir:
        raise FileNotFoundError(
            "Required local models not found under "
            f"{root}. Expected folders like: "
            "PP-OCRv5_server_det, en_PP-OCRv5_mobile_rec, PP-LCNet_x1_0_textline_ori"
        )

    _validate_model_dir(det_dir, "Text detection")
    _validate_model_dir(rec_dir, "Text recognition")
    _validate_model_dir(textline_dir, "Textline orientation")

    return {
        "text_detection_model_name": det_dir.name,
        "text_detection_model_dir": str(det_dir),
        "text_recognition_model_name": rec_dir.name,
        "text_recognition_model_dir": str(rec_dir),
        "textline_orientation_model_name": textline_dir.name,
        "textline_orientation_model_dir": str(textline_dir),
    }


def _word_entry(detection) -> Dict[str, float]:
    points, text, confidence = detection
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    left = float(min(xs))
    right = float(max(xs))
    top = float(min(ys))
    bottom = float(max(ys))
    height = max(1.0, bottom - top)

    return {
        "text": text.strip(),
        "confidence": float(confidence),
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "height": height,
        "center_y": (top + bottom) / 2.0,
    }


def _load_image_variants(image_path: Path) -> List[np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    upscaled = cv2.resize(gray_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    clahe_bgr = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

    return [image, gray_bgr, upscaled, clahe_bgr, adaptive_bgr]


def _create_reader(model_root: Optional[Path] = None) -> PaddleOCR:
    local_dirs = _resolve_local_model_dirs(model_root)
    return PaddleOCR(
        device="cpu",
        text_detection_model_name=local_dirs["text_detection_model_name"],
        text_detection_model_dir=local_dirs["text_detection_model_dir"],
        text_recognition_model_name=local_dirs["text_recognition_model_name"],
        text_recognition_model_dir=local_dirs["text_recognition_model_dir"],
        textline_orientation_model_name=local_dirs["textline_orientation_model_name"],
        textline_orientation_model_dir=local_dirs["textline_orientation_model_dir"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


def _is_paddle_detection(item: object) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return False
    points, rec = item
    if not isinstance(points, (list, tuple)) or not isinstance(rec, (list, tuple)):
        return False
    if len(rec) < 2 or not isinstance(rec[0], str):
        return False
    return True


def _flatten_paddle_result(result: object) -> List:
    flattened: List = []

    def _walk(node: object) -> None:
        if _is_paddle_detection(node):
            flattened.append(node)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)

    _walk(result)
    return flattened


def _read_detections(reader: PaddleOCR, image: np.ndarray, numeric_only: bool) -> List:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raw = reader.predict(image)
    detections = []

    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "rec_texts" in raw[0]:
        for page in raw:
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polys = page.get("dt_polys", [])
            for index, text in enumerate(texts):
                normalized_text = str(text).strip()
                if not normalized_text:
                    continue
                if numeric_only and not re.search(r"\d", normalized_text):
                    continue

                if index < len(polys):
                    poly = polys[index]
                    points = poly.tolist() if hasattr(poly, "tolist") else poly
                else:
                    points = [[0, 0], [1, 0], [1, 1], [0, 1]]

                confidence = float(scores[index]) if index < len(scores) else 0.0
                detections.append((points, normalized_text, confidence))
        return detections

    for entry in _flatten_paddle_result(raw):
        points, rec = entry
        text = str(rec[0]).strip()
        if not text:
            continue
        confidence = float(rec[1]) if len(rec) > 1 else 0.0
        if numeric_only and not re.search(r"\d", text):
            continue
        detections.append((points, text, confidence))
    return detections


def _best_detections(reader: PaddleOCR, variants: List[np.ndarray], numeric_only: bool) -> List:
    best = []
    best_score = -1.0
    for variant in variants:
        detections = _read_detections(reader, variant, numeric_only=numeric_only)
        if not detections:
            continue
        confidence_sum = sum(float(item[2]) for item in detections)
        score = confidence_sum / max(1, len(detections))
        if score > best_score:
            best_score = score
            best = detections
    return best


def _vertical_overlap_ratio(word: Dict[str, float], line: Dict[str, float]) -> float:
    overlap = max(0.0, min(word["bottom"], line["bottom"]) - max(word["top"], line["top"]))
    denom = min(word["height"], max(1.0, line["median_height"]))
    return overlap / denom


def _cluster_lines(words: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    if not words:
        return []

    heights = [word["height"] for word in words]
    median_height = median(heights)
    center_tolerance = max(4.0, 0.35 * median_height)
    min_overlap_ratio = 0.45

    sorted_words = sorted(words, key=lambda word: (word["center_y"], word["left"]))
    line_clusters: List[Dict[str, object]] = []

    for word in sorted_words:
        best_index = -1
        best_score = -1.0

        for index, line in enumerate(line_clusters):
            overlap_ratio = _vertical_overlap_ratio(word, line)
            center_distance = abs(word["center_y"] - line["center_y"])
            if overlap_ratio >= min_overlap_ratio or center_distance <= center_tolerance:
                score = overlap_ratio - (center_distance / max(center_tolerance, 1.0)) * 0.1
                if score > best_score:
                    best_score = score
                    best_index = index

        if best_index == -1:
            line_clusters.append(
                {
                    "items": [word],
                    "top": word["top"],
                    "bottom": word["bottom"],
                    "center_y": word["center_y"],
                    "median_height": word["height"],
                }
            )
            continue

        selected = line_clusters[best_index]
        selected_items = selected["items"]
        selected_items.append(word)
        selected["top"] = min(selected["top"], word["top"])
        selected["bottom"] = max(selected["bottom"], word["bottom"])
        selected["center_y"] = sum(item["center_y"] for item in selected_items) / len(selected_items)
        selected["median_height"] = median(item["height"] for item in selected_items)

    ordered_lines = sorted(line_clusters, key=lambda line: line["center_y"])
    return [sorted(line["items"], key=lambda item: item["left"]) for line in ordered_lines]


def _normalize_numeric_text(text: str) -> str:
    replacements = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "Q": "0",
            "I": "1",
            "l": "1",
            "|": "1",
            "S": "5",
            "s": "5",
            "B": "8",
            "G": "6",
            "g": "9",
        }
    )
    cleaned = text.translate(replacements)
    return re.sub(r"[^0-9,\-\s\[\]]", " ", cleaned)


def _number_tokens(detection) -> List[Dict[str, float]]:
    points, raw_text, confidence = detection
    text = _normalize_numeric_text(raw_text)
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return []

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left = float(min(xs))
    right = float(max(xs))
    top = float(min(ys))
    bottom = float(max(ys))
    height = max(1.0, bottom - top)
    width = max(1.0, right - left)

    tokens: List[Dict[str, float]] = []
    segment_width = width / max(1, len(numbers))
    for idx, token in enumerate(numbers):
        token_left = left + idx * segment_width
        token_right = token_left + segment_width
        tokens.append(
            {
                "text": token,
                "value": int(token),
                "confidence": float(confidence),
                "left": token_left,
                "right": token_right,
                "top": top,
                "bottom": bottom,
                "height": height,
                "center_y": (top + bottom) / 2.0,
                "center_x": (token_left + token_right) / 2.0,
            }
        )
    return tokens


def _infer_column_count(rows: List[List[Dict[str, float]]], expected_cols: Optional[int]) -> int:
    if expected_cols and expected_cols > 0:
        return expected_cols

    lengths = [len(row) for row in rows if row]
    if not lengths:
        return 0

    freq: Dict[int, int] = {}
    for count in lengths:
        freq[count] = freq.get(count, 0) + 1
    mode_count = max(freq.items(), key=lambda item: (item[1], -item[0]))[0]
    if mode_count <= 1:
        return int(median(lengths))
    return mode_count


def _column_centers(rows: List[List[Dict[str, float]]], columns: int) -> List[float]:
    if columns <= 0:
        return []

    buckets: List[List[float]] = [[] for _ in range(columns)]
    for row in rows:
        ordered = sorted(row, key=lambda item: item["center_x"])
        if len(ordered) < max(2, int(0.8 * columns)):
            continue
        if len(ordered) == columns:
            sampled = ordered
        else:
            sampled = []
            for index in range(columns):
                source_index = round(index * (len(ordered) - 1) / max(1, columns - 1))
                sampled.append(ordered[source_index])

        for idx, token in enumerate(sampled):
            buckets[idx].append(token["center_x"])

    if any(buckets):
        fallback = []
        for bucket in buckets:
            if bucket:
                fallback.append(float(median(bucket)))
            elif fallback:
                fallback.append(fallback[-1] + 20.0)
            else:
                fallback.append(0.0)
        return fallback

    widest_row = max(rows, key=lambda row: len(row), default=[])
    widest = sorted(widest_row, key=lambda item: item["center_x"])
    if not widest:
        return []
    if len(widest) == columns:
        return [item["center_x"] for item in widest]

    min_x = min(item["center_x"] for item in widest)
    max_x = max(item["center_x"] for item in widest)
    if columns == 1:
        return [min_x]
    step = (max_x - min_x) / max(1, columns - 1)
    return [min_x + idx * step for idx in range(columns)]


def _assign_row_to_columns(
    row_tokens: List[Dict[str, float]],
    centers: List[float],
) -> List[Optional[int]]:
    rebuilt: List[Optional[int]] = [None] * len(centers)
    if not centers:
        return rebuilt

    for token in sorted(row_tokens, key=lambda item: item["center_x"]):
        candidate_order = sorted(
            range(len(centers)),
            key=lambda index: abs(token["center_x"] - centers[index]),
        )
        for index in candidate_order:
            if rebuilt[index] is None:
                rebuilt[index] = token["value"]
                break

    return rebuilt


def _filter_numeric_tokens(tokens: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not tokens:
        return []

    confident = [token for token in tokens if token["confidence"] >= 0.25]
    if not confident:
        confident = tokens

    lengths = [len(token["text"].lstrip("-")) for token in confident if token["text"].lstrip("-").isdigit()]
    if lengths:
        freq: Dict[int, int] = {}
        for length in lengths:
            freq[length] = freq.get(length, 0) + 1
        dominant_length = max(freq.items(), key=lambda item: (item[1], item[0]))[0]
        confident = [
            token
            for token in confident
            if abs(len(token["text"].lstrip("-")) - dominant_length) <= 1
            and len(token["text"].lstrip("-")) >= 2
        ]

    row_clusters = _cluster_lines(confident)
    if not row_clusters:
        return confident

    row_sizes = [len(row) for row in row_clusters]
    typical_row_size = max(2, int(median(row_sizes)))
    kept_rows = [row for row in row_clusters if len(row) >= max(2, int(0.5 * typical_row_size))]

    filtered: List[Dict[str, float]] = []
    for row in kept_rows:
        filtered.extend(row)
    return filtered


def extract_lines(image_path: Path, model_root: Optional[Path] = None) -> List[str]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    reader = _create_reader(model_root=model_root)
    variants = _load_image_variants(image_path)[:1]
    detections = _best_detections(reader, variants, numeric_only=False)

    words = [_word_entry(detection) for detection in detections if detection[1].strip()]
    grouped_lines = _cluster_lines(words)

    line_array: List[str] = []
    for line_words in grouped_lines:
        line_text = " ".join(item["text"] for item in line_words if item["text"])
        if line_text:
            line_array.append(line_text)

    return line_array


def _normalize_array_text(text: str) -> str:
    replacements = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "Q": "0",
            "I": "1",
            "l": "1",
            "|": "1",
            "S": "5",
            "s": "5",
            "B": "8",
            "G": "6",
            "g": "9",
            "（": "(",
            "）": ")",
            "，": ",",
            ";": ",",
        }
    )
    return text.translate(replacements)


def _parse_row_text(row_text: str) -> List[int]:
    parsed: List[int] = []
    for part in row_text.split(","):
        match = re.search(r"-?\d+", part)
        if match:
            parsed.append(int(match.group(0)))
    return parsed


def _extract_assigned_parenthesis_groups(text: str) -> List[str]:
    groups: List[str] = []
    index = 0
    while index < len(text):
        eq_pos = text.find("=", index)
        if eq_pos == -1:
            break

        pointer = eq_pos + 1
        while pointer < len(text) and text[pointer].isspace():
            pointer += 1

        if pointer >= len(text) or text[pointer] != "(":
            index = eq_pos + 1
            continue

        start = pointer
        depth = 0
        end = -1
        while pointer < len(text):
            char = text[pointer]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = pointer
                    break
            pointer += 1

        if end != -1:
            groups.append(text[start : end + 1])
            index = end + 1
        else:
            index = eq_pos + 1

    return groups


def _extract_rows_from_section(section_text: str) -> List[List[int]]:
    rows: List[List[int]] = []

    ordered_matches = re.finditer(
        r"\(([^()]*)\)|\(?\s*-?\d+(?:\s*,\s*-?\d+){3,}\s*\)?",
        section_text,
    )
    for match in ordered_matches:
        source = match.group(1) if match.group(1) is not None else match.group(0)
        row = _parse_row_text(source)
        if len(row) >= 4:
            rows.append(row)

    unique_rows: List[List[int]] = []
    seen = set()
    for row in rows:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def _parse_arrays_from_lines(lines: List[str]) -> List[List[List[int]]]:
    text = _normalize_array_text("\n".join(lines))
    arrays: List[List[List[int]]] = []

    label_matches = list(re.finditer(r"[A-Za-z_][A-Za-z0-9_]*_u16", text))
    if label_matches:
        for index, match in enumerate(label_matches):
            start = match.start()
            end = label_matches[index + 1].start() if index + 1 < len(label_matches) else len(text)
            section = text[start:end]
            rows = _extract_rows_from_section(section)
            if rows:
                arrays.append(rows)

    if arrays:
        return arrays

    assigned_groups = _extract_assigned_parenthesis_groups(text)
    candidate_groups = assigned_groups if assigned_groups else [text]
    for group_text in candidate_groups:
        rows = _extract_rows_from_section(group_text)
        if rows:
            arrays.append(rows)

    return arrays


def _row_candidates_from_detections(detections: List) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for points, raw_text, _confidence in detections:
        text = _normalize_array_text(str(raw_text))
        matches = re.finditer(r"\(?\s*-?\d+(?:\s*,\s*-?\d+){5,}\s*\)?", text)
        ys = [p[1] for p in points]
        xs = [p[0] for p in points]
        center_y = float(sum(ys) / max(1, len(ys)))
        left = float(min(xs)) if xs else 0.0

        for match in matches:
            row = _parse_row_text(match.group(0))
            if len(row) >= 6:
                candidates.append(
                    {
                        "row": row,
                        "center_y": center_y,
                        "left": left,
                    }
                )

    if not candidates:
        return []

    candidates.sort(key=lambda item: (float(item["center_y"]), float(item["left"])))

    ys = [float(item["center_y"]) for item in candidates]
    if len(ys) > 1:
        gaps = [abs(ys[index] - ys[index - 1]) for index in range(1, len(ys))]
        positive_gaps = [gap for gap in gaps if gap > 0]
        typical_gap = median(positive_gaps) if positive_gaps else 12.0
    else:
        typical_gap = 12.0
    y_tolerance = max(6.0, 0.25 * typical_gap)

    merged: List[Dict[str, object]] = []
    active_group: List[Dict[str, object]] = [candidates[0]]

    for candidate in candidates[1:]:
        if abs(float(candidate["center_y"]) - float(active_group[-1]["center_y"])) <= y_tolerance:
            active_group.append(candidate)
            continue

        active_group.sort(key=lambda item: float(item["left"]))
        merged_row: List[int] = []
        for item in active_group:
            merged_row.extend(int(value) for value in item["row"])
        if merged_row:
            merged.append(
                {
                    "row": merged_row,
                    "center_y": sum(float(item["center_y"]) for item in active_group) / len(active_group),
                    "left": min(float(item["left"]) for item in active_group),
                }
            )
        active_group = [candidate]

    active_group.sort(key=lambda item: float(item["left"]))
    trailing_row: List[int] = []
    for item in active_group:
        trailing_row.extend(int(value) for value in item["row"])
    if trailing_row:
        merged.append(
            {
                "row": trailing_row,
                "center_y": sum(float(item["center_y"]) for item in active_group) / len(active_group),
                "left": min(float(item["left"]) for item in active_group),
            }
        )

    lengths = [len(item["row"]) for item in merged if item["row"]]
    if not lengths:
        return []

    freq: Dict[int, int] = {}
    for length in lengths:
        freq[length] = freq.get(length, 0) + 1
    dominant_len = max(freq.items(), key=lambda entry: (entry[1], entry[0]))[0]

    filtered = [
        item
        for item in merged
        if abs(len(item["row"]) - dominant_len) <= 1
    ]

    return filtered


def _array_labels_from_detections(detections: List) -> List[Dict[str, object]]:
    labels: List[Dict[str, object]] = []
    for points, raw_text, _confidence in detections:
        text = str(raw_text)
        if not re.search(r"[A-Za-z_][A-Za-z0-9_]*_u16", text):
            continue
        ys = [p[1] for p in points]
        center_y = float(sum(ys) / max(1, len(ys)))
        labels.append({"center_y": center_y, "text": text})

    labels.sort(key=lambda item: float(item["center_y"]))
    return labels


def _group_rows_by_labels(
    row_candidates: List[Dict[str, object]],
    labels: List[Dict[str, object]],
) -> List[List[List[int]]]:
    if not row_candidates:
        return []

    if not labels:
        return [[item["row"] for item in row_candidates]]

    buckets: List[List[List[int]]] = [[] for _ in labels]
    for item in row_candidates:
        row_y = float(item["center_y"])
        label_index = 0
        for idx, label in enumerate(labels):
            if row_y >= float(label["center_y"]):
                label_index = idx
            else:
                break
        buckets[label_index].append(item["row"])

    return [bucket for bucket in buckets if bucket]


def _group_row_candidates_by_labels(
    row_candidates: List[Dict[str, object]],
    labels: List[Dict[str, object]],
) -> List[List[Dict[str, object]]]:
    if not row_candidates:
        return []

    if not labels:
        return [list(row_candidates)]

    buckets: List[List[Dict[str, object]]] = [[] for _ in labels]
    for item in row_candidates:
        row_y = float(item["center_y"])
        label_index = 0
        for idx, label in enumerate(labels):
            if row_y >= float(label["center_y"]):
                label_index = idx
            else:
                break
        buckets[label_index].append(item)
    return [bucket for bucket in buckets if bucket]


def _collapse_candidate_bucket(
    bucket: List[Dict[str, object]],
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
) -> List[List[int]]:
    if not bucket:
        return []

    ordered = sorted(bucket, key=lambda item: float(item["center_y"]))
    ys = [float(item["center_y"]) for item in ordered]
    if len(ys) > 1:
        gaps = [abs(ys[index] - ys[index - 1]) for index in range(1, len(ys))]
        positive_gaps = [gap for gap in gaps if gap > 0]
        typical_gap = median(positive_gaps) if positive_gaps else 8.0
    else:
        typical_gap = 8.0

    def _cluster_with_tolerance(tolerance: float) -> List[List[Dict[str, object]]]:
        result: List[List[Dict[str, object]]] = [[ordered[0]]]
        for candidate in ordered[1:]:
            if abs(float(candidate["center_y"]) - float(result[-1][-1]["center_y"])) <= tolerance:
                result[-1].append(candidate)
            else:
                result.append([candidate])
        return result

    tolerance_candidates = [
        max(1.0, 0.12 * typical_gap),
        max(1.0, 0.16 * typical_gap),
        max(1.0, 0.20 * typical_gap),
        max(1.0, 0.24 * typical_gap),
        max(1.0, 0.30 * typical_gap),
    ]

    cluster_candidates = [_cluster_with_tolerance(tolerance) for tolerance in tolerance_candidates]
    if expected_rows and expected_rows > 0:
        clusters = min(cluster_candidates, key=lambda grouped: abs(len(grouped) - expected_rows))
    else:
        clusters = cluster_candidates[2]

    if expected_cols and expected_cols > 0:
        target_cols = int(expected_cols)
    else:
        lengths = [len(item["row"]) for item in bucket if item.get("row")]
        target_cols = int(median(lengths)) if lengths else 0

    selected: List[Dict[str, object]] = []
    for cluster in clusters:
        best = max(
            cluster,
            key=lambda item: (
                -abs(len(item["row"]) - target_cols) if target_cols > 0 else len(item["row"]),
                len(item["row"]),
                -float(item["left"]),
            ),
        )
        selected.append(best)

    selected.sort(key=lambda item: float(item["center_y"]))

    if expected_rows and expected_rows > 0 and len(selected) < expected_rows:
        selected_keys = {(tuple(item["row"]), float(item["center_y"])) for item in selected}
        remaining = [
            item
            for item in ordered
            if (tuple(item["row"]), float(item["center_y"])) not in selected_keys
        ]

        def _row_quality(item: Dict[str, object]) -> float:
            row_len = len(item["row"])
            len_score = -abs(row_len - target_cols) if target_cols > 0 else float(row_len)
            y = float(item["center_y"])
            if not selected:
                spacing_score = 0.0
            else:
                spacing_score = min(abs(y - float(existing["center_y"])) for existing in selected)
            return len_score + (0.05 * spacing_score)

        while len(selected) < expected_rows and remaining:
            best_extra = max(remaining, key=_row_quality)
            selected.append(best_extra)
            remaining.remove(best_extra)

        selected.sort(key=lambda item: float(item["center_y"]))

    eligible_rows: List[Dict[str, object]] = []
    collapsed_rows: List[List[int]] = []
    for item in selected:
        row = [int(value) for value in item["row"]]
        if target_cols > 0:
            if len(row) > target_cols:
                row = row[:target_cols]
            elif len(row) < target_cols:
                continue

        eligible_rows.append({"center_y": float(item["center_y"]), "row": row})
        collapsed_rows.append(row)

    if expected_rows and expected_rows > 0 and len(collapsed_rows) < expected_rows:
        all_eligible: List[Dict[str, object]] = []
        for item in ordered:
            row = [int(value) for value in item["row"]]
            if target_cols > 0:
                if len(row) > target_cols:
                    row = row[:target_cols]
                elif len(row) < target_cols:
                    continue
            all_eligible.append({"center_y": float(item["center_y"]), "row": row})

        if len(all_eligible) >= expected_rows:
            all_eligible.sort(key=lambda candidate: float(candidate["center_y"]))
            evenly_spaced: List[List[int]] = []
            used_indices = set()

            if expected_rows == 1:
                evenly_spaced.append(list(all_eligible[0]["row"]))
            else:
                max_index = len(all_eligible) - 1
                for slot in range(expected_rows):
                    raw_index = round(slot * max_index / (expected_rows - 1))
                    index = int(raw_index)
                    if index in used_indices:
                        shift = 1
                        while index in used_indices and (index - shift >= 0 or index + shift <= max_index):
                            left = index - shift
                            right = index + shift
                            if left >= 0 and left not in used_indices:
                                index = left
                                break
                            if right <= max_index and right not in used_indices:
                                index = right
                                break
                            shift += 1
                    used_indices.add(index)
                    evenly_spaced.append(list(all_eligible[index]["row"]))

            if len(evenly_spaced) >= expected_rows:
                collapsed_rows = evenly_spaced[:expected_rows]

    if expected_rows and expected_rows > 0:
        collapsed_rows = collapsed_rows[:expected_rows]

    return collapsed_rows


def _normalize_array_block(rows: List[List[int]]) -> List[List[int]]:
    cleaned: List[List[int]] = [list(row) for row in rows if row]
    if not cleaned:
        return []

    lengths = [len(row) for row in cleaned]
    freq: Dict[int, int] = {}
    for length in lengths:
        freq[length] = freq.get(length, 0) + 1
    dominant_len = max(freq.items(), key=lambda item: (item[1], item[0]))[0]

    filtered = [row for row in cleaned if abs(len(row) - dominant_len) <= 1]
    if not filtered:
        filtered = cleaned

    return filtered


def _score_array_block(rows: List[List[int]]) -> float:
    if not rows:
        return -1.0

    lengths = [len(row) for row in rows if row]
    if not lengths:
        return -1.0

    freq: Dict[int, int] = {}
    for length in lengths:
        freq[length] = freq.get(length, 0) + 1
    dominant_len = max(freq.items(), key=lambda item: (item[1], item[0]))[0]
    median_len = float(median(lengths))
    total_values = float(sum(lengths))
    consistent_rows = sum(1 for length in lengths if abs(length - dominant_len) <= 1)
    consistency_ratio = float(consistent_rows) / float(len(lengths))
    row_count_penalty = float(len(lengths) * 2)
    return (median_len * 100.0) + (consistency_ratio * 50.0) + total_values - row_count_penalty


def _score_arrays(arrays: List[List[List[int]]]) -> float:
    if not arrays:
        return -1.0
    return sum(_score_array_block(block) for block in arrays)


def extract_numeric_arrays(
    image_path: Path,
    model_root: Optional[Path] = None,
    expected_arrays: Optional[int] = None,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
) -> List[List[List[int]]]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    reader = _create_reader(model_root=model_root)
    variants = _load_image_variants(image_path)

    variant_rows: List[List[Dict[str, object]]] = []
    variant_labels: List[List[Dict[str, object]]] = []
    for variant in variants:
        detections_variant = _read_detections(reader, variant, numeric_only=False)
        row_candidates_variant = _row_candidates_from_detections(detections_variant)
        labels_variant = _array_labels_from_detections(detections_variant)
        if row_candidates_variant:
            variant_rows.append(row_candidates_variant)
            variant_labels.append(labels_variant)

    if variant_rows:
        label_anchor = max(variant_labels, key=lambda labels: len(labels)) if variant_labels else []
        merged_buckets: List[List[Dict[str, object]]] = []
        for row_candidates_variant in variant_rows:
            grouped = _group_row_candidates_by_labels(row_candidates_variant, label_anchor)
            for index, bucket in enumerate(grouped):
                while len(merged_buckets) <= index:
                    merged_buckets.append([])
                merged_buckets[index].extend(bucket)

        merged_arrays = [
            _collapse_candidate_bucket(
                bucket,
                expected_rows=expected_rows,
                expected_cols=expected_cols,
            )
            for bucket in merged_buckets
        ]
        merged_arrays = [array_block for array_block in merged_arrays if array_block]

        if expected_arrays and expected_arrays > 0:
            merged_arrays = merged_arrays[:expected_arrays]

        if merged_arrays and (expected_rows or expected_arrays):
            return merged_arrays

        if merged_arrays:
            return merged_arrays

    candidate_arrays: List[List[List[List[int]]]] = []

    detections = _best_detections(reader, variants[:1], numeric_only=False)

    row_candidates = _row_candidates_from_detections(detections)
    labels = _array_labels_from_detections(detections)
    arrays = _group_rows_by_labels(row_candidates, labels)
    arrays = [_normalize_array_block(block) for block in arrays]
    arrays = [block for block in arrays if block]

    if arrays:
        candidate_arrays.append(arrays)

    lines = extract_lines(image_path, model_root=model_root)
    arrays = _parse_arrays_from_lines(lines)
    arrays = [_normalize_array_block(block) for block in arrays]
    arrays = [block for block in arrays if block]

    if arrays:
        candidate_arrays.append(arrays)

    detections = _best_detections(reader, variants, numeric_only=True)

    tokens: List[Dict[str, float]] = []
    for detection in detections:
        tokens.extend(_number_tokens(detection))

    tokens = _filter_numeric_tokens(tokens)

    row_clusters = _cluster_lines(tokens)
    row_clusters = [sorted(row, key=lambda item: item["center_x"]) for row in row_clusters if row]
    if not row_clusters:
        return []

    columns = _infer_column_count(row_clusters, expected_cols=None)
    centers = _column_centers(row_clusters, columns)
    rebuilt = [_assign_row_to_columns(row, centers) for row in row_clusters]
    fallback_rows = [[value for value in row if value is not None] for row in rebuilt]
    fallback_rows = [row for row in fallback_rows if row]

    if fallback_rows:
        candidate_arrays.append([_normalize_array_block(fallback_rows)])

    if not candidate_arrays:
        return []

    return max(candidate_arrays, key=_score_arrays)


def extract_numeric_array(
    image_path: Path,
    expected_cols: Optional[int] = None,
    model_root: Optional[Path] = None,
) -> List[List[Optional[int]]]:
    arrays = extract_numeric_arrays(
        image_path,
        model_root=model_root,
        expected_cols=expected_cols,
    )
    if not arrays:
        return []

    first_array = arrays[0]
    if not expected_cols or expected_cols <= 0:
        return [list(row) for row in first_array]

    adjusted: List[List[Optional[int]]] = []
    for row in first_array:
        values = list(row[:expected_cols])
        if len(values) < expected_cols:
            values.extend([None] * (expected_cols - len(values)))
        adjusted.append(values)
    return adjusted


def _write_lines_csv(lines: List[str], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["line"])
        for line in lines:
            writer.writerow([line])


def _write_arrays_csv(array_data: List[List[List[int]]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for index, array_block in enumerate(array_data):
            if len(array_data) > 1:
                writer.writerow([f"array_{index + 1}"])
            for row in array_block:
                writer.writerow(row)
            if index < len(array_data) - 1:
                writer.writerow([])


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR line and array extraction")
    parser.add_argument("--image", default="pic1.png", help="Path to image file")
    parser.add_argument("--mode", choices=["lines", "array"], default="array")
    parser.add_argument("--cols", type=int, default=22, help="Expected number of array columns")
    parser.add_argument("--expected-arrays", type=int, default=3, help="Expected number of arrays")
    parser.add_argument("--expected-rows", type=int, default=10, help="Expected rows per array")
    parser.add_argument("--csv", default='out.csv', help="Optional output CSV path")
    parser.add_argument(
        "--model-root",
        default=None,
        help="Local PaddleOCR model root (contains PP-OCR model folders).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    model_root = Path(args.model_root).expanduser().resolve() if args.model_root else None
    if args.mode == "array":
        array_data = extract_numeric_arrays(
            image_path,
            model_root=model_root,
            expected_arrays=args.expected_arrays,
            expected_rows=args.expected_rows,
            expected_cols=args.cols,
        )
        if args.cols and args.cols > 0:
            constrained: List[List[List[Optional[int]]]] = []
            for array_block in array_data:
                constrained_block: List[List[Optional[int]]] = []
                for row in array_block:
                    values = list(row[: args.cols])
                    if len(values) < args.cols:
                        values.extend([None] * (args.cols - len(values)))
                    constrained_block.append(values)
                constrained.append(constrained_block)
            print(json.dumps(constrained, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(array_data, ensure_ascii=False, indent=2))
        if args.csv:
            _write_arrays_csv(array_data, Path(args.csv))
        return

    lines = extract_lines(image_path, model_root=model_root)
    print(json.dumps(lines, ensure_ascii=False, indent=2))
    if args.csv:
        _write_lines_csv(lines, Path(args.csv))


if __name__ == "__main__":
    main()

