import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR


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
    return [image, gray_bgr, upscaled]


def _create_reader() -> PaddleOCR:
    return PaddleOCR(
        lang="en",
        device="cpu",
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


def extract_lines(image_path: Path) -> List[str]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    reader = _create_reader()
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
        matches = re.finditer(r"\(?\s*-?\d+(?:\s*,\s*-?\d+){3,}\s*\)?", text)
        ys = [p[1] for p in points]
        xs = [p[0] for p in points]
        center_y = float(sum(ys) / max(1, len(ys)))
        left = float(min(xs)) if xs else 0.0

        for match in matches:
            row = _parse_row_text(match.group(0))
            if len(row) >= 4:
                candidates.append(
                    {
                        "row": row,
                        "center_y": center_y,
                        "left": left,
                    }
                )

    candidates.sort(key=lambda item: (float(item["center_y"]), float(item["left"])))
    unique: List[Dict[str, object]] = []
    seen = set()
    for item in candidates:
        key = tuple(item["row"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


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


def extract_numeric_arrays(image_path: Path) -> List[List[List[int]]]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    reader = _create_reader()
    variants = _load_image_variants(image_path)
    detections = _best_detections(reader, variants[:1], numeric_only=False)

    row_candidates = _row_candidates_from_detections(detections)
    labels = _array_labels_from_detections(detections)
    arrays = _group_rows_by_labels(row_candidates, labels)

    if arrays:
        return arrays

    lines = extract_lines(image_path)
    arrays = _parse_arrays_from_lines(lines)
    if arrays:
        return arrays

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
    return [fallback_rows] if fallback_rows else []


def extract_numeric_array(image_path: Path, expected_cols: Optional[int] = None) -> List[List[Optional[int]]]:
    arrays = extract_numeric_arrays(image_path)
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
    parser.add_argument("--image", default="pic.jpeg", help="Path to image file")
    parser.add_argument("--mode", choices=["lines", "array"], default="lines")
    parser.add_argument("--cols", type=int, default=None, help="Expected number of array columns")
    parser.add_argument("--csv", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if args.mode == "array":
        array_data = extract_numeric_arrays(image_path)
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

    lines = extract_lines(image_path)
    print(json.dumps(lines, ensure_ascii=False, indent=2))
    if args.csv:
        _write_lines_csv(lines, Path(args.csv))


if __name__ == "__main__":
    main()
