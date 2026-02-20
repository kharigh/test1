from paddleocr import PaddleOCR
import re
import numpy as np
import csv

# -----------------------------
IMAGE_PATH = "image.png"
CSV_PATH = "extracted_matrix.csv"
MODEL_DIR = "./paddle_models"  # local model directory
Y_THRESHOLD_FACTOR = 0.6       # how tight to group rows
# -----------------------------

# 1️⃣ Initialize PaddleOCR using local model dirs
ocr = PaddleOCR(
    det_model_dir=f"{MODEL_DIR}/PP-OCRv5_server_det",
    rec_model_dir=f"{MODEL_DIR}/PP-OCRv5_server_rec",
    cls_model_dir=f"{MODEL_DIR}/PP-OCRv5_server_cls",
    use_angle_cls=True,
    lang="en"
)

# 2️⃣ Run OCR
result = ocr.ocr(IMAGE_PATH, cls=True)

# 3️⃣ Collect number boxes with centroid
boxes = []
heights = []
for line in result:
    # Each `line` has format: [bbox, (text, conf)]
    bbox, (text, score) = line
    # Fix common OCR misreads
    text = text.replace('O', '0').replace('o', '0')
    text = text.replace('l', '1').replace('I', '1')

    # Only keep if contains digit
    if re.search(r"\d", text):
        # get box centroid Y
        coords = np.array(bbox)
        y_centroid = (coords[:,1].min() + coords[:,1].max()) / 2
        x_min = coords[:,0].min()
        box_height = coords[:,1].max() - coords[:,1].min()
        heights.append(box_height)

        # extract one or more numbers
        nums = re.findall(r'\d+', text)
        for n in nums:
            boxes.append((x_min, y_centroid, int(n)))

if not boxes:
    print("No numbers detected!")
    exit()

# 4️⃣ Adaptive row grouping
median_h = np.median(heights) if heights else 10
y_thresh = max(5, median_h * Y_THRESHOLD_FACTOR)

boxes.sort(key=lambda b: (b[1], b[0]))

rows = []
current = []
row_y = None

for x, y, n in boxes:
    if row_y is None:
        row_y = y
        current.append((x, n))
    elif abs(y - row_y) <= y_thresh:
        current.append((x, n))
    else:
        current.sort(key=lambda r: r[0])
        rows.append([num for _, num in current])
        current = [(x, n)]
        row_y = y

if current:
    current.sort(key=lambda r: r[0])
    rows.append([num for _, num in current])

# 5️⃣ Save rows to CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    for r in rows:
        writer.writerow(r)

print(f"Saved {len(rows)} rows to {CSV_PATH}")
