import easyocr
import re
import numpy as np
import csv

# -----------------------------
IMAGE_PATH = "image.png"
CSV_PATH = "extracted_matrix.csv"
# -----------------------------

# 1️⃣ Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(IMAGE_PATH)

# 2️⃣ Extract bounding box centroids and numbers
boxes = []
heights = []  # for adaptive Y threshold
for bbox, text, conf in results:
    # Fix common OCR misreads
    text = text.replace('O','0').replace('o','0').replace('l','1').replace('I','1')
    nums = re.findall(r'\d+', text)  # only positive integers
    if nums:
        y_min = min(pt[1] for pt in bbox)
        y_max = max(pt[1] for pt in bbox)
        centroid_y = (y_min + y_max) / 2
        x_min = min(pt[0] for pt in bbox)
        box_height = y_max - y_min
        heights.append(box_height)
        for n in nums:
            boxes.append((x_min, centroid_y, int(n)))

if not boxes:
    print("No numbers detected!")
    exit()

# 3️⃣ Compute adaptive Y threshold
if heights:
    median_height = np.median(heights)
    Y_THRESHOLD = max(5, median_height * 0.8)  # 80% of median box height
else:
    Y_THRESHOLD = 10  # fallback

# 4️⃣ Sort by centroid Y then X
boxes.sort(key=lambda b: (b[1], b[0]))

# 5️⃣ Group boxes into rows by centroid Y (adaptive)
rows = []
current_row = []
row_y = None

for x, y, n in boxes:
    if row_y is None:
        row_y = y
        current_row.append((x, n))
    elif abs(y - row_y) <= Y_THRESHOLD:
        current_row.append((x, n))
    else:
        current_row.sort(key=lambda r: r[0])
        row_nums = [num for _, num in current_row]
        rows.append(row_nums)
        current_row = [(x, n)]
        row_y = y

# last row
if current_row:
    current_row.sort(key=lambda r: r[0])
    row_nums = [num for _, num in current_row]
    rows.append(row_nums)

# 6️⃣ Save as CSV (integers)
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)  # each number as int

print(f"Saved {len(rows)} rows to {CSV_PATH}")
