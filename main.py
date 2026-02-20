import easyocr
import re
import numpy as np
import csv

# -----------------------------
IMAGE_PATH = "image.png"
CSV_PATH = "extracted_matrix.csv"
MIN_X_GAP = 5   # horizontal gap to merge partial numbers
# -----------------------------

# 1️⃣ Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(IMAGE_PATH)

# 2️⃣ Extract bounding box centroids and numbers
boxes = []
heights = []
for bbox, text, conf in results:
    # Fix common OCR misreads
    text = text.replace('O','0').replace('o','0').replace('l','1').replace('I','1')
    if re.search(r'\d', text):  # keep only boxes with numbers
        y_min = min(pt[1] for pt in bbox)
        y_max = max(pt[1] for pt in bbox)
        centroid_y = (y_min + y_max) / 2
        x_min = min(pt[0] for pt in bbox)
        box_height = y_max - y_min
        heights.append(box_height)
        boxes.append((x_min, centroid_y, text))

if not boxes:
    print("No numbers detected!")
    exit()

# 3️⃣ Compute adaptive Y threshold
median_height = np.median(heights)
Y_THRESHOLD = max(5, median_height * 0.8)  # adaptive row height

# 4️⃣ Sort by centroid Y then X
boxes.sort(key=lambda b: (b[1], b[0]))

# 5️⃣ Initial grouping by centroid Y
rows = []
current_row = []
row_y = None

for x, y, text in boxes:
    if row_y is None:
        row_y = y
        current_row.append((x, text))
    elif abs(y - row_y) <= Y_THRESHOLD:
        current_row.append((x, text))
    else:
        rows.append(current_row)
        current_row = [(x, text)]
        row_y = y
if current_row:
    rows.append(current_row)

# 6️⃣ Soft merge adjacent rows if they are too close vertically
merged_rows = []
i = 0
while i < len(rows):
    row = rows[i]
    y_values = [b[1] for b in row]
    row_centroid = np.mean(y_values)
    # merge next row if centroids are close
    j = i + 1
    while j < len(rows):
        next_row = rows[j]
        next_centroid = np.mean([b[1] for b in next_row])
        if next_centroid - row_centroid <= Y_THRESHOLD:
            row += next_row
            row_centroid = np.mean([b[1] for _, y, _ in row])
            j += 1
        else:
            break
    merged_rows.append(row)
    i = j

# 7️⃣ Merge partial numbers horizontally and sort X
final_rows = []
for row in merged_rows:
    row.sort(key=lambda r: r[0])  # left to right
    numbers = []
    buffer = ''
    last_x = None
    for x, text in row:
        if last_x is not None and x - last_x <= MIN_X_GAP:
            buffer += text  # merge partial number
        else:
            if buffer:
                # extract numbers from previous buffer
                nums = re.findall(r'\d+', buffer)
                numbers.extend([int(n) for n in nums])
            buffer = text
        last_x = x
    # flush last buffer
    if buffer:
        nums = re.findall(r'\d+', buffer)
        numbers.extend([int(n) for n in nums])
    final_rows.append(numbers)

# 8️⃣ Save as CSV (integers)
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    for row in final_rows:
        writer.writerow(row)

print(f"Saved {len(final_rows)} rows to {CSV_PATH}")
