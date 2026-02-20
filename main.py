import cv2
import easyocr
import re
import numpy as np

# -----------------------------
IMAGE_PATH = "image.png"
NUM_COLUMNS = 22
PREPROCESS_THRESHOLD = True
# -----------------------------

# 1️⃣ Load image
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if PREPROCESS_THRESHOLD:
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

cv2.imwrite("preprocessed.png", img)

# 2️⃣ Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext("preprocessed.png")

# 3️⃣ Extract numeric boxes and fix OCR mistakes
def fix_ocr_number(s):
    s = s.replace('O', '0').replace('o', '0')
    s = s.replace('l', '1').replace('I', '1')
    return s

numeric_boxes = []
for bbox, text, conf in results:
    text = fix_ocr_number(text)
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        x = bbox[0][0]  # top-left X
        y = bbox[0][1]  # top-left Y
        for n in nums:
            numeric_boxes.append((x, y, n))

# 4️⃣ Sort by Y to start row grouping
numeric_boxes.sort(key=lambda b: b[1])

# 5️⃣ Group numbers into rows by Y proximity
rows = []
current_row = []
row_y = None
y_threshold = 10  # pixels; adjust slightly if needed

for x, y, n in numeric_boxes:
    if row_y is None:
        row_y = y
        current_row.append((x, n))
    elif abs(y - row_y) <= y_threshold:
        current_row.append((x, n))
    else:
        # sort current row by X
        current_row.sort(key=lambda r: r[0])
        # extract numbers only
        row_numbers = [r[1] for r in current_row]
        # pad if less than NUM_COLUMNS
        while len(row_numbers) < NUM_COLUMNS:
            row_numbers.append('0')
        rows.append(row_numbers)
        # start new row
        current_row = [(x, n)]
        row_y = y

# last row
if current_row:
    current_row.sort(key=lambda r: r[0])
    row_numbers = [r[1] for r in current_row]
    while len(row_numbers) < NUM_COLUMNS:
        row_numbers.append('0')
    rows.append(row_numbers)

# 6️⃣ Convert to NumPy array
matrix = np.array(rows, dtype=float)

print("Matrix shape:", matrix.shape)
print(matrix)

# Optional: save CSV
np.savetxt("extracted_matrix.csv", matrix, delimiter=",", fmt="%.2f")
