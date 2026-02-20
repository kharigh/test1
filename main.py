import cv2
import easyocr
import re
import numpy as np

# -----------------------------
IMAGE_PATH = "image.png"
NUM_COLUMNS = 22
PREPROCESS_THRESHOLD = True
Y_THRESHOLD = 10  # pixels for merging boxes into a row
# -----------------------------

# 1️⃣ Load and preprocess
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if PREPROCESS_THRESHOLD:
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

# 2️⃣ Initialize EasyOCR (offline)
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext("image.png")

# 3️⃣ Fix OCR common mistakes
def fix_ocr_number(s):
    s = s.replace('O', '0').replace('o', '0')
    s = s.replace('l', '1').replace('I', '1')
    return s

# 4️⃣ Merge boxes into rows based on Y coordinate
boxes = []
for bbox, text, conf in results:
    text = fix_ocr_number(text)
    x = bbox[0][0]
    y = bbox[0][1]
    boxes.append((x, y, text))

# Sort by Y
boxes.sort(key=lambda b: b[1])

merged_lines = []
current_row = []
row_y = None

for x, y, text in boxes:
    if row_y is None:
        row_y = y
        current_row.append((x, text))
    elif abs(y - row_y) <= Y_THRESHOLD:
        current_row.append((x, text))
    else:
        # Sort current row by X
        current_row.sort(key=lambda r: r[0])
        # Merge text
        merged_text = " ".join([t for _, t in current_row])
        merged_lines.append(merged_text)
        # Start new row
        current_row = [(x, text)]
        row_y = y

# Last row
if current_row:
    current_row.sort(key=lambda r: r[0])
    merged_text = " ".join([t for _, t in current_row])
    merged_lines.append(merged_text)

# 5️⃣ Filter merged lines: only numbers, comma, brackets
numeric_lines = []
for line in merged_lines:
    if re.fullmatch(r'[\d\[\]\s\-,.]+', line):
        numeric_lines.append(line)

# 6️⃣ Extract exactly 22 numbers per row
rows = []
for line in numeric_lines:
    line_clean = line.replace('[','').replace(']','')
    nums = re.findall(r'-?\d+\.?\d*', line_clean)
    if len(nums) == NUM_COLUMNS:
        rows.append(nums)

# 7️⃣ Convert to NumPy array
if rows:
    matrix = np.array(rows, dtype=float)
    print("Matrix shape:", matrix.shape)
    print(matrix)
    # Save CSV
    np.savetxt("extracted_matrix.csv", matrix, delimiter=",", fmt="%.2f")
else:
    print("No valid numeric array rows found!")
