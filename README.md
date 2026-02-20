# test
## this is branch1


import cv2
import easyocr
import re
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
IMAGE_PATH = "image.png"  # path to your screenshot
NUM_COLUMNS = 22          # guaranteed number of columns
PREPROCESS_THRESHOLD = True  # whether to apply thresholding

# -----------------------------
# 1️⃣ Load and preprocess image
# -----------------------------
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if PREPROCESS_THRESHOLD:
    # Adaptive thresholding improves OCR on screenshots
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

# Optional: save preprocessed image to check
cv2.imwrite("preprocessed.png", img)

# -----------------------------
# 2️⃣ Initialize EasyOCR reader (offline)
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

# Run OCR
results = reader.readtext("preprocessed.png")

# -----------------------------
# 3️⃣ Extract numeric values and fix OCR mistakes
# -----------------------------
def fix_ocr_number(s):
    """Fix common OCR misreads in numeric data."""
    s = s.replace('O', '0').replace('o', '0')
    s = s.replace('l', '1').replace('I', '1')
    s = s.replace(',', '')  # remove commas if misread inside number
    return s

numeric_boxes = []

for bbox, text, conf in results:
    # fix OCR text
    text = fix_ocr_number(text)
    # extract numeric values
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        x = bbox[0][0]  # top-left x
        y = bbox[0][1]  # top-left y
        for n in nums:
            numeric_boxes.append((x, y, n))

# -----------------------------
# 4️⃣ Sort boxes (top-to-bottom, left-to-right)
# -----------------------------
numeric_boxes.sort(key=lambda b: (b[1], b[0]))

# -----------------------------
# 5️⃣ Reconstruct rows with NUM_COLUMNS
# -----------------------------
rows = []
buffer = []

for x, y, n in numeric_boxes:
    buffer.append(n)
    if len(buffer) == NUM_COLUMNS:
        rows.append(buffer)
        buffer = []

# Optional: handle leftover numbers if OCR splits row incorrectly
if buffer:
    # pad missing values with 0
    while len(buffer) < NUM_COLUMNS:
        buffer.append('0')
    rows.append(buffer)

# -----------------------------
# 6️⃣ Convert to NumPy array
# -----------------------------
matrix = np.array(rows, dtype=float)

# -----------------------------
# 7️⃣ Output
# -----------------------------
print("Extracted matrix shape:", matrix.shape)
print(matrix)

# Optional: save to CSV
np.savetxt("extracted_matrix.csv", matrix, delimiter=",", fmt="%.2f")
print("Saved to extracted_matrix.csv")
