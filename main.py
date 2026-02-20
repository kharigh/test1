import cv2
import easyocr
import re
import numpy as np

# -----------------------------
IMAGE_PATH = "image.png"
NUM_COLUMNS = 22
PREPROCESS_THRESHOLD = True
# -----------------------------

# 1️⃣ Load and preprocess image
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if PREPROCESS_THRESHOLD:
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

cv2.imwrite("preprocessed.png", img)  # optional

# 2️⃣ Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext("preprocessed.png")

# 3️⃣ Function to fix OCR mistakes
def fix_ocr_number(s):
    s = s.replace('O', '0').replace('o', '0')
    s = s.replace('l', '1').replace('I', '1')
    return s

# 4️⃣ Filter lines: only keep lines with numbers, commas, brackets
filtered_lines = []

for bbox, text, conf in results:
    text = fix_ocr_number(text)
    # Check if line contains only digits, comma, bracket, dot, minus, spaces
    if re.fullmatch(r'[\d\[\]\s\-,.]+', text):
        filtered_lines.append(text)

# 5️⃣ Extract numeric values
rows = []

for line in filtered_lines:
    # Remove brackets
    line_clean = line.replace('[','').replace(']','')
    # Split by comma
    nums = re.findall(r'-?\d+\.?\d*', line_clean)
    if len(nums) == NUM_COLUMNS:
        rows.append(nums)  # only accept exact 22 numbers

# 6️⃣ Convert to NumPy array
if rows:
    matrix = np.array(rows, dtype=float)
    print("Matrix shape:", matrix.shape)
    print(matrix)
    # Optional: save CSV
    np.savetxt("extracted_matrix.csv", matrix, delimiter=",", fmt="%.2f")
else:
    print("No valid numeric array rows found!")
