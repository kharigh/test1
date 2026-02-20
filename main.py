import cv2
import easyocr
import re
import numpy as np

# -----------------------------
IMAGE_PATH = "image.png"
NUM_COLUMNS = 22
PREPROCESS_THRESHOLD = True
Y_THRESHOLD = 10  # pixels to merge boxes into the same row
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

# 3️⃣ Fix OCR mistakes and keep only boxes with digits
boxes = []
for bbox, text, conf in results:
    # correct common OCR misreads
    text = text.replace('O','0').replace('o','0').replace('l','1').replace('I','1')
    # Only keep boxes containing digits
    if re.search(r'\d', text):
        x = bbox[0][0]
        y = bbox[0][1]
        boxes.append((x, y, text))

if not boxes:
    print("No numeric boxes detected!")
    exit()

# 4️⃣ Sort boxes by Y coordinate
boxes.sort(key=lambda b: b[1])

# 5️⃣ Merge boxes into rows by Y proximity
merged_rows = []
current_row = []
row_y = None

for x, y, text in boxes:
    if row_y is None:
        row_y = y
        current_row.append((x, text))
    elif abs(y - row_y) <= Y_THRESHOLD:
        current_row.append((x, text))
    else:
        # Sort by X
        current_row.sort(key=lambda r: r[0])
        merged_rows.append(" ".join([t for _, t in current_row]))
        # Start new row
        current_row = [(x, text)]
        row_y = y

# last row
if current_row:
    current_row.sort(key=lambda r: r[0])
    merged_rows.append(" ".join([t for _, t in current_row]))

# 6️⃣ Extract numbers from each merged row
rows = []
for line in merged_rows:
    nums = re.findall(r'\d+', line)  # only positive integers
    if len(nums) == NUM_COLUMNS:
        rows.append(nums)

# 7️⃣ Convert to NumPy array
if rows:
    matrix = np.array(rows, dtype=int)
    print("Matrix shape:", matrix.shape)
    print(matrix)
    np.savetxt("extracted_matrix.csv", matrix, delimiter=",", fmt="%d")
else:
    print("No valid 22-number rows found!")
