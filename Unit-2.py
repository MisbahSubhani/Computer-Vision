import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload image from system
uploaded = files.upload()

# Selecting uploaded file
for file_name in uploaded.keys():
    img_path = file_name

# Read uploaded image
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binarization (Otsu thresholding)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Connected components
count, labels = cv2.connectedComponents(binary)

# Remove background count
print("Total objects detected:", count - 1)

# Show output
print("Binary Image (after thresholding):")
cv2_imshow(binary)
