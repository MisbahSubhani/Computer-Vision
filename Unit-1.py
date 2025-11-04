import cv2
from google.colab.patches import cv2_imshow
from google.colab import files
import numpy as np

# File upload from local system
uploaded = files.upload()

# Get the file name dynamically
for file_name in uploaded.keys():
    img_path = file_name

# Read the uploaded image
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filtering
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Thresholding (Otsu)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Edge Detection using Canny
edges = cv2.Canny(blur, 100, 200)

# Show Results
print("Original Image:")
cv2_imshow(img)
print("Thresholded Image (Otsu):")
cv2_imshow(thresh)
print("Edge Detection (Canny):")
cv2_imshow(edges)
