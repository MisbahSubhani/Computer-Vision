# --- Lane Line Detection using Hough Transform ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests

# Download the image from the URL
image_url = 'https://www.shutterstock.com/image-photo/drone-picture-motorway-m2-facing-260nw-1819823105.jpg'
response = requests.get(image_url)
with open('image.jpg', 'wb') as f:
    f.write(response.content)

# Load image
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150)

# Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=5)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Display
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Lane Detection using Hough Transform")
plt.axis('off')
plt.show()
