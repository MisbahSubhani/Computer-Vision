# --- 3D Vision: Depth Map using Stereo Images ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# URLs of the sample stereo pair (left and right images)
left_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeL.jpg'
right_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeR.jpg'

# Download the images
left_file = 'aloeL.jpg'
right_file = 'aloeR.jpg'
urllib.request.urlretrieve(left_url, left_file)
urllib.request.urlretrieve(right_url, right_file)


# Load sample stereo pair (left and right images)
left_img = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE)

# Check if images loaded
if left_img is None or right_img is None:
    raise Exception("Error loading images!")

# Create StereoBM object
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# Compute disparity map
disparity = stereo.compute(left_img, right_img)

# Normalize the disparity for better visualization
disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(left_img, cmap='gray')
plt.title('Left Image')

plt.subplot(1,2,2)
plt.imshow(disparity_norm, cmap='plasma')
plt.title('Depth Map (Disparity)')
plt.colorbar(label='Depth')
plt.show()
