# --- Foreground–Background Separation using Background Subtraction ---
import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import cv2_imshow


video_url = "https://www.pexels.com/download/video/3105196/"
cap = cv2.VideoCapture(video_url)

# Use the uploaded video file
# cap = cv2.VideoCapture(uploaded_video_filename)


# Create background subtractor (Mixture of Gaussians method)
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Remove noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours of moving objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter small movements/noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2_imshow(frame) # Use cv2_imshow
    cv2_imshow(fgmask) # Use cv2_imshow

    # The waitKey and destroyAllWindows are not typically used in Colab with cv2_imshow
    # if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit
    #     break


cap.release()
# cv2.destroyAllWindows() # Not needed with cv2_imshow
