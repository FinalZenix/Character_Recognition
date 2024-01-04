import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Start the camera
cap = cv2.VideoCapture(0)
desired_width = 320 / 2
desired_height = 240 / 2
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Define the color ranges for red, yellow, and green in HSV color space
color_ranges = {
    'red': [(np.array([0, 100, 100]), np.array([10, 255, 255])), (np.array([160, 100, 100]), np.array([179, 255, 255]))],
    'yellow': [(np.array([15, 100, 100]), np.array([35, 255, 255]))],
    'green': [(np.array([36, 100, 100]), np.array([70, 255, 255]))]
}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color, ranges in color_ranges.items():
        # Create a mask for each color range and find contours in the mask
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes around the detected objects
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = w * h

                # Check if the aspect ratio is close to 1 for square or not equal to 1 for rectangle
                # and if the area is above a certain threshold
                if 0.9 <= aspect_ratio <= 1.1 and area > 1000:  # Adjust the area threshold as needed
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert to grayscale, Otsu's threshold, invert
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = 255 - thresh

    # OCR with whitelist of characters
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=HSU'
    data = pytesseract.image_to_string(invert, config=custom_config)

    # Get bounding box estimates
    boxes = pytesseract.image_to_boxes(invert)
    h, w, _ = frame.shape
    for b in boxes.splitlines():
        b = b.split(' ')
        frame = cv2.rectangle(frame, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
        cv2.putText(frame, b[0], (int(b[1]), h - int(b[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Press "q" to quit', frame)

    # If 'q' is pressed, break from the loop
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
