import cv2
import numpy as np

# Create a VideoCapture object to read from the video file
cap = cv2.VideoCapture(r'C:\Users\rrakh\Downloads\road5.mp4')  # Replace 'your_video.mp4' with your video file path

# Define the Region of Interest (ROI) polygon
roi_vertices = [(580, 550), (750, 550), (410, 700), (1000, 700)] # Adjust the coordinates as needed
roi_mask = np.zeros((720, 1280), dtype=np.uint8)
cv2.fillPoly(roi_mask, [np.array(roi_vertices, np.int32)], 255)

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (1280, 720))  # Adjust the frame rate as needed

frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_counter += 1

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 75, 0])
    decrease_value = 30
    upper_white = np.array([255 - decrease_value, 255 - decrease_value, 255 - decrease_value])
    mask = cv2.inRange(frame, lower_white, upper_white)
    hls_result = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)

    roi_canny = cv2.bitwise_and(canny, roi_mask)

    lines = cv2.HoughLinesP(roi_canny, rho=1, theta=np.pi / 180, threshold=50, minLineLength=150, maxLineGap=200)

    # Draw the detected lines on the frame
    line_image = np.copy(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for _ in range(4):
        out.write(line_image)

    # Display the frame with detected lines
    cv2.imshow('Lane Detection', line_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
