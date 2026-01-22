

import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def detect_pupil(eye_frame):
    gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_eye, (11, 11), 0)
    _, thresholded = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            pupil_x = int(moments["m10"] / moments["m00"])
            pupil_y = int(moments["m01"] / moments["m00"])
            return pupil_x, pupil_y
    return None


video_file = '0_Normal_3.1.mp4'
cap = cv2.VideoCapture(video_file)


initial_pupil_x, initial_pupil_y = None, None
correct_gaze_frames = 0
total_gaze_frames = 0
accuracy = 0.0  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]


        eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(80, 80))

        for (ex, ey, ew, eh) in eyes:
            eye_region = roi_color[ey:ey + eh, ex:ex + ew]
            pupil_position = detect_pupil(eye_region)

            if pupil_position:
                pupil_x, pupil_y = pupil_position
                pupil_x += ex
                pupil_y += ey

                if initial_pupil_x is not None:
                    movement = abs(initial_pupil_x - pupil_x) + abs(initial_pupil_y - pupil_y)
                    if movement > 5:
                        accuracy = 1  
                    else:
                        accuracy = 0  

                
                    correct_gaze_frames += accuracy
                    total_gaze_frames += 1

                initial_pupil_x, initial_pupil_y = pupil_x, pupil_y

                cv2.circle(frame, (x + pupil_x, y + pupil_y), 5, (0, 0, 255), -1)

    cv2.imshow('Pupil Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if total_gaze_frames > 0:
    accuracy = correct_gaze_frames / total_gaze_frames

print(f'Gaze Tracking Accuracy: {accuracy * 100:.2f}%')
##code to calculate accuracy. when the pupil tracker is a certain distance awat from the pupil , it gives a score 0 and 1 if it is within the pupils.
#the total is cumulated to find accuracy.