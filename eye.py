import cv2
import numpy as np

cap = cv2.VideoCapture("0_Normal_3.1.mp4")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def find_eye_center(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    if frame_number % int(cap.get(cv2.CAP_PROP_FPS) * 2.5) == 0:
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
           
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

          
            left_ref_x = x + w // 4
            left_ref_y = y + h // 4 + h // 10
            right_ref_x = x + 3 * w // 4
            right_ref_y = y + h // 4 + h // 10

          
            eye_region = frame[y:y + h, x:x + w]

         
            left_eye_region = eye_region[10:10 + h // 2, 10:10 + w // 2]
            right_eye_region = eye_region[10:10 + h // 2, w // 2 + 10:w - 10]

            left_eye_center = find_eye_center(left_eye_region)
            right_eye_center = find_eye_center(right_eye_region)

            if left_eye_center:
            
                left_eye_center_x = x + 10 + left_eye_center[0]
                left_eye_center_y = y + 10 + left_eye_center[1]

              
                if left_eye_center_x < left_ref_x:
                    eye_position = "Left"
                elif left_eye_center_x > left_ref_x:
                    eye_position = "Right"
                else:
                    eye_position = "Center"

                print(f"Left Eye Position: {eye_position}")

             
                cv2.circle(frame, (left_eye_center_x, left_eye_center_y), 2, (0, 0, 255), -1)
                cv2.circle(frame, (left_ref_x, left_ref_y), 2, (0, 255, 0), -1)

            if right_eye_center:
               
                right_eye_center_x = x + w // 2 + 10 + right_eye_center[0]
                right_eye_center_y = y + 10 + right_eye_center[1]

               
                if right_eye_center_x < right_ref_x:
                    eye_position = "Left"
                elif right_eye_center_x > right_ref_x:
                    eye_position = "Right"
                else:
                    eye_position = "Center"

                print(f"Right Eye Position: {eye_position}")

              
                cv2.circle(frame, (right_eye_center_x, right_eye_center_y), 2, (0, 0, 255), -1)
                cv2.circle(frame, (right_ref_x, right_ref_y), 2, (0, 255, 0), -1)

       
        cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
