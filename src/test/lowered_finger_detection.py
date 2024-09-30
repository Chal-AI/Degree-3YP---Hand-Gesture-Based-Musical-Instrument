import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), 
                             cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_x = hand_landmarks.landmark[
                    mp_hands.HandLandmark.WRIST].x
                wrist_y = hand_landmarks.landmark[
                    mp_hands.HandLandmark.WRIST].y
                thumb_x = hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_TIP].x
                thumb_y = hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_TIP].y
                index_x = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                index_y = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                middle_x = hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                middle_y = hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                ring_x = hand_landmarks.landmark[
                    mp_hands.HandLandmark.RING_FINGER_TIP].x
                ring_y = hand_landmarks.landmark[
                    mp_hands.HandLandmark.RING_FINGER_TIP].y
                pinky_x = hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_TIP].x
                pinky_y = hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_TIP].y

                distance_1 = ((wrist_x - thumb_x) ** 2+(wrist_y- 
                                thumb_y) ** 2) ** 0.5
                distance_2 = ((wrist_x - index_x) ** 2+(wrist_y- 
                                index_y) ** 2) ** 0.5
                distance_3 = ((wrist_x - middle_x) ** 2+(wrist_y- 
                                middle_y) ** 2) ** 0.5
                distance_4 = ((wrist_x - ring_x) ** 2+(wrist_y- 
                                ring_y) ** 2) ** 0.5
                distance_5 = ((wrist_x - pinky_x) ** 2+(wrist_y- 
                                pinky_y) ** 2) ** 0.5

                threshold1 = 0.25 #Thumb
                threshold2 = 0.35 #Index
                threshold3 = 0.35 #Middle
                threshold4 = 0.35 #Ring
                threshold5 = 0.35 #Pinky

                if distance_1 < threshold1:
                    cv2.putText(image,"Thumb Lowered",(50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,
                                0),2,cv2.LINE_AA)                    
                if distance_2 < threshold2:
                    cv2.putText(image,"Index Lowered",(50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,
                                0),2,cv2.LINE_AA)
                if distance_3 < threshold3:
                    cv2.putText(image,"Middle Lowered",(50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,
                                0),2,cv2.LINE_AA)
                if distance_4 < threshold4:
                    cv2.putText(image,"Ring Lowered",(50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,
                                0),2,cv2.LINE_AA)
                if distance_5 < threshold5:
                    cv2.putText(image,"Pinky Lowered",(50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,
                                0),2, cv2.LINE_AA)
                    

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow('Lowering of Finger Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
