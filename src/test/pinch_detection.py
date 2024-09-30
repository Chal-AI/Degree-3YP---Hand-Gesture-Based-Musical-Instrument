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
                thumb_x = hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_TIP].x
                thumb_y = hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_TIP].y
                index_x = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                index_y = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                distance_1 = ((thumb_x - index_x) ** 2 + 
                            (thumb_y - index_y) ** 2) ** 0.5

                pinch_threshold = 0.05

                if distance_1 < pinch_threshold:
                    cv2.putText(image, "Pinch detected", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,
                                255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "No pinch detected", (50,
                                50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow('Hand Gesture Control', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
