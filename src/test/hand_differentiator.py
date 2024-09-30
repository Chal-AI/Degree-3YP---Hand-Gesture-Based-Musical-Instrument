import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

while cap.isOpened():
    ret, image = cap.read()

    if not ret:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness_info in zip(
            results.multi_hand_landmarks, 
            results.multi_handedness):
            handedness_str = handedness_info.classification[
                                                       0].label
            hand_side = handedness_str

            cv2.putText(image, f"{hand_side} Hand", (10,
                30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Side Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
