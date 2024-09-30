# Putting differentiator_3, enhanced_HTN2, and hand_to_chord2 together
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe components for drawing and hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the hand tracking module
hands = mp_hands.Hands()

# Open the default camera
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        # Read a frame from the camera
        ret, image = cap.read()

        # Check if the video was captured successfully
        if not ret:
            break  # Exit the loop if there's an issue with capturing the video

        # Convert the video to RGB color (MediaPipe uses RGB)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Process the frame with the hand tracking module
        results = hands.process(image)

        # Convert the image back to BGR format for displaying on the camera (OpenCV)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if hand landmarks are detected in the frame
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Check handedness
                hand_side = handedness_info.classification[0].label

                # Map the thumb position to a chord or note based on hand side
                if hand_side == 'Left':
                # Extract finger knuckle landmarks and wrist landmark
                    knuckle_landmarks = [hand_landmarks.landmark[idx] for idx in [mp_hands.HandLandmark.THUMB_MCP,
                                                                                   mp_hands.HandLandmark.INDEX_FINGER_MCP,
                                                                                   mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                                                                                   mp_hands.HandLandmark.RING_FINGER_MCP,
                                                                                   mp_hands.HandLandmark.PINKY_MCP]]
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                    # Convert landmarks to pixel coordinates
                    knuckle_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in knuckle_landmarks]
                    wrist_point = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))

                    # Calculate the bounding rectangle around all hand landmarks
                    x, y, w, h = cv2.boundingRect(np.array(knuckle_points + [wrist_point]))

                    # Draw the bounding rectangle around knuckle landmarks and wrist
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Check if the thumb fingertip enters the bounding rectangle
                    thumb_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumb_fingertip_point = (int(thumb_fingertip.x * image.shape[1]), int(thumb_fingertip.y * image.shape[0]))
                    if x < thumb_fingertip_point[0] < x + w and y < thumb_fingertip_point[1] < y + h:
                        thumb_finger_entered = True
                    else:
                        thumb_finger_entered = False

                    # Check if the index fingertip enters the bounding rectangle
                    index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_fingertip_point = (int(index_fingertip.x * image.shape[1]), int(index_fingertip.y * image.shape[0]))
                    if x < index_fingertip_point[0] < x + w and y < index_fingertip_point[1] < y + h:
                        index_finger_entered = True
                    else:
                        index_finger_entered = False

                    # Check if the middle fingertip enters the bounding rectangle
                    middle_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    middle_fingertip_point = (int(middle_fingertip.x * image.shape[1]), int(middle_fingertip.y * image.shape[0]))
                    if x < middle_fingertip_point[0] < x + w and y < middle_fingertip_point[1] < y + h:
                        middle_finger_entered = True
                    else:
                        middle_finger_entered = False

                    # Check if the ring fingertip enters the bounding rectangle
                    ring_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    ring_fingertip_point = (int(ring_fingertip.x * image.shape[1]), int(ring_fingertip.y * image.shape[0]))
                    if x < ring_fingertip_point[0] < x + w and y < ring_fingertip_point[1] < y + h:
                        ring_finger_entered = True
                    else:
                        ring_finger_entered = False

                    # Check if the pinky fingertip enters the bounding rectangle
                    pinky_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    pinky_fingertip_point = (int(pinky_fingertip.x * image.shape[1]), int(pinky_fingertip.y * image.shape[0]))
                    if x < pinky_fingertip_point[0] < x + w and y < pinky_fingertip_point[1] < y + h:
                        pinky_finger_entered = True
                    else:
                        pinky_finger_entered = False


                    # Display fingertip entry status in the frame for each finger
                    if thumb_finger_entered:
                        cv2.putText(image, 'TL', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if index_finger_entered:
                        cv2.putText(image, 'IL', (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if middle_finger_entered:
                        cv2.putText(image, 'ML', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if ring_finger_entered:
                        cv2.putText(image, 'RL', (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if pinky_finger_entered:
                        cv2.putText(image, 'PL', (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


                elif hand_side == 'Right':
                # Extract finger knuckle landmarks and wrist landmark
                    knuckle_landmarks = [hand_landmarks.landmark[idx] for idx in [mp_hands.HandLandmark.THUMB_MCP,
                                                                                   mp_hands.HandLandmark.INDEX_FINGER_MCP,
                                                                                   mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                                                                                   mp_hands.HandLandmark.RING_FINGER_MCP,
                                                                                   mp_hands.HandLandmark.PINKY_MCP]]
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                    # Convert landmarks to pixel coordinates
                    knuckle_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in knuckle_landmarks]
                    wrist_point = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))

                    # Calculate the bounding rectangle around all hand landmarks
                    x, y, w, h = cv2.boundingRect(np.array(knuckle_points + [wrist_point]))

                    # Draw the bounding rectangle around knuckle landmarks and wrist
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Check if the thumb fingertip enters the bounding rectangle
                    thumb_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumb_fingertip_point = (int(thumb_fingertip.x * image.shape[1]), int(thumb_fingertip.y * image.shape[0]))
                    if x < thumb_fingertip_point[0] < x + w and y < thumb_fingertip_point[1] < y + h:
                        thumb_finger_entered = True
                    else:
                        thumb_finger_entered = False

                    # Check if the index fingertip enters the bounding rectangle
                    index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_fingertip_point = (int(index_fingertip.x * image.shape[1]), int(index_fingertip.y * image.shape[0]))
                    if x < index_fingertip_point[0] < x + w and y < index_fingertip_point[1] < y + h:
                        index_finger_entered = True
                    else:
                        index_finger_entered = False

                    # Check if the middle fingertip enters the bounding rectangle
                    middle_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    middle_fingertip_point = (int(middle_fingertip.x * image.shape[1]), int(middle_fingertip.y * image.shape[0]))
                    if x < middle_fingertip_point[0] < x + w and y < middle_fingertip_point[1] < y + h:
                        middle_finger_entered = True
                    else:
                        middle_finger_entered = False

                    # Check if the ring fingertip enters the bounding rectangle
                    ring_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    ring_fingertip_point = (int(ring_fingertip.x * image.shape[1]), int(ring_fingertip.y * image.shape[0]))
                    if x < ring_fingertip_point[0] < x + w and y < ring_fingertip_point[1] < y + h:
                        ring_finger_entered = True
                    else:
                        ring_finger_entered = False

                    # Check if the pinky fingertip enters the bounding rectangle
                    pinky_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    pinky_fingertip_point = (int(pinky_fingertip.x * image.shape[1]), int(pinky_fingertip.y * image.shape[0]))
                    if x < pinky_fingertip_point[0] < x + w and y < pinky_fingertip_point[1] < y + h:
                        pinky_finger_entered = True
                    else:
                        pinky_finger_entered = False


                    # Display fingertip entry status in the frame for each finger
                    if thumb_finger_entered:
                        cv2.putText(image, 'TR', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if index_finger_entered:
                        cv2.putText(image, 'IR', (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if middle_finger_entered:
                        cv2.putText(image, 'MR', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if ring_finger_entered:
                        cv2.putText(image, 'RR', (140, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if pinky_finger_entered:
                        cv2.putText(image, 'PR', (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


                # Draw the hand landmarks and connections on the frame
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Display the frame with the hand landmarks
        cv2.imshow('Standardized R and L', image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Release the camera
cap.release()
cv2.destroyAllWindows()
