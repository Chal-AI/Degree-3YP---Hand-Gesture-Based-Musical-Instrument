import cv2
import mediapipe as mp

# Initialize MediaPipe components for drawing and hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open the default camera
cap = cv2.VideoCapture(0)

# Initialize the hand tracking module
hands = mp_hands.Hands()

# Start an infinite loop to capture videos from the camera
while cap.isOpened():
    # Read a frame from the camera
    ret, image = cap.read()

    # Check if the video was captured successfully
    if not ret:
        break  # Exit the loop if there's an issue with capturing frames

    # Convert the video to RGB color (MediaPipe uses RGB)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the frame with the hand tracking module
    results = hands.process(image)

    # Convert the image back to BGR format for displaying on the camera (OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Check if hand landmarks are detected in the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Display the frame with the hand landmarks
    cv2.namedWindow('Handtracker', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Handtracker', 1200, 1000) # ADJUST VALUES TO SET WIDTH  AND HEIGHT OF POP UP WINDOW
    cv2.imshow('Handtracker', image)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stops the camera and close the camera window
cap.release()
cv2.destroyAllWindows()
