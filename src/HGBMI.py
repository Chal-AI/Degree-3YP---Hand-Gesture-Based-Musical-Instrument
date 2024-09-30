import cv2
import mediapipe as mp
import mido
import numpy as np
import tkinter as tk

output_port_name = 'LM1 1'
output_port = mido.open_output(output_port_name)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_tracking_confidence=0.8, max_num_hands=2)
cap = cv2.VideoCapture(0)

playing_note0 = None
playing_note1 = None 
playing_note2 = None
playing_note3 = None
playing_note4 = None
playing_note5 = None
playing_note6 = None 
playing_note7 = None
playing_note8 = None
playing_note9 = None
base_note = None

note_mapping1 = { # Mapping base note for the LEFT HAND
    1: 'Amin', 2: 'Amin', 
    3: 'Cmaj', 
    4: 'Dmin',
    5: 'Emin', 
    6: 'Fmaj',
    7: 'Gmaj', 8: 'Gmaj', 9: 'Gmaj', 10: 'Gmaj'
}

note_mapping2 = { # Mapping base note for the RIGHT HAND
    1: 'A1', 2: 'A1', 
    3: 'B', 
    4: 'C',
    5: 'D', 
    6: 'E',
    7: 'F', 
    8: 'G', 
    9: 'A2', 10: 'A2', 11: 'A2', 12: 'A2', 13: 'A2', 14: 'A2'
}

# Defining the playing of a note function for the thumb (Left) ____________________________________________________________________________
def play_note0(note):
    global playing_note0
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note0 = note

def stop_note0():
    global playing_note0
    if playing_note0 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note0)))
        playing_note0 = None 

# Defining the playing of a note function for the index (Left) ____________________________________________________________________________
def play_note1(note):
    global playing_note1
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note1 = note 

def stop_note1():
    global playing_note1
    if playing_note1 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note1)))
        playing_note1 = None

# Defining the playing of a note function for the middle (Left) ___________________________________________________________________________
def play_note2(note):
    global playing_note2
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note2 = note 

def stop_note2():
    global playing_note2
    if playing_note2 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note2)))
        playing_note2 = None 

# Defining the playing of a note function for the ring (Left) _____________________________________________________________________________
def play_note3(note):
    global playing_note3
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note3 = note 

def stop_note3():
    global playing_note3
    if playing_note3 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note3)))
        playing_note3 = None 
        
# Defining the playing of a note function for the pinky (Left) ____________________________________________________________________________
def play_note4(note):
    global playing_note4
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note4 = note  

def stop_note4():
    global playing_note4
    if playing_note4 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note4)))
        playing_note4 = None

# Defining the playing of a note function for the thumb (Right) ___________________________________________________________________________
def play_note5(note):
    global playing_note5
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note5 = note

def stop_note5():
    global playing_note5
    if playing_note5 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note5)))
        playing_note5 = None 

# Defining the playing of a note function for the index (Right) ___________________________________________________________________________
def play_note6(note):
    global playing_note6
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note6 = note 

def stop_note6():
    global playing_note6
    if playing_note6 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note6)))
        playing_note6 = None

# Defining the playing of a note function for the middle (Right) __________________________________________________________________________
def play_note7(note):
    global playing_note7
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note7 = note 

def stop_note7():
    global playing_note7
    if playing_note7 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note7)))
        playing_note7 = None 

# Defining the playing of a note function for the ring (Right) ____________________________________________________________________________
def play_note8(note):
    global playing_note8
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note8 = note 

def stop_note8():
    global playing_note8
    if playing_note8 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note8)))
        playing_note8 = None 
        
# Defining the playing of a note function for the pinky (Right) ___________________________________________________________________________
def play_note9(note):
    global playing_note9
    if note is not None:
        output_port.send(mido.Message('note_on', note=int(note), velocity=64))
        playing_note9 = note  

def stop_note9():
    global playing_note9
    if playing_note9 is not None:
        output_port.send(mido.Message('note_off', note=int(playing_note9)))
        playing_note9 = None

def stop_all_notes():
    stop_note0()
    stop_note1()
    stop_note2()
    stop_note3()
    stop_note4()
    stop_note5()
    stop_note6()
    stop_note7()
    stop_note8()
    stop_note9()

def start_camera():
    global transpose
    camera_view = camera_view_var.get()
    show_division_visualizer = show_division_var.get()
    transpose = transpose_var.get()
    global cap
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(0)

    c_major_scale_T1 = [33, 35, 36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81] 
    c_major_scale1 = [x + transpose for x in c_major_scale_T1]
    note_range = len(c_major_scale1)
    note_offset = min(c_major_scale1)

    try:
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            default = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            black_background = np.zeros_like(image)
            blurred_image = cv2.GaussianBlur(image, (45, 45), 0)
            blurred_image_bgr = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)

            if camera_view == 0:
                selected_view = default
            elif camera_view == 1:
                selected_view = black_background
            elif camera_view == 2:
                selected_view = blurred_image_bgr
            else:
                print("Invalid camera view")
                processed_image = None

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_side = handedness_info.classification[0].label

                    if hand_side == 'Left':
                        knuckle_landmarks = [hand_landmarks.landmark[idx] for idx in [mp_hands.HandLandmark.THUMB_MCP,
                                                                                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                                                                                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                                                                                mp_hands.HandLandmark.RING_FINGER_MCP,
                                                                                mp_hands.HandLandmark.PINKY_MCP]]
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        wrist_yl = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

                        inverted_level = 11 - int((wrist_yl * 10) + 1)
                        musical_note = note_mapping1.get(inverted_level, 'Unknown')

                        cv2.putText(selected_view, f"Note: {inverted_level} ({musical_note})", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if musical_note == 'Amin':
                            base_note = c_major_scale1[0]
                        elif musical_note == 'Cmaj':
                            base_note = c_major_scale1[2]
                        elif musical_note == 'Dmin':
                            base_note = c_major_scale1[3]
                        elif musical_note == 'Emin':
                            base_note = c_major_scale1[4]
                        elif musical_note == 'Fmaj':
                            base_note = c_major_scale1[5]
                        elif musical_note == 'Gmaj':
                            base_note = c_major_scale1[6]
                        else:
                            musical_note == None

                        knuckle_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in knuckle_landmarks]
                        wrist_point = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
                        x, y, w, h = cv2.boundingRect(np.array(knuckle_points + [wrist_point]))
                        cv2.rectangle(selected_view, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        thumb_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumb_fingertip_point = (int(thumb_fingertip.x * image.shape[1]), int(thumb_fingertip.y * image.shape[0]))
                        if x < thumb_fingertip_point[0] < x + w and y < thumb_fingertip_point[1] < y + h:
                            thumb_finger_entered = True
                        else:
                            thumb_finger_entered = False

                        index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_fingertip_point = (int(index_fingertip.x * image.shape[1]), int(index_fingertip.y * image.shape[0]))
                        if x < index_fingertip_point[0] < x + w and y < index_fingertip_point[1] < y + h:
                            index_finger_entered = True
                        else:
                            index_finger_entered = False

                        middle_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        middle_fingertip_point = (int(middle_fingertip.x * image.shape[1]), int(middle_fingertip.y * image.shape[0]))
                        if x < middle_fingertip_point[0] < x + w and y < middle_fingertip_point[1] < y + h:
                            middle_finger_entered = True
                        else:
                            middle_finger_entered = False

                        ring_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                        ring_fingertip_point = (int(ring_fingertip.x * image.shape[1]), int(ring_fingertip.y * image.shape[0]))
                        if x < ring_fingertip_point[0] < x + w and y < ring_fingertip_point[1] < y + h:
                            ring_finger_entered = True
                        else:
                            ring_finger_entered = False

                        pinky_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                        pinky_fingertip_point = (int(pinky_fingertip.x * image.shape[1]), int(pinky_fingertip.y * image.shape[0]))
                        if x < pinky_fingertip_point[0] < x + w and y < pinky_fingertip_point[1] < y + h:
                            pinky_finger_entered = True
                        else:
                            pinky_finger_entered = False
    
                        if pinky_finger_entered:
                            note = base_note
                            if note != playing_note0 and note in c_major_scale1:
                                if playing_note0 is not None:
                                    stop_note0()
                                play_note0(note)
                        else:
                            stop_note0()

                        if ring_finger_entered:
                            base_note_index = c_major_scale1.index(base_note)
                            next_note_index = min(base_note_index + 2, len(c_major_scale1) - 1)
                            note = c_major_scale1[next_note_index]

                            if note != playing_note1 and note in c_major_scale1:
                                if playing_note1 is not None:
                                    stop_note1()
                                play_note1(note)
                        else:
                            stop_note1()

                        if middle_finger_entered:
                            base_note_index = c_major_scale1.index(base_note)
                            next_note_index = min(base_note_index + 4, len(c_major_scale1) - 1)
                            note = c_major_scale1[next_note_index]

                            if note != playing_note2 and note in c_major_scale1:
                                if playing_note2 is not None:
                                    stop_note2()
                                play_note2(note)
                        else:
                            stop_note2()

                        if index_finger_entered:
                            base_note_index = c_major_scale1.index(base_note)
                            next_note_index = min(base_note_index + 7, len(c_major_scale1) - 1)
                            note = c_major_scale1[next_note_index]

                            if note != playing_note3 and note in c_major_scale1:
                                if playing_note3 is not None:
                                    stop_note3()
                                play_note3(note)
                        else:
                            stop_note3()

                        if thumb_finger_entered:
                            base_note_index = c_major_scale1.index(base_note)
                            next_note_index = min(base_note_index + 9, len(c_major_scale1) - 1)
                            note = c_major_scale1[next_note_index]

                            if note != playing_note4 and note in c_major_scale1:
                                if playing_note4 is not None:
                                    stop_note4()
                                play_note4(note)
                        else:
                            stop_note4()


                    elif hand_side == 'Right':
                        knuckle_landmarks = [hand_landmarks.landmark[idx] for idx in [mp_hands.HandLandmark.THUMB_MCP,
                                                                                    mp_hands.HandLandmark.INDEX_FINGER_MCP,
                                                                                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                                                                                    mp_hands.HandLandmark.RING_FINGER_MCP,
                                                                                    mp_hands.HandLandmark.PINKY_MCP]]
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        wrist_yr = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

                        inverted_level = 15 - int((wrist_yr * 14) + 1)
                        musical_note = note_mapping2.get(inverted_level, 'Unknown')

                        cv2.putText(selected_view, f"Note: {inverted_level} ({musical_note})", (320, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if musical_note == 'A1':
                            base_note = c_major_scale1[14]
                        elif musical_note == 'B':
                            base_note = c_major_scale1[15]
                        elif musical_note == 'C':
                            base_note = c_major_scale1[16]
                        elif musical_note == 'D':
                            base_note = c_major_scale1[17]
                        elif musical_note == 'E':
                            base_note = c_major_scale1[18]
                        elif musical_note == 'F':
                            base_note = c_major_scale1[19]
                        elif musical_note == 'G':
                            base_note = c_major_scale1[20]
                        elif musical_note == 'A2':
                            base_note = c_major_scale1[21]
                        else:
                            musical_note == None

                        knuckle_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in knuckle_landmarks]
                        wrist_point = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
                        x, y, w, h = cv2.boundingRect(np.array(knuckle_points + [wrist_point]))
                        cv2.rectangle(selected_view, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        thumb_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumb_fingertip_point = (int(thumb_fingertip.x * image.shape[1]), int(thumb_fingertip.y * image.shape[0]))
                        if x < thumb_fingertip_point[0] < x + w and y < thumb_fingertip_point[1] < y + h:
                            thumb_finger_entered = True
                        else:
                            thumb_finger_entered = False

                        index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_fingertip_point = (int(index_fingertip.x * image.shape[1]), int(index_fingertip.y * image.shape[0]))
                        if x < index_fingertip_point[0] < x + w and y < index_fingertip_point[1] < y + h:
                            index_finger_entered = True
                        else:
                            index_finger_entered = False

                        middle_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        middle_fingertip_point = (int(middle_fingertip.x * image.shape[1]), int(middle_fingertip.y * image.shape[0]))
                        if x < middle_fingertip_point[0] < x + w and y < middle_fingertip_point[1] < y + h:
                            middle_finger_entered = True
                        else:
                            middle_finger_entered = False

                        ring_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                        ring_fingertip_point = (int(ring_fingertip.x * image.shape[1]), int(ring_fingertip.y * image.shape[0]))
                        if x < ring_fingertip_point[0] < x + w and y < ring_fingertip_point[1] < y + h:
                            ring_finger_entered = True
                        else:
                            ring_finger_entered = False

                        pinky_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                        pinky_fingertip_point = (int(pinky_fingertip.x * image.shape[1]), int(pinky_fingertip.y * image.shape[0]))
                        if x < pinky_fingertip_point[0] < x + w and y < pinky_fingertip_point[1] < y + h:
                            pinky_finger_entered = True
                        else:
                            pinky_finger_entered = False


                        if thumb_finger_entered:
                            note = base_note
                            if note != playing_note5 and note in c_major_scale1:
                                if playing_note5 is not None:
                                    stop_note5()
                                play_note5(note)
                        else:
                            stop_note5()

                        if index_finger_entered:
                            base_note_index = c_major_scale1.index(base_note)
                            next_note_index = min(base_note_index + 1, len(c_major_scale1) - 1)
                            note = c_major_scale1[next_note_index]

                            if note != playing_note6 and note in c_major_scale1:
                                if playing_note6 is not None:
                                    stop_note6()
                                play_note6(note)
                        else:
                            stop_note6()

                        if middle_finger_entered:
                            base_note_index = c_major_scale1.index(base_note)
                            next_note_index = min(base_note_index + 2, len(c_major_scale1) - 1)
                            note = c_major_scale1[next_note_index]

                            if note != playing_note7 and note in c_major_scale1:
                                if playing_note7 is not None:
                                    stop_note7()
                                play_note7(note)
                        else:
                            stop_note7()

                        if ring_finger_entered:
                            base_note_index = c_major_scale1.index(base_note)
                            next_note_index = min(base_note_index + 3, len(c_major_scale1) - 1)
                            note = c_major_scale1[next_note_index]

                            if note != playing_note8 and note in c_major_scale1:
                                if playing_note8 is not None:
                                    stop_note8()
                                play_note8(note)
                        else:
                            stop_note8()

                        if pinky_finger_entered:
                            base_note_index = c_major_scale1.index(base_note)
                            next_note_index = min(base_note_index + 4, len(c_major_scale1) - 1)
                            note = c_major_scale1[next_note_index]

                            if note != playing_note9 and note in c_major_scale1:
                                if playing_note9 is not None:
                                    stop_note9()
                                play_note9(note)
                        else:
                            stop_note9()

                    mp_drawing.draw_landmarks(selected_view, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if show_division_visualizer:
                for i in range(1, 10):
                    y = i * (image.shape[0] // 10)
                    cv2.line(selected_view, (0, y), (image.shape[1] // 2, y), (255, 255, 255), 1)

                for i in range(1, 14):
                    y = i * (image.shape[0] // 14)
                    cv2.line(selected_view, (image.shape[1] // 2, y), (image.shape[1], y), (255, 255, 255), 1)

            cv2.imshow('Hand Gesture Control', selected_view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_all_notes()

root = tk.Tk()
root.title("Hand Gesture Music")

camera_view_var = tk.IntVar(value=2)
show_division_var = tk.BooleanVar(value=False)
transpose_var = tk.IntVar(value=0)

title_label = tk.Label(root, text="HAND GESTURE MUSIC", font=("Helvetica", 20))
title_label.grid(row=0, column=0, columnspan=2, pady=10)

camera_views = [(0, "Original View"), (1, "Black Background"), (2, "Blurred Image")]
camera_view_label = tk.Label(root, text="Camera View:")
camera_view_label.grid(row=1, column=0, padx=5, pady=5)

for idx, (value, text) in enumerate(camera_views):
    rb = tk.Radiobutton(root, text=text, variable=camera_view_var, value=value)
    rb.grid(row=1, column=idx+1, padx=5, pady=5)

show_division_label = tk.Label(root, text="Show Divisions:")
show_division_label.grid(row=2, column=0, padx=5, pady=5)
show_division_check = tk.Checkbutton(root, variable=show_division_var)
show_division_check.grid(row=2, column=1, padx=5, pady=5)

transpose_label = tk.Label(root, text="Transpose:")
transpose_label.grid(row=3, column=0, padx=5, pady=5)
transpose_slider = tk.Scale(root, from_=0, to=9, orient=tk.HORIZONTAL, variable=transpose_var)
transpose_slider.grid(row=3, column=1, padx=5, pady=5)

apply_button = tk.Button(root, text="Apply Changes & Start Camera", command=start_camera)
apply_button.grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
