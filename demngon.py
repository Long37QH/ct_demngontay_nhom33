import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands, draw=True, display = True):
    output_image = image.copy()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks and draw:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                     thickness=2, circle_radius=2))
    if display:
        plt.figure(figsize=[15, 15])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output");
        plt.axis('off');

    else:
        return output_image, results

#thiêt lập xây dựng bộ đếm
def countFingers(image, results, draw=True, display=True):
    height, width, _ = image.shape
    output_image = image.copy()
    count = {'RIGHT': 0, 'LEFT': 0}
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_label = hand_info.classification[0].label

        hand_landmarks = results.multi_hand_landmarks[hand_index]

        for tip_index in fingers_tips_ids:
            finger_name = tip_index.name.split("_")[0]
            for tip_index in fingers_tips_ids:

                finger_name = tip_index.name.split("_")[0]
                if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                    fingers_statuses[hand_label.upper() + "_" + finger_name] = True
                    count[hand_label.upper()] += 1/4

            thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
            #đếm ngón tay cái
            if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
                fingers_statuses[hand_label.upper() + "_THUMB"] = True
                count[hand_label.upper()] += 1/4

    if draw:
        cv2.putText(output_image, " So Ngon Tay : ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (20, 255, 155), 2)
        cv2.putText(output_image, str(int(sum(count.values()))), (350,95), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (20, 255, 155), 10, 5)
    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    else:
        return output_image, fingers_statuses, count


#thiêt lập đọc cam
camera_video = cv2.VideoCapture(0)

cv2.namedWindow('Chuong trinh dem ngon tay', cv2.WINDOW_NORMAL)
while True:
    ok, frame = camera_video.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    frame, results = detectHandsLandmarks(frame, hands_videos, display=False)
    if results.multi_hand_landmarks:
        frame, fingers_statuses, count = countFingers(frame, results, display=False)
    cv2.imshow('Chuong trinh dem ngon tay', frame)
    k = cv2.waitKey(1)
    if (k == ord('q')):
        break
camera_video.release()
cv2.destroyWindow()
