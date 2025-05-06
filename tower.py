import cv2
import mediapipe as mp
import numpy as np
import math

# ------------------------ Tower of Hanoi Setup ------------------------ #

NUM_DISKS = 3
TOWERS = [[], [], []]  # 3 Towers as stacks
for i in range(NUM_DISKS, 0, -1):
    TOWERS[0].append(i)

picked_disk = None
picked_from = None
move_count = 0

# ------------------------ MediaPipe Setup ------------------------ #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ------------------------ Helper Functions ------------------------ #
def draw_towers(img):
    h, w = img.shape[:2]
    base_y = h - 50
    tower_x = [w // 4, w // 2, 3 * w // 4]

    for i, tower in enumerate(TOWERS):
        x = tower_x[i]
        cv2.line(img, (x, base_y), (x, base_y - 200), (0, 255, 255), 6)

        for j, disk in enumerate(tower):
            disk_w = 30 + disk * 20
            y = base_y - (j + 1) * 30
            cv2.rectangle(img, (x - disk_w // 2, y - 10), (x + disk_w // 2, y + 10), (255, 100, 100), -1)
            cv2.putText(img, str(disk), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return tower_x, base_y

def detect_pinch(hand_landmarks):
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]

    dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
    return dist < 0.05

def get_tower_by_x(x_coord, tower_x_coords):
    diffs = [abs(x_coord - tx) for tx in tower_x_coords]
    return diffs.index(min(diffs))

def move_disk(from_tower, to_tower):
    global move_count
    if not TOWERS[from_tower]:
        return False
    if not TOWERS[to_tower] or TOWERS[to_tower][-1] > TOWERS[from_tower][-1]:
        TOWERS[to_tower].append(TOWERS[from_tower].pop())
        move_count += 1
        return True
    return False

# ------------------------ Main Game Loop ------------------------ #
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    tower_x_coords, base_y = draw_towers(frame)

    h, w = frame.shape[:2]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[8]
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)

            is_pinch = detect_pinch(hand_landmarks)
            selected_tower = get_tower_by_x(cx, tower_x_coords)

            if is_pinch:
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)

                if picked_disk is None and TOWERS[selected_tower]:
                    picked_disk = TOWERS[selected_tower][-1]
                    picked_from = selected_tower
                elif picked_disk is not None and selected_tower != picked_from:
                    success = move_disk(picked_from, selected_tower)
                    picked_disk = None
                    picked_from = None
            else:
                picked_disk = None
                picked_from = None

    # Draw move counter
    cv2.putText(frame, f"Moves: {move_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Win condition
    if len(TOWERS[2]) == NUM_DISKS:
        cv2.putText(frame, "🎉 Puzzle Solved!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Gesture Controlled Tower of Hanoi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
