import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

data = []
labels = []

print("Press 'h' to save HIT | 'n' to save NO HIT | 'q' to quit")

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            lm_y = [lm.y for lm in hand.landmark]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('h'):
                data.append(lm_y)
                labels.append(1)
                print("HIT saved")
            elif key == ord('n'):
                data.append(lm_y)
                labels.append(0)
                print("NO HIT saved")

    cv2.imshow("Collect Gesture Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df['gesture_label'] = labels
df.to_csv("gesture_data.csv", index=False)
print("Saved to gesture_data.csv")
