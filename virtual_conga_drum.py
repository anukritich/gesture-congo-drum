import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
import pandas as pd
import time

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load sounds
pygame.mixer.init()
slap = pygame.mixer.Sound("Slap.wav")
bass = pygame.mixer.Sound("Bass.wav")
tone = pygame.mixer.Sound("Tone.wav")

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)

# For FPS calculation
prev_time = 0

# Variable to track the previous prediction (0: No Hit, 1: Hit)
last_prediction = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    h, w, _ = frame.shape

    # Draw drum zones
    zone_width = w // 3
    cv2.rectangle(frame, (0, 0), (zone_width, h), (255, 0, 0), 2)
    cv2.putText(frame, "Left", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.rectangle(frame, (zone_width, 0), (2 * zone_width, h), (0, 255, 0), 2)
    cv2.putText(frame, "Center", (zone_width + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(frame, (2 * zone_width, 0), (w, h), (0, 0, 255), 2)
    cv2.putText(frame, "Right", (2 * zone_width + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Process hand landmarks
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Extract Y-coordinates for prediction
            lm_y = [lm.y for lm in hand.landmark]
            # Get X-coordinates to determine the zone
            x_coords = [lm.x for lm in hand.landmark]
            avg_x = np.mean(x_coords)

            # Prepare input data for the model as a DataFrame with correct columns
            input_df = pd.DataFrame([lm_y], columns=[str(i) for i in range(21)])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            # Determine which zone based on the average x-coordinate
            zone = "Center"
            if avg_x < 0.33:
                zone = "Left"
            elif avg_x > 0.66:
                zone = "Right"

            # Play sound and log only on transition from no-hit to hit
            if prediction == 1 and last_prediction == 0:
                if zone == "Left":
                    slap.play()
                elif zone == "Center":
                    bass.play()
                elif zone == "Right":
                    tone.play()
                print(f"🥁 Hit detected! Zone: {zone}")

            # Update last_prediction with the current prediction
            last_prediction = prediction

    # FPS counter calculation
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps}', (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display frame
    cv2.imshow("Virtual Conga Drum", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
