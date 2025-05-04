

````markdown
# 🥁 Gesture-Based Virtual Conga Drum

A real-time computer vision-based virtual conga drum using hand gestures powered by MediaPipe, OpenCV, and a KNN classifier. Detects hits in left, center, and right zones and plays corresponding drum sounds.

## 🚀 Features

- Real-time hand tracking with MediaPipe
- Gesture classification using K-Nearest Neighbors (KNN)
- Plays conga drum sounds (Slap, Bass, Tone) based on hit zone
- Live visual feedback with FPS counter
- Visual evaluation: Confusion matrix, heatmap, gesture trajectory

## 🖥️ Demo

![Demo Screenshot](screenshots/demo.png)  
*Live gesture recognition and zone-based drum hits*

---

## 📁 Project Structure

```bash
gesture-congo-drum/
│
├── virtual_conga_drum.py         # Main app - live webcam drum interaction
├── train_model.py                # Train gesture classifier (KNN)
├── evaluate_model.py             # Evaluate model visually (heatmap, scatter, confusion matrix)
├── gesture_data.csv              # Hand gesture dataset
├── knn_model.pkl                 # Trained KNN model
├── scaler.pkl                    # Feature scaler
├── Slap.wav, Bass.wav, Tone.wav # Drum sounds
├── requirements.txt              # Python dependencies
└── README.md
````

---

## 📚 Technologies Used

* Python
* OpenCV
* MediaPipe
* scikit-learn
* pygame
* pandas, matplotlib, seaborn

---

