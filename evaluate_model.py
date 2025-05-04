import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load data and model
df = pd.read_csv("gesture_data.csv")
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Split dataset
X = df.drop("gesture_label", axis=1)
y = df["gesture_label"]
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------------
# Figure 1: Feature Correlation Heatmap
# -------------------------------------
def plot_feature_correlation():
    plt.figure(figsize=(10, 7))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, cbar=True)
    plt.title("Figure 1: Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# -------------------------------------
# Figure 2: Gesture Movement Plot (Δy between wrist and fingertip)
# -------------------------------------
def plot_movement_sample():
    if len(df) >= 10:
        sample = df.iloc[:10]
        frame_ids = list(range(10))
        delta_y = sample["0"] - sample["8"]  # Wrist (0) - Index Tip (8)
        labels = sample["gesture_label"].replace({1: "Hit", 0: "No Hit"})

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=frame_ids, y=delta_y, marker="o", label="Δy (wrist - tip)")
        for i, label in enumerate(labels):
            plt.text(frame_ids[i], delta_y.iloc[i] + 0.005, label, fontsize=9,
                     color='red' if label == 'Hit' else 'gray')
        plt.title("Figure 2: Gesture Movement Plot")
        plt.xlabel("Frame")
        plt.ylabel("Δy")
        plt.grid()
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Not enough samples for gesture movement plot.")

# -------------------------------------
# Figure 3: KNN Accuracy vs K Plot
# -------------------------------------
def plot_knn_accuracy_range():
    accuracies = []
    k_range = range(1, 10, 2)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(k_range), y=accuracies, palette="Blues_d")
    plt.title("Figure 3: KNN Accuracy vs K Values")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.0)
    plt.grid()
    plt.tight_layout()
    plt.show()

# -------------------------------------
# Figure 4: Confusion Matrix
# -------------------------------------
def plot_conf_matrix():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Hit', 'Hit'], yticklabels=['No Hit', 'Hit'])
    plt.title("Figure 4: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# -------------------------------------
# Figure 5: Actual vs Predicted Gestures (Scatter Plot)
# -------------------------------------
def plot_actual_vs_predicted():
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6, marker='o')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.4, marker='x')
    plt.title("Figure 5: Actual vs Predicted Gestures")
    plt.xlabel("Sample Index")
    plt.ylabel("Gesture Label (0 = No Hit, 1 = Hit)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_timeline():
    if len(y_test) == 0 or len(y_pred) == 0:
        print("⚠️ Not enough data to show confusion plot.")
        return

    actual = y_test.reset_index(drop=True)
    predicted = pd.Series(y_pred)

    match = actual == predicted
    colors = ['green' if m else 'red' for m in match]

    plt.figure(figsize=(10, 2.5))
    plt.scatter(range(len(actual)), [1]*len(actual), c=colors, marker='|', s=300)
    plt.title("Figure 6: Confusion Timeline Plot")
    plt.yticks([])
    plt.xlabel("Sample Index")
    plt.legend(handles=[
        plt.Line2D([0], [0], color='green', lw=4, label='Correct'),
        plt.Line2D([0], [0], color='red', lw=4, label='Incorrect')
    ])
    plt.tight_layout()
    plt.show()

# 🔍 Run all evaluations
# plot_feature_correlation()
# plot_movement_sample()
# plot_knn_accuracy_range()
# plot_conf_matrix()
# plot_actual_vs_predicted()
plot_confusion_timeline()

