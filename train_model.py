import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("gesture_data.csv")
X = df.drop("gesture_label", axis=1)
y = df["gesture_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved")
