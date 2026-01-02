import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load Data
try:
    data = pd.read_csv("gestures.csv", header=None)
except FileNotFoundError:
    print("Error: gestures.csv not found. Run collect_data.py first!")
    exit()

X = data.iloc[:, 1:].values # Landmarks
y = data.iloc[:, 0].values  # Gesture Labels

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 5. Save Model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as gesture_model.pkl")
