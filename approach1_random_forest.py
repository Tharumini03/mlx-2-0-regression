import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

#print(f"Train shape: {train.shape}")
#print(f"Test shape:  {test.shape}")

# Separate features and target
target_col = "target"
id_col = "id"

y = train[target_col]
X = train.drop(columns=[target_col, id_col])
X_test = test.drop(columns=[id_col])

# print(f"\nTarget (y) shape: {y.shape}")
# print(f"Features (X) shape: {X.shape}")
# print(f"Test features shape: {X_test.shape}")

# Handle categorical (text) columns
label_encoders = {}

for col in X.columns:
    if X[col].dtype == "object" :
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

X = X.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Handle missing values
X = X.replace([np.inf, -np.inf], np.nan) # Replace infinity values with NaN first
X_test = X_test.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.mean(numeric_only=True))
X_test = X_test.fillna(X.mean(numeric_only=True))


# Split training data 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print(f"Training samples:   {len(X_train)}")
# print(f"Validation samples: {len(X_val)}")

# Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate on validation set
val_preds = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, val_preds))
mae  = mean_absolute_error(y_val, val_preds)
r2   = r2_score(y_val, val_preds)

print("\n--- Validation Results (Approach 1: Random Forest) ---")
print(f"RMSE (lower is better): {rmse:.4f}")
print(f"MAE  (lower is better): {mae:.4f}")
print(f"R²   (higher is better, max=1): {r2:.4f}")

# Retrain on full data and predict
print("\nRetraining on full training data...")
model.fit(X, y)

test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, 100)

submission = pd.DataFrame({
    "id": test[id_col],
    "target": test_preds
})

submission.to_csv("submissions/submission1_random_forest.csv", index=False)
print("\nSubmission saved to: submissions/submission1_random_forest.csv")
print(f"Total predictions: {len(submission)}")
print("\nFirst 5 predictions:")
print(submission.head())