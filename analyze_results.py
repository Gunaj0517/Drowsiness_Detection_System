import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# STEP 1: Load predictions
# -------------------------------------------------------
csv_path = "predictions.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {csv_path}")
except FileNotFoundError:
    print("❌ predictions.csv not found. Run main_app.py first.")
    exit()

# -------------------------------------------------------
# STEP 2: Data Cleaning & Label Handling
# -------------------------------------------------------
# Ensure we only use rows where we have ground truth
df = df.dropna(subset=['gt_label'])
# Convert empty strings or 'None' to NaN and drop
df = df[df['gt_label'] != '']

if len(df) < 10:
    print("⚠️ Not enough labeled data to analyze. Please annotate more frames.")
    exit()

# Convert labels to numbers: Drowsy=1, Awake=0
df['y_true'] = df['gt_label'].apply(lambda x: 1 if x == 'drowsy' else 0)
df['y_pred_heuristic'] = df['pred_label'].apply(lambda x: 1 if x == 'drowsy' else 0)

# Ensure features are numeric
df['avg_ear'] = pd.to_numeric(df['avg_ear'], errors='coerce')
df['mar'] = pd.to_numeric(df['mar'], errors='coerce')
df = df.dropna(subset=['avg_ear', 'mar'])

# -------------------------------------------------------
# STEP 3: Evaluate Your Current (Heuristic) System
# -------------------------------------------------------
print("\n--- 🧠 Current Heuristic System Results ---")
y_true = df['y_true']
y_pred_heur = df['y_pred_heuristic']

print(f"Accuracy : {accuracy_score(y_true, y_pred_heur)*100:.2f}%")
print(f"F1-score : {f1_score(y_true, y_pred_heur)*100:.2f}%")
print("\nConfusion Matrix (Heuristic):")
print(confusion_matrix(y_true, y_pred_heur))

# -------------------------------------------------------
# STEP 4: Train Logistic Regression (The "Pro" Way)
# -------------------------------------------------------
print("\n--- 🤖 Logistic Regression Training ---")

# Features: EAR and MAR
X = df[['avg_ear', 'mar']]
y = df['y_true']

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred_lr = log_reg.predict(X_test)

# Evaluation
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"✅ Model Trained on {len(X_train)} frames, Tested on {len(X_test)} frames.")
print(f"Logistic Reg Accuracy : {acc_lr*100:.2f}%")
print(f"Logistic Reg F1-score : {f1_lr*100:.2f}%")

# Compare
improvement = f1_lr - f1_score(y_true, y_pred_heur)
if improvement > 0:
    print(f"\n🚀 Logistic Regression is BETTER by {improvement*100:.2f}% points!")
else:
    print(f"\n📉 Heuristic rules are performing similarly or better.")

# -------------------------------------------------------
# STEP 5: Interpret the Model
# -------------------------------------------------------
# This tells you exactly how important EAR vs MAR is
coef_ear = log_reg.coef_[0][0]
coef_mar = log_reg.coef_[0][1]
intercept = log_reg.intercept_[0]

print("\n🔍 Model Insights (The Formula):")
print(f"Score = ({coef_ear:.2f} * EAR) + ({coef_mar:.2f} * MAR) + {intercept:.2f}")
print("If Score > 0, prediction is DROWSY.")
print("-------------------------------------------------------")
print(f"Weight of EAR: {coef_ear:.2f} (Negative means lower EAR -> Drowsy)")
print(f"Weight of MAR: {coef_mar:.2f} (Positive means higher MAR -> Drowsy)")

# -------------------------------------------------------
# STEP 6: Visualize Decision Boundary
# -------------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot actual data points
drowsy = df[df['y_true'] == 1]
awake = df[df['y_true'] == 0]
plt.scatter(awake['avg_ear'], awake['mar'], color='green', label='Awake', alpha=0.5, s=10)
plt.scatter(drowsy['avg_ear'], drowsy['mar'], color='red', label='Drowsy', alpha=0.5, s=10)

# Calculate decision boundary line
# 0 = w1*EAR + w2*MAR + b  =>  MAR = -(w1*EAR + b) / w2
x_vals = np.array([df['avg_ear'].min(), df['avg_ear'].max()])
y_vals = -(coef_ear * x_vals + intercept) / coef_mar

plt.plot(x_vals, y_vals, color='blue', linestyle='--', linewidth=2, label='LogReg Boundary')

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('EAR (Eye Aspect Ratio)')
plt.ylabel('MAR (Mouth Aspect Ratio)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()