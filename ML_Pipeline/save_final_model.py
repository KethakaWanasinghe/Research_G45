import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "master_features.csv"
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = "features/master_features.csv"

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print("ERROR: Could not find 'master_features.csv'.")
    print("Please make sure you have run extract_features.py first!")
    exit()

# 40s Window + LSP 
df_40 = df[df['Window_Size'] == 40].copy()
features_lsp = ['MeanRR', 'SDNN', 'RMSSD', 'SD1', 'SD2', 'LSP_LF', 'LSP_HF', 'LSP_LFHF']

X = df_40[features_lsp].values
y = df_40['Label'].values

print("Training the Model...")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Final Random Forest
final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
final_rf.fit(X_scaled, y)

#Save
os.makedirs("models", exist_ok=True)
joblib.dump(final_rf, "models/stress_rf_model_40s.pkl")
joblib.dump(scaler, "models/stress_scaler_40s.pkl")

print("SUCCESS!")
print("File 1: models/stress_rf_model_40s.pkl ")
print("File 2: models/stress_scaler_40s.pkl ")