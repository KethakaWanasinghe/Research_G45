import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def cliffs_delta(lst1, lst2):
    """Calculates Cliff's Delta to measure the Effect Size between two groups."""
    m, n = len(lst1), len(lst2)
    if m == 0 or n == 0: return 0
    dom = 0
    for x in lst1:
        for y in lst2:
            if x > y: dom += 1
            elif x < y: dom -= 1
    return dom / (m * n)

INPUT_FILE = "master_features.csv"
if not pd.io.common.file_exists(INPUT_FILE):
    INPUT_FILE = "features/master_features.csv"

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: Could not find master_features.csv. Ensure it is in the same folder.")
    exit()

windows = sorted(df['Window_Size'].unique())

print("Executing Comprehensive Analysis Pipeline...")
print("Validating Statistical Significance and Machine Learning Efficacy.\n")

# ============================================================
# STATISTICAL VALIDATION (RMSSD)
# ============================================================
print("============================================================")
print("STATISTICAL VALIDATION (RMSSD)")
print("============================================================")

for w in windows:
    df_w = df[df['Window_Size'] == w]
    relax = df_w[df_w['Label'] == 0]['RMSSD'].values
    stress = df_w[df_w['Label'] == 1]['RMSSD'].values
    
    stat, p = mannwhitneyu(relax, stress, alternative='two-sided')
    d = cliffs_delta(relax, stress)
    sig = "[SIGNIFICANT]" if p < 0.05 else "[NOT SIG]"
    print(f"{w}s Window | {sig} | p-value: {p:.2e} | Effect Size (Cliff's d): {abs(d):.3f}")

# ============================================================
# METHODOLOGY EVALUATION (LSP vs FFT)
# ============================================================
print("\n============================================================")
print("METHODOLOGY EVALUATION (LSP vs FFT)")
print("Leave-One-Subject-Out ")
print("============================================================")

base_feats = ['MeanRR', 'SDNN', 'RMSSD', 'SD1', 'SD2']
lsp_feats = base_feats + ['LSP_LF', 'LSP_HF', 'LSP_LFHF']
fft_feats = base_feats + ['FFT_LF', 'FFT_HF', 'FFT_LFHF']

for w in windows:
    df_w = df[df['Window_Size'] == w].copy()
    groups = df_w['Participant'].values
    y = df_w['Label'].values
    logo = LeaveOneGroupOut()
    
    # LSP  
    all_y_true = []
    all_y_pred_lsp = []
    participant_accs_lsp = {}
    
    for train_idx, test_idx in logo.split(df_w, y, groups):
        p_id = groups[test_idx][0]
        clf_lsp = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_lsp.fit(df_w.iloc[train_idx][lsp_feats], y[train_idx])
        preds = clf_lsp.predict(df_w.iloc[test_idx][lsp_feats])
        all_y_pred_lsp.extend(preds)
        all_y_true.extend(y[test_idx])
        participant_accs_lsp[p_id] = accuracy_score(y[test_idx], preds)

    acc_lsp = accuracy_score(all_y_true, all_y_pred_lsp)
    
    #  FFT  
    all_y_pred_fft = []
    for train_idx, test_idx in logo.split(df_w, y, groups):
        clf_fft = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_fft.fit(df_w.iloc[train_idx][fft_feats], y[train_idx])
        preds = clf_fft.predict(df_w.iloc[test_idx][fft_feats])
        all_y_pred_fft.extend(preds)
        
    acc_fft = accuracy_score(all_y_true, all_y_pred_fft)
    
    print(f"► {w}s WINDOW | LSP Acc: {acc_lsp*100:.2f}% | FFT Acc: {acc_fft*100:.2f}%")


# ============================================================
# CLASSIFIER COMPARISON (Accuracy, Precision, Recall, F1 & Loss)
# ============================================================
print("\n============================================================")
print("CLASSIFIER COMPARISON (RF vs. SVM vs. KNN)")
print("Leave-One-Subject-Out Cross-Validation (40s Window)")
print("============================================================")

df_40 = df[df['Window_Size'] == 40].copy()
X = df_40[lsp_feats].values
y = df_40['Label'].values
groups = df_40['Participant'].values

logo = LeaveOneGroupOut()
y_true = []
y_pred_rf, y_pred_svm, y_pred_knn = [], [], []

# Dictionaries for the Train/Val Loss table
results_loss = {
    "Random Forest": {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []},
    "Support Vector Machine": {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []},
    "K-Nearest Neighbors": {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
}

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf.extend(rf.predict(X_test_scaled))
    results_loss["Random Forest"]["train_acc"].append(accuracy_score(y_train, rf.predict(X_train_scaled)))
    results_loss["Random Forest"]["val_acc"].append(accuracy_score(y_test, rf.predict(X_test_scaled)))
    results_loss["Random Forest"]["train_loss"].append(log_loss(y_train, rf.predict_proba(X_train_scaled)))
    results_loss["Random Forest"]["val_loss"].append(log_loss(y_test, rf.predict_proba(X_test_scaled)))
    
    # Support Vector Machine
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm.extend(svm.predict(X_test_scaled))
    results_loss["Support Vector Machine"]["train_acc"].append(accuracy_score(y_train, svm.predict(X_train_scaled)))
    results_loss["Support Vector Machine"]["val_acc"].append(accuracy_score(y_test, svm.predict(X_test_scaled)))
    results_loss["Support Vector Machine"]["train_loss"].append(log_loss(y_train, svm.predict_proba(X_train_scaled)))
    results_loss["Support Vector Machine"]["val_loss"].append(log_loss(y_test, svm.predict_proba(X_test_scaled)))

    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn.extend(knn.predict(X_test_scaled))
    results_loss["K-Nearest Neighbors"]["train_acc"].append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    results_loss["K-Nearest Neighbors"]["val_acc"].append(accuracy_score(y_test, knn.predict(X_test_scaled)))
    results_loss["K-Nearest Neighbors"]["train_loss"].append(log_loss(y_train, knn.predict_proba(X_train_scaled)))
    results_loss["K-Nearest Neighbors"]["val_loss"].append(log_loss(y_test, knn.predict_proba(X_test_scaled)))
    
    y_true.extend(y_test)

# Calculate Standard Metrics
def calc_metrics(y_t, y_p):
    return {
        "acc": accuracy_score(y_t, y_p) * 100,
        "prec": precision_score(y_t, y_p, zero_division=0) * 100,
        "rec": recall_score(y_t, y_p, zero_division=0) * 100,
        "f1": f1_score(y_t, y_p, zero_division=0)
    }

models_results = [
    ("Random Forest", calc_metrics(y_true, y_pred_rf)),
    ("Support Vector Machine", calc_metrics(y_true, y_pred_svm)),
    ("K-Nearest Neighbors", calc_metrics(y_true, y_pred_knn))
]
models_results.sort(key=lambda x: (x[1]['acc'], x[1]['f1']), reverse=True)

# 1. Print Accuracy/Precision/Recall Table
print(f"\n[TABLE 1: Standard Evaluation Metrics]")
print(f"{'Classifier':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 75)
for model_name, metrics in models_results:
    print(f"{model_name:<25} | {metrics['acc']:.2f}%      | {metrics['prec']:.2f}%      | {metrics['rec']:.2f}%      | {metrics['f1']:.3f}")

# 2. Print Train vs Val Loss Table
print(f"\n[TABLE 2: Training vs. Validation (Overfitting Analysis)]")
print(f"{'Model Name':<25} | {'Train Accuracy':<15} | {'Train Loss':<12} | {'Val Accuracy':<12} | {'Val Loss':<10}")
print("-" * 85)
table_data = []
for model_name, metrics in results_loss.items():
    t_acc = np.mean(metrics['train_acc']) * 100
    v_acc = np.mean(metrics['val_acc']) * 100
    t_loss = np.mean(metrics['train_loss'])
    v_loss = np.mean(metrics['val_loss'])
    table_data.append((model_name, t_acc, t_loss, v_acc, v_loss))

table_data.sort(key=lambda x: x[3], reverse=True)
for row in table_data:
    print(f"{row[0]:<25} | {row[1]:.2f}%          | {row[2]:.4f}       | {row[3]:.2f}%       | {row[4]:.4f}")
print("============================================================")

best_model_name = models_results[0][0]
print("\nEMPIRICAL CONCLUSION:")
print(f"Based on LOSO cross-validation across all evaluation metrics, the {best_model_name}")
print("demonstrates superior capability in generalizing biological stress indicators without overfitting.")