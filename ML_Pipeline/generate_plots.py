import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

OUT_DIR = "thesis_plots"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv("features/master_features.csv")
df['Condition'] = df['Label'].map({0: 'Relaxed', 1: 'Stressed'})
windows = sorted(df['Window_Size'].unique())

# =====================================================================
# PLOT 1: Physiological Validation
# =====================================================================
print("Generating Plot 1: Physiological Validation...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(x='Condition', y='RMSSD', data=df[df['Window_Size']==40], ax=axes[0], palette=['#2ecc71', '#e74c3c'])
axes[0].set_title("Time-Domain: Vagal Tone (RMSSD) at 40s")
axes[0].set_ylabel("RMSSD (ms)")

sns.boxplot(x='Condition', y='LSP_LFHF', data=df[df['Window_Size']==40], ax=axes[1], palette=['#3498db', '#f39c12'])
axes[1].set_title("Freq-Domain: Sympathovagal Balance (LF/HF) at 40s")
axes[1].set_ylabel("Lomb-Scargle LF/HF Ratio")
axes[1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Stress Threshold')
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_Physiological_Validation.png", dpi=300)
plt.close()


# =====================================================================
# RUN ML EVALUATIONS FOR PLOTS
# =====================================================================
print("Running LOSO Cross-Validation for ML Plots...")
base_feats = ['MeanRR', 'SDNN', 'RMSSD', 'SD1', 'SD2']
lsp_feats = base_feats + ['LSP_LF', 'LSP_HF', 'LSP_LFHF']
fft_feats = base_feats + ['FFT_LF', 'FFT_HF', 'FFT_LFHF']

acc_lsp_list, acc_fft_list = [], []
final_y_true, final_y_pred_rf = [], []
rf_metrics, svm_metrics, knn_metrics = {}, {}, {}

for w in windows:
    df_w = df[df['Window_Size'] == w].copy()
    groups = df_w['Participant'].values
    y = df_w['Label'].values
    X_lsp = df_w[lsp_feats].values
    X_fft = df_w[fft_feats].values
    logo = LeaveOneGroupOut()
    
    # LSP Eval
    y_true_w, y_pred_lsp_rf = [], []
    y_pred_svm, y_pred_knn = [], []
    
    for train_idx, test_idx in logo.split(X_lsp, y, groups):
        # Scale for 3-model comparison
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_lsp[train_idx])
        X_test_scaled = scaler.transform(X_lsp[test_idx])
        
        # RF (LSP)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y[train_idx])
        y_pred_lsp_rf.extend(rf.predict(X_test_scaled))
        
        if w == 40:
            # SVM
            svm = SVC(kernel='rbf', random_state=42)
            svm.fit(X_train_scaled, y[train_idx])
            y_pred_svm.extend(svm.predict(X_test_scaled))
            # KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_scaled, y[train_idx])
            y_pred_knn.extend(knn.predict(X_test_scaled))
            
        y_true_w.extend(y[test_idx])
        
    acc_lsp_list.append(accuracy_score(y_true_w, y_pred_lsp_rf) * 100)
    
    if w == 40:
        final_y_true = y_true_w
        final_y_pred_rf = y_pred_lsp_rf

        def get_mets(y_t, y_p):
            return [accuracy_score(y_t, y_p)*100, precision_score(y_t, y_p)*100, 
                    recall_score(y_t, y_p)*100, f1_score(y_t, y_p)*100]
        rf_metrics = get_mets(y_true_w, y_pred_lsp_rf)
        svm_metrics = get_mets(y_true_w, y_pred_svm)
        knn_metrics = get_mets(y_true_w, y_pred_knn)

    # FFT Eval
    y_pred_fft = []
    for train_idx, test_idx in logo.split(X_fft, y, groups):
        rf_fft = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_fft.fit(X_fft[train_idx], y[train_idx])
        y_pred_fft.extend(rf_fft.predict(X_fft[test_idx]))
    acc_fft_list.append(accuracy_score(y_true_w, y_pred_fft) * 100)


# =====================================================================
# PLOT 2: Optimization Curve
# =====================================================================
print("Generating Plot 2: Optimization Curve...")
plt.figure(figsize=(10, 6))
plt.plot(windows, acc_lsp_list, marker='o', linewidth=3, markersize=10, color='#9b59b6', label='RF with LSP')
plt.axvline(x=40, color='red', linestyle='--', alpha=0.7, label='Optimal Window (40s)')
plt.title("Algorithm Accuracy vs. Time Window Size")
plt.xlabel("Window Size (Seconds)")
plt.ylabel("LOSO Cross-Validation Accuracy (%)")
plt.xticks(windows)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_Optimization_Curve.png", dpi=300)
plt.close()


# =====================================================================
# PLOT 3: LSP vs FFT Comparison
# =====================================================================
print("Generating Plot 3: LSP vs FFT Comparison...")
plt.figure(figsize=(10, 6))
bar_width = 3
plt.bar(np.array(windows) - bar_width/2, acc_lsp_list, width=bar_width, color='#34495e', label='Lomb-Scargle (LSP)')
plt.bar(np.array(windows) + bar_width/2, acc_fft_list, width=bar_width, color='#bdc3c7', label='Standard FFT')
plt.title("Methodology Resilience: LSP vs Standard FFT")
plt.xlabel("Window Size (Seconds)")
plt.ylabel("Accuracy (%)")
plt.xticks(windows)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_LSP_vs_FFT_Comparison.png", dpi=300)
plt.close()


# =====================================================================
# PLOT 4: 3-Model Classifier Comparison
# =====================================================================
print("Generating Plot 4: Classifier Comparison (RF vs SVM vs KNN)...")
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score (x100)']
x = np.arange(len(metrics_names))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, rf_metrics, width, label='Random Forest', color='#2ecc71')
plt.bar(x, svm_metrics, width, label='SVM (RBF)', color='#e74c3c')
plt.bar(x + width, knn_metrics, width, label='KNN (k=5)', color='#3498db')

plt.ylabel('Percentage (%)')
plt.title('Classifier Performance Comparison at Optimal 40s Window')
plt.xticks(x, metrics_names)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)


for i, v in enumerate(rf_metrics): plt.text(i - width, v + 1, f"{v:.1f}", ha='center', fontsize=10)
for i, v in enumerate(svm_metrics): plt.text(i, v + 1, f"{v:.1f}", ha='center', fontsize=10)
for i, v in enumerate(knn_metrics): plt.text(i + width, v + 1, f"{v:.1f}", ha='center', fontsize=10)

plt.ylim(0, 105)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_Classifier_Comparison.png", dpi=300)
plt.close()


# =====================================================================
# PLOT 5: Feature Importance
# =====================================================================
print("Generating Plot 5: Feature Importance...")
df_40 = df[df['Window_Size'] == 40]
X_40 = df_40[lsp_feats]
y_40 = df_40['Label']
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(X_40, y_40)

importances = rf_final.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], color='#2980b9', align='center')
plt.yticks(range(len(indices)), [lsp_feats[i] for i in indices])
plt.xlabel('Relative Importance (Gini Impurity)')
plt.title('Explainable AI: Random Forest Feature Importance (40s Window)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/5_Feature_Importance.png", dpi=300)
plt.close()


# =====================================================================
# PLOT 6: Confusion Matrix
# =====================================================================
print("Generating Plot 6: Confusion Matrix...")
cm = confusion_matrix(final_y_true, final_y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16},
            xticklabels=['Predicted Relaxed', 'Predicted Stressed'],
            yticklabels=['Actual Relaxed', 'Actual Stressed'])
plt.title(f"Final Random Forest Confusion Matrix (40s Window)\nAccuracy: {acc_lsp_list[windows.index(40)]:.1f}%")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/6_Confusion_Matrix.png", dpi=300)
plt.close()

print(f"\nSuccessfully saved to the '{OUT_DIR}'")