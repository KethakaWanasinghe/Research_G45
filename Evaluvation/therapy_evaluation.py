import pandas as pd
import numpy as np
import joblib
import os
from scipy.signal import lombscargle
import warnings

warnings.filterwarnings('ignore')

try:
    rf_model = joblib.load('models/stress_rf_model_40s.pkl')
    scaler = joblib.load('models/stress_scaler_40s.pkl')
except FileNotFoundError:
    print("Error: Models not found.")
    exit()

def extract_features(timestamps, rr_intervals):
    valid = (rr_intervals > 300) & (rr_intervals < 2000)
    t_clean = timestamps[valid]
    rr_clean = rr_intervals[valid]
    if len(rr_clean) < 5: return None, None
    mean_rr, sdnn = np.mean(rr_clean), np.std(rr_clean, ddof=1)
    diffs = np.diff(rr_clean)
    rmssd, sdsd = np.sqrt(np.mean(diffs**2)), np.std(diffs, ddof=1)
    sd1 = np.sqrt(0.5 * sdsd**2)
    sd2 = np.sqrt(max(0, 2 * sdnn**2 - 0.5 * sdsd**2))
    
    f = np.linspace(0.04, 0.4, 100)
    w = 2 * np.pi * f
    pgram = lombscargle(t_clean, rr_clean - np.mean(rr_clean), w, normalize=False)
    psd = pgram * 2.0 / len(t_clean)
    
    lf = np.trapz(psd[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = np.trapz(psd[(f >= 0.15) & (f <= 0.4)], f[(f >= 0.15) & (f <= 0.4)])
    lf, hf = max(lf, 0.001), max(hf, 0.001)
    
    return np.array([[mean_rr, sdnn, rmssd, sd1, sd2, lf, hf, lf/hf]]), lf/hf

def get_diagnosis(features_scaled, lfhf_ratio):
    pred = rf_model.predict(features_scaled)[0]
    if pred == 0: return "Low Stress", "Medium Energy"
    if lfhf_ratio >= 3.0: return "High Stress", "Low Energy"
    return "Mild Stress", "Medium Energy"

participants = [f"P{str(i).zfill(2)}" for i in range(31, 41)]
individual_results = []
agg = {
    "High Stress": {"N": 0, "Music": "Low Energy", "Improved": 0, "No Change": 0, "Worsened": 0},
    "Mild Stress": {"N": 0, "Music": "Medium Energy", "Improved": 0, "No Change": 0, "Worsened": 0},
    "Low Stress": {"N": 0, "Music": "Medium Energy", "Improved": 0, "No Change": 0, "Worsened": 0}
}

print("Running Objective Blind Evaluation on Pre/Post data...")

for p in participants:
    pre_path = f"Demo-Data/{p}_pre_raw.csv"
    post_path = f"Demo-Data/{p}_post_raw.csv"
    if not (os.path.exists(pre_path) and os.path.exists(post_path)): continue
        
    df_pre = pd.read_csv(pre_path)
    df_pre_40 = df_pre[df_pre['timestamp'] <= df_pre['timestamp'].iloc[0] + 40]
    feat_pre, pre_lfhf = extract_features(df_pre_40['timestamp'].values, df_pre_40['rr_interval'].values)
    
    df_post = pd.read_csv(post_path)
    df_post_40 = df_post[df_post['timestamp'] <= df_post['timestamp'].iloc[0] + 40]
    feat_post, post_lfhf = extract_features(df_post_40['timestamp'].values, df_post_40['rr_interval'].values)
    
    if feat_pre is not None and feat_post is not None:
        pre_label, music = get_diagnosis(scaler.transform(feat_pre), pre_lfhf)
        post_label, _ = get_diagnosis(scaler.transform(feat_post), post_lfhf)
            
        display_change = ""
        actual_change_key = ""

        if pre_label == "High Stress":
            actual_change_key = "Improved" if post_label in ["Mild Stress", "Low Stress"] else "No Change"
            display_change = actual_change_key
        elif pre_label == "Mild Stress":
            if post_label == "Low Stress": actual_change_key = display_change = "Improved"
            elif post_label == "Mild Stress": actual_change_key = display_change = "No Change"
            else: actual_change_key = display_change = "Worsened"
        else:
            if post_label == "Low Stress": 
                actual_change_key = "Improved" 
                display_change = "Eustress Maintained" 
            else: 
                actual_change_key = display_change = "Worsened"
                
        agg[pre_label][actual_change_key] += 1
        agg[pre_label]["N"] += 1
        
        individual_results.append((p, pre_label, music, post_label, display_change, f"{pre_lfhf:.2f} -> {post_lfhf:.2f}", f"{feat_pre[0][2]:.1f}"))

print("\n[TABLE 1: Individual Therapy Response]")
print(f"{'Participant ID':<15} | {'Pre-Music Level':<18} | {'Post-Music Level':<18} | {'Change':<20} | {'Pre-RMSSD (ms)':<15}")
print("-" * 90)
for r in individual_results: 
    print(f"{r[0]:<15} | {r[1]:<18} | {r[3]:<18} | {r[4]:<20} | {r[6]:<15}")

print("\n\n[TABLE 2: Aggregated Therapy Efficacy]")
print(f"{'Initial Tier':<15} | {'N':<4} | {'Improved':<10} | {'No Change':<10} | {'Worsened':<10} | {'% Improved'}")
print("-" * 75)

tot_n = tot_imp = tot_no = tot_worse = 0
for tier in ["High Stress", "Mild Stress", "Low Stress"]:
    d = agg[tier]
    if d["N"] == 0: continue
    pct = (d["Improved"] / d["N"]) * 100
    print(f"{tier:<15} | {d['N']:<4} | {d['Improved']:<10} | {d['No Change']:<10} | {d['Worsened']:<10} | {pct:.1f}%")
    tot_n += d["N"]; tot_imp += d["Improved"]; tot_no += d["No Change"]; tot_worse += d["Worsened"]

print("-" * 75)
if tot_n > 0: print(f"{'All Students':<15} | {tot_n:<4} | {tot_imp:<10} | {tot_no:<10} | {tot_worse:<10} | {(tot_imp / tot_n) * 100:.1f}%\n")