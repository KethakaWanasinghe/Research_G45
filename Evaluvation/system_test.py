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
    print("Error: Could not find models. Ensure 'stress_rf_model_40s.pkl' is in the models folder.")
    exit()

def extract_features(timestamps, rr_intervals):
    valid = (rr_intervals > 300) & (rr_intervals < 2000)
    t_clean = timestamps[valid]
    rr_clean = rr_intervals[valid]
    
    if len(rr_clean) < 5: return None, None
    
    mean_rr = np.mean(rr_clean)
    sdnn = np.std(rr_clean, ddof=1)
    diffs = np.diff(rr_clean)
    rmssd = np.sqrt(np.mean(diffs**2))
    sdsd = np.std(diffs, ddof=1)
    sd1 = np.sqrt(0.5 * sdsd**2)
    sd2 = np.sqrt(max(0, 2 * sdnn**2 - 0.5 * sdsd**2))
    
    f = np.linspace(0.04, 0.4, 100)
    w = 2 * np.pi * f
    pgram = lombscargle(t_clean, rr_clean - np.mean(rr_clean), w, normalize=False)
    psd = pgram * 2.0 / len(t_clean)
    
    lf = np.trapz(psd[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = np.trapz(psd[(f >= 0.15) & (f <= 0.4)], f[(f >= 0.15) & (f <= 0.4)])
    
    lf = max(lf, 0.001)
    hf = max(hf, 0.001)
    lf_hf = lf / hf
    
    return np.array([[mean_rr, sdnn, rmssd, sd1, sd2, lf, hf, lf_hf]]), lf_hf

print("Running End-to-End Integration Test")
print("=======================================================================================================")
print(f"{'Subject':<8} | {'State':<8} | {'RMSSD (Vagal)':<15} | {'LF/HF Ratio':<12} | {'AI Output':<15} | {'Prescribed Therapy'}")
print("=======================================================================================================")

participants = [f"P{str(i).zfill(2)}" for i in range(31, 41)]
states = ['relax', 'exam']

for p in participants:
    for state in states:

        file_path = f"Demo-Data/{p}_{state}_raw.csv"
        if not os.path.exists(file_path): 
            file_path = f"raw_data/{p}_{state}_raw.csv"
            if not os.path.exists(file_path):
                continue
            
        df = pd.read_csv(file_path)

        df = df[df['timestamp'] <= df['timestamp'].iloc[0] + 40]
        
        features, lsp_lfhf = extract_features(df['timestamp'].values, df['rr_interval'].values)
        
        if features is not None:
            features_scaled = scaler.transform(features)
            prediction = rf_model.predict(features_scaled)[0]
            

            if prediction == 0:
                final_label = "Low Stress"
                therapy = "Medium Energy (Focus)"
            else:
                if lsp_lfhf >= 3.0:
                    final_label = "High Stress"
                    therapy = "Low Energy (Calming)"
                else:
                    final_label = "Mild Stress"
                    therapy = "Medium Energy (Focus)"
                    
            rmssd_val = features[0][2]
            
            print(f"{p:<8} | {state.upper():<8} | {rmssd_val:<15.2f} | {lsp_lfhf:<12.2f} | {final_label:<15} | {therapy}")

print("=======================================================================================================")