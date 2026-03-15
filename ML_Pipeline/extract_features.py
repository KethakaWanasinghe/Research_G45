import pandas as pd
import numpy as np
import os
import glob
from scipy.signal import periodogram, lombscargle, detrend

UNEVEN_DIR = "cleaned_uneven"
UNIFORM_DIR = "cleaned_uniform"
OUTPUT_DIR = "features"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calc_time_nonlinear(rr):
    mean_rr, sdnn = np.mean(rr), np.std(rr, ddof=1)
    diffs = np.diff(rr)
    rmssd = np.sqrt(np.mean(diffs**2))
    sdsd = np.std(diffs, ddof=1)
    sd1 = np.sqrt(0.5 * sdsd**2)
    sd2 = np.sqrt(max(0, 2 * sdnn**2 - 0.5 * sdsd**2))
    return mean_rr, sdnn, rmssd, sd1, sd2

def calc_lsp_features(t, rr):
    f = np.linspace(0.04, 0.4, 100)
    w = 2 * np.pi * f
    pgram = lombscargle(t, rr - np.mean(rr), w, normalize=False)
    psd = pgram * 2.0 / len(t)
    lf = np.trapz(psd[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = np.trapz(psd[(f >= 0.15) & (f <= 0.4)], f[(f >= 0.15) & (f <= 0.4)])
    return lf, hf, lf / max(hf, 0.0001)

def calc_fft_features(rr_uniform):
    rr_detrended = detrend(rr_uniform, type='linear')
    f, pxx = periodogram(rr_detrended, fs=4.0, window='hamming', scaling='density')
    lf = np.trapz(pxx[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = np.trapz(pxx[(f >= 0.15) & (f <= 0.4)], f[(f >= 0.15) & (f <= 0.4)])
    return lf, hf, lf / max(hf, 0.0001)

all_features = []
file_list = glob.glob(f"{UNEVEN_DIR}/*.csv")

for uneven_path in file_list:
    filename = os.path.basename(uneven_path)
    df_u = pd.read_csv(uneven_path)
    df_i = pd.read_csv(os.path.join(UNIFORM_DIR, filename.replace("_uneven", "_uniform")))
    
    # Windows
    for w in [10, 20, 30, 40, 60]:
        for seg in range(int(df_u['timestamp'].max() // w)):
           
            s_u = df_u[(df_u['timestamp'] >= seg*w) & (df_u['timestamp'] < (seg+1)*w)]
            s_i = df_i[(df_i['timestamp'] >= seg*w) & (df_i['timestamp'] < (seg+1)*w)]
            
            # Protect against empty slices crashing the math
            if len(s_u) < 5 or len(s_i) < 5: 
                continue
            
            # Extract
            tm, sn, rm, s1, s2 = calc_time_nonlinear(s_u['rr_interval'].values)
            llf, lhf, llh = calc_lsp_features(s_u['timestamp'].values, s_u['rr_interval'].values)
            flf, fhf, flh = calc_fft_features(s_i['rr_interval'].values)
            
            all_features.append({
                'Participant': filename.split('_')[0], 
                'Label': 1 if "exam" in filename else 0,
                'Window_Size': w, 'MeanRR': tm, 'SDNN': sn, 'RMSSD': rm, 'SD1': s1, 'SD2': s2,
                'LSP_LF': llf, 'LSP_HF': lhf, 'LSP_LFHF': llh,
                'FFT_LF': flf, 'FFT_HF': fhf, 'FFT_LFHF': flh
            })

pd.DataFrame(all_features).to_csv(os.path.join(OUTPUT_DIR, "master_features.csv"), index=False)
print("Extraction Completed")