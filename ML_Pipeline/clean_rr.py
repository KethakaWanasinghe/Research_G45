import pandas as pd
import numpy as np
import os
import glob
from scipy.interpolate import interp1d
from scipy.signal import detrend

RAW_DIR = "raw_data"
UNEVEN_DIR = "cleaned_uneven"
UNIFORM_DIR = "cleaned_uniform"

# Output
for d in [UNEVEN_DIR, UNIFORM_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def clean_and_interpolate_rr(df, filename, threshold=0.20):
    # Sort and extract raw arrays
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    t = df['timestamp'].values
    rr = df['rr_interval'].values
    original_count = len(rr)
    
    # Limits 
    hard_limit_mask = (rr > 300) & (rr < 2000)
    hard_rejected = original_count - hard_limit_mask.sum()
    
    t_bounded = t[hard_limit_mask]
    rr_bounded = rr[hard_limit_mask]
    
    # Median Filter (Artifact)
    df_bounded = pd.DataFrame({'rr': rr_bounded})
    rolling_median = df_bounded['rr'].rolling(window=11, center=True, min_periods=1).median()
    
    lower_bound = rolling_median * (1 - threshold)
    upper_bound = rolling_median * (1 + threshold)
    
    median_mask = ~((df_bounded['rr'] < lower_bound) | (df_bounded['rr'] > upper_bound))
    median_rejected = len(rr_bounded) - median_mask.sum()
    
 
    t_clean = t_bounded[median_mask]
    rr_clean = rr_bounded[median_mask]
    

    total_rejected = hard_rejected + median_rejected
    rejection_rate = (total_rejected / original_count) * 100
    print(f"{filename}: {rejection_rate:.1f}% corrected "
          f"(Hard limits: {hard_rejected}, Median filter: {median_rejected})")


    # DATASET 1 - UNEVEN SIGNAL ( RMSSD, SD1/SD2, LSP)


    f_cubic = interp1d(t_clean, rr_clean, kind='cubic', fill_value="extrapolate")
    rr_uneven_interp = f_cubic(t)

    rr_uneven_interp = np.clip(rr_uneven_interp, 300, 2000)
    
    df_uneven = pd.DataFrame({
        'timestamp': np.round(t, 3),
        'rr_interval': np.round(rr_uneven_interp, 2)
    })


    # DATASET 2 - UNIFORM SIGNAL ( for FFT)

    fs = 4.0 

    t_uniform = np.arange(t[0], t[-1], 1/fs)
    
    # Interpolate
    f_linear = interp1d(t, rr_uneven_interp, kind='linear', fill_value="extrapolate")
    rr_uniform_raw = f_linear(t_uniform)
    

    rr_uniform_detrended = detrend(rr_uniform_raw, type='linear')
    
    df_uniform = pd.DataFrame({
        'timestamp': np.round(t_uniform, 3),
        'rr_interval': np.round(rr_uniform_detrended, 2) 
    })
    
    return df_uneven, df_uniform


file_list = glob.glob(f"{RAW_DIR}/*.csv")
print("Starting Artifact Rejection & Resampling...\n")

for file_path in file_list:
    filename = os.path.basename(file_path)
    df_raw = pd.read_csv(file_path)
    

    df_uneven, df_uniform = clean_and_interpolate_rr(df_raw, filename)
    
    # Save Uneven
    uneven_path = os.path.join(UNEVEN_DIR, filename.replace("_raw", "_uneven"))
    df_uneven.to_csv(uneven_path, index=False)
    
    # Save Uniform
    uniform_path = os.path.join(UNIFORM_DIR, filename.replace("_raw", "_uniform"))
    df_uniform.to_csv(uniform_path, index=False)

print("\n Success!")