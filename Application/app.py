from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import random
import sqlite3
import os
import glob
from scipy.signal import lombscargle
import warnings
import sqlite3

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

SYNC_FOLDER = "Watch-data"
os.makedirs(SYNC_FOLDER, exist_ok=True)

print("Loading Production (40s LSP)...")
try:
    rf_model = joblib.load('stress_rf_model_40s.pkl')
    scaler = joblib.load('stress_scaler_40s.pkl')
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR loading models: {e}")

def get_db_connection():
    conn = sqlite3.connect('exam_stress.db')
    conn.row_factory = sqlite3.Row  
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            genre TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()    

def get_music_recommendation(stress_label, user_genre):
    with open('songs.json', 'r', encoding='utf-8') as f:
        db = json.load(f)
    

    if isinstance(db, list):
        db = db[0] if len(db) > 0 else {}

    genres_dict = db.get('genres', {})
    
    # Clean up
    safe_user_genre = str(user_genre).strip().title()
    
    genre_data = None
    actual_genre = safe_user_genre
    
    for key, value in genres_dict.items():
        if safe_user_genre.lower() in key.lower() or key.lower() in safe_user_genre.lower():
            genre_data = value
            actual_genre = key
            break
            
    # 3. FALLBACK GENRE
    if not genre_data:
        actual_genre = list(genres_dict.keys())[0] if genres_dict else "Unknown"
        genre_data = genres_dict.get(actual_genre, {})


    if stress_label == 'High Stress':
        stress_key = "high_stress"
        msg = f"Alert: High Stress detected. Playing calming {actual_genre}."
    elif stress_label == 'Mild Stress':
        stress_key = "moderate_stress"
        msg = f"Note: Mild Stress detected. Playing relaxing {actual_genre} to prevent escalation."
    else:
        stress_key = "moderate_stress" 
        msg = f"Low Stress state (Balanced). Playing focus {actual_genre}."

    tracks = genre_data.get(stress_key, {}).get('tracks', [])
    
    if not tracks and genre_data:
        for k, v in genre_data.items():
            if isinstance(v, dict) and 'tracks' in v and len(v['tracks']) > 0:
                tracks = v['tracks']
                break

    if tracks:
        selected_song = random.choice(tracks)
    else:
        selected_song = {
            "title": "Default LoFi Focus",
            "artist": "Lofi Girl",
            "url": "https://www.youtube.com/embed/jfKfPfyJRdk"
        }
    
    url = selected_song.get('url', '')
    if "watch?v=" in url:
        url = url.replace("watch?v=", "embed/")
        selected_song['url'] = url
        
    return {"message": msg, "track": selected_song}

def extract_features(timestamps, rr_intervals):
    valid = (rr_intervals > 300) & (rr_intervals < 2000)
    t_clean = timestamps[valid]
    rr_clean = rr_intervals[valid]
    
    if len(rr_clean) < 5: return None, None

    mean_rr, sdnn = np.mean(rr_clean), np.std(rr_clean, ddof=1)
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
    
    features = np.array([[mean_rr, sdnn, rmssd, sd1, sd2, lf, hf, lf_hf]])
    return features, lf_hf

@app.route('/login', methods=['POST'])
def login():
    student_id = request.json.get('student_id', '').lower().strip()
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM students WHERE LOWER(student_id) = ?', (student_id,)).fetchone()
    conn.close()
    if user: return jsonify(dict(user))
    return jsonify({"error": "User not found"}), 404

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    student_id = data.get('student_id', '').lower().strip()
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO students (student_id, name, genre) VALUES (?, ?, ?)',
                     (student_id, data['name'], data['genre']))
        conn.commit()
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Student ID exists"}), 400
    finally:
        conn.close()

@app.route('/update_preference', methods=['POST'])
def update_preference():
    data = request.json
    student_id = data.get('student_id', '').lower().strip()
    new_genre = data.get('genre')
    conn = get_db_connection()
    conn.execute('UPDATE students SET genre = ? WHERE student_id = ?', (new_genre, student_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route('/predict_auto', methods=['POST'])
def predict_auto():
    try:
        student_id = request.json.get('student_id', '').lower().strip()

        list_of_files = glob.glob(os.path.join(SYNC_FOLDER, '*.csv'))
        
        if not list_of_files:
            return jsonify({"error": "No smartwatch data found. Please drop a CSV into the Watch-data folder."}), 400
            
        latest_file = max(list_of_files, key=os.path.getctime)

        conn = get_db_connection()
        user = conn.execute('SELECT genre FROM students WHERE student_id = ?', (student_id,)).fetchone()
        conn.close()
        
        if not user: return jsonify({"error": "User not found"}), 404
        user_genre = user['genre']

        df = pd.read_csv(latest_file)
        time_col = 'timestamp_sec' if 'timestamp_sec' in df.columns else 'timestamp'
        rr_col = 'rr_ms' if 'rr_ms' in df.columns else 'rr_interval'
        
        if time_col not in df.columns or rr_col not in df.columns:
             return jsonify({"error": f"Invalid CSV. Requires '{time_col}' and '{rr_col}'."}), 400

        features_array, lsp_lfhf = extract_features(df[time_col].values, df[rr_col].values)
        if features_array is None: return jsonify({"error": "CSV contains too few valid heartbeats."}), 400

        mean_rr, sdnn, rmssd, sd1, sd2, lf, hf, lfhf_val = features_array[0]
        bpm = 60000 / mean_rr if mean_rr > 0 else 0

        features_scaled = scaler.transform(features_array)
        prediction_binary = rf_model.predict(features_scaled)[0] 
        
        if prediction_binary == 0:
            final_label = "Low Stress"
        else:
            final_label = "High Stress" if lsp_lfhf >= 3.0 else "Mild Stress"

        music_data = get_music_recommendation(final_label, user_genre)

        return jsonify({
            "stress_level": final_label,
            "music_recommendation": music_data,
            "stats": {
                "bpm": f"{bpm:.0f}",
                "rmssd": f"{rmssd:.1f} ms",
                "lfhf": f"{lsp_lfhf:.2f}"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)