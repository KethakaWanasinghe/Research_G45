
# Exam Stress Detector: HRV & Machine Learning Web Application

This project is a complete end-to-end Machine Learning pipeline and full-stack web application designed to detect acute cognitive stress during university exams. It utilizes real human R-R intervals (heartbeats) captured via Apple Watch PPG sensors, processes the biological data using advanced signal processing (Lomb-Scargle Periodogram), and classifies the stress state using a Random Forest algorithm.

The final output is a production-ready React web application that acts as an "Intelligent Music Therapy Engine," prescribing specific therapeutic music based on the student's real-time physiological stress intensity.

---

## 📂 Project Structure

```text
FINAL_PROJECT/
├── raw_data/                  # Real Apple Watch RR intervals (Relax vs Exam)
├── cleaned_uneven/            # Output: Artifact-free data with natural time gaps
├── cleaned_uniform/           # Output: Interpolated data (4.0Hz) for FFT comparison
├── features/                  # Output: Extracted time/frequency features
├── models/                    # Output: Serialized ML models (.pkl)
├── thesis_plots/              # Output: Generated academic visualizations
│
├── clean_rr.py                # Pipeline Phase 1: Artifact Rejection & Detrending
├── extract_features.py        # Pipeline Phase 2: Feature Engineering (LSP & FFT)
├── evaluate_ml.py             # Pipeline Phase 3: Statistical & ML Validation (LOSO)
├── generate_thesis_plots.py   # Pipeline Phase 4: High-Res Academic Visualizations
├── save_final_model.py        # Pipeline Phase 5: Train & Save Production Model
│
└── Application/               # Phase 6: Production Web Environment
    ├── app.py                 # Flask Backend & Real-time Inference API
    ├── exam_stress.db         # SQLite User Database
    ├── songs.json             # Music Therapy Logic mapping
    ├── stress_rf_model_40s.pkl# Production AI Brain (Copied from models/)
    ├── stress_scaler_40s.pkl  # Production Scaler (Copied from models/)
    └── frontend/              # React.js Glassmorphism UI

```

---

## ⚙️ Prerequisites

Before running the pipeline, ensure you have the following installed on your system:

1. **Python 3.8+**
2. **Node.js & npm** (For the React frontend)
3. **Python Libraries:** Install the required dependencies using your terminal:
```bash
pip install pandas numpy scipy scikit-learn flask flask-cors joblib matplotlib seaborn

```



---

## 🚀 Step-by-Step Execution Guide

To reproduce the entire project from raw data to the final web application, run the following scripts in order from the root directory.

### Phase 1: Data Preprocessing & Artifact Rejection

This script reads the raw Apple Watch CSV files from the `raw_data/` folder. It applies a strict 300ms-2000ms biological hard limit and an 11-beat rolling median filter to remove sensor motion artifacts and ectopic beats.

```bash
python clean_rr.py

```

* **Output:** Creates two new folders (`cleaned_uneven/` and `cleaned_uniform/`) containing the filtered time-series data.

### Phase 2: Feature Extraction

This script dynamically segments the cleaned data into 10s, 20s, 30s, 40s, and 60s windows. It calculates Time-Domain features (RMSSD, SDNN, SD1, SD2) and Frequency-Domain features using both the Lomb-Scargle Periodogram (LSP) and Fast Fourier Transform (FFT).

```bash
python extract_features.py

```

* **Output:** Generates `features/master_features.csv` which contains the complete mathematical matrix for ML training.

### Phase 3: Machine Learning & Statistical Validation

This script proves the clinical validity of the data using Mann-Whitney U tests and evaluates the Machine Learning models using strict Leave-One-Subject-Out (LOSO) cross-validation.

```bash
python evaluate_ml.py

```

* **Output:** Terminal printouts proving that the 40-second window utilizing LSP features and a Random Forest Classifier is the optimal configuration (92.38% Accuracy).

### Phase 4: Generate Academic Visualizations (Optional)

Generates high-resolution graphs for the thesis report, including Optimization Curves, Feature Importance (Explainable AI), and Confusion Matrices.

```bash
python generate_thesis_plots.py

```

* **Output:** Creates the `thesis_plots/` folder containing publication-ready `.png` images.

### Phase 5: Model Finalization

Trains the ultimate Random Forest model on the 40-second data and serializes it for production deployment.

```bash
python save_final_model.py

```

* **Output:** Creates the `models/` folder containing `stress_rf_model_40s.pkl` and `stress_scaler_40s.pkl`.
* ⚠️ **CRITICAL STEP:** Manually copy both of these `.pkl` files and paste them into the `Application/` folder, replacing any existing ones!

---

## 🌐 Running the Production Web Application

Once the ML pipeline is complete and the models are moved to the `Application/` folder, you can boot up the real-time detector.

### 1. Start the Flask Backend (Terminal 1)

Open a terminal, navigate to the `Application` folder, and start the Python server.

```bash
cd Application
python app.py

```

*The backend is now listening on `http://127.0.0.1:5000`.*

### 2. Start the React Frontend (Terminal 2)

Open a **new** terminal, navigate to the `frontend` folder, install the necessary node modules (first time only), and start the Vite development server.

```bash
cd Application/frontend
npm install
npm run dev

```

*The frontend is now available at `http://localhost:5173`.*

---

## 🧪 How to use the Web App (Demo)

1. Open the frontend URL in your browser.
2. If you are a new user, click **Enter** to be redirected to the Onboarding screen to set your Name and Music Genre preference.
3. If you have an account, type your Student ID (e.g., `ICT/21/001`) and login. (IDs are automatically sanitized for case-insensitivity).
4. On the Dashboard, upload a smartwatch CSV file (ensure it has `timestamp_sec` and `rr_ms` columns).
5. Click **Initialize AI Analysis**. The system will perform real-time extraction, scale the features, and predict the biological stress state, immediately initiating dynamic background rendering, breathing pacing, and a YouTube iFrame with the therapeutic music prescription.

```

```