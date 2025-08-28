from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis as _kurtosis, skew as _skew, iqr
from numpy.fft import rfft, rfftfreq
import math
import pandas as pd
# -------------------- Initialize FastAPI -------------------- #
app = FastAPI(title="HAR Prediction API")

# -------------------- Load trained artifacts -------------------- #
model = joblib.load("svc_model.pkl")
pca = joblib.load("pca_transform.pkl")            # trained on 561 features
label_encoder = joblib.load("label_encoder.pkl")  # encodes activity labels
model = joblib.load("walking_classifier.pkl")

# -------------------- HAR constants -------------------- #
FS = 50.0      # Hz (UCI HAR)
DT = 1.0 / FS
FC = 0.3       # Hz (cutoff for gravity separation)
WINDOW_N = 128 # samples per window (≈2.56s)

# -------------------- Helpers -------------------- #
# -------------------- Safe stats -------------------- #
def safe_skew(x):
    x = np.asarray(x, dtype=float)
    if np.std(x) == 0: return 0.0
    v = float(_skew(x, bias=False))
    return v if np.isfinite(v) else 0.0

def safe_kurtosis(x):
    x = np.asarray(x, dtype=float)
    if np.std(x) == 0: return 0.0
    v = float(_kurtosis(x, bias=False))
    return v if np.isfinite(v) else 0.0

# -------------------- DSP helpers -------------------- #
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    wn = cutoff / nyq
    return butter(order, wn, btype='low', analog=False)

def gravity_body_split(acc_xyz):
    """Split acceleration into gravity and body via 4th-order Butterworth low-pass fc=0.3Hz."""
    b, a = butter_lowpass(FC, FS, order=4)
    g = np.zeros_like(acc_xyz)
    for i in range(3):
        # filtfilt requires length > order; our window is 128 so fine
        g[:, i] = filtfilt(b, a, acc_xyz[:, i])
    body = acc_xyz - g
    return body, g

def jerk(sig_xyz):
    d = np.diff(sig_xyz, axis=0) / DT
    d = np.vstack([d[0:1, :], d])
    return d

def magnitude(sig_xyz):
    return np.sqrt(np.sum(sig_xyz**2, axis=1))

def sma(sig_xyz):
    return np.mean(np.sum(np.abs(sig_xyz), axis=1))

def energy(sig):
    sig = np.asarray(sig, dtype=float)
    if len(sig) == 0: return 0.0
    return float(np.sum(sig**2) / len(sig))

def entropy_custom(sig, bins=30, eps=1e-12):
    sig = np.asarray(sig, dtype=float)
    if len(sig) == 0:
        return 0.0
    hist, _ = np.histogram(sig, bins=bins, density=True)
    p = (hist + eps) / np.sum(hist + eps)
    return float(-np.sum(p * np.log2(p)))

def ar_coeff(sig, order=4):
    """Solve Yule-Walker style AR coefficients with plain linear system (robust to singular)."""
    sig = np.asarray(sig, dtype=float)
    if len(sig) < order + 1 or np.std(sig) == 0.0:
        return np.zeros(order)
    sig = sig - np.mean(sig)
    r = np.correlate(sig, sig, mode='full')
    r = r[r.size // 2:] / len(sig)
    # Build Toeplitz-like R matrix manually (order x order)
    R = np.empty((order, order), dtype=float)
    for i in range(order):
        for j in range(order):
            R[i, j] = r[abs(i - j)]
    r_vec = r[1:order+1]
    try:
        a = np.linalg.solve(R, r_vec)
    except np.linalg.LinAlgError:
        a = np.zeros(order)
    return a

def correlation3(x, y, z):
    def corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    return corr(x,y), corr(x,z), corr(y,z)

def mean_freq(mag, freqs):
    p = mag**2
    denom = np.sum(p)
    return float(np.sum(freqs * p) / denom) if denom != 0 else 0.0

def max_inds(mag):
    if len(mag) == 0:
        return 0
    return int(np.argmax(mag))

def bands_energy(mag, freqs, bands=8, fmax=None):
    if len(freqs) == 0:
        return [0.0] * bands
    if fmax is None:
        fmax = freqs[-1]
    band_edges = np.linspace(0, fmax, bands + 1)
    energies = []
    p = mag**2
    for i in range(bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
        denom = np.sum(mask)
        energies.append(float(np.sum(p[mask]) / (denom if denom > 0 else 1.0)))
    return energies

# -------------------- Feature blocks -------------------- #
def basic_stats_1d(sig, prefix):
    feats = {}
    sig = np.asarray(sig, dtype=float)
    feats[f"{prefix}-mean"] = float(np.mean(sig))
    feats[f"{prefix}-std"]  = float(np.std(sig))
    feats[f"{prefix}-mad"]  = float(np.median(np.abs(sig - np.median(sig))))
    feats[f"{prefix}-max"]  = float(np.max(sig))
    feats[f"{prefix}-min"]  = float(np.min(sig))
    feats[f"{prefix}-energy"] = float(energy(sig))
    feats[f"{prefix}-iqr"]  = float(iqr(sig)) if len(sig) > 0 else 0.0
    feats[f"{prefix}-entropy"] = float(entropy_custom(sig))
    feats[f"{prefix}-skewness"] = safe_skew(sig)
    feats[f"{prefix}-kurtosis"] = safe_kurtosis(sig)
    ac = ar_coeff(sig, order=4)
    for k in range(4):
        feats[f"{prefix}-arCoeff{(k+1)}"] = float(ac[k])
    return feats

def basic_stats_3d(sig_xyz, base, include_corr=True, include_sma=True):
    feats = {}
    sig_xyz = np.asarray(sig_xyz, dtype=float)
    axes = ['X', 'Y', 'Z']
    for i, ax in enumerate(axes):
        feats.update(basic_stats_1d(sig_xyz[:, i], f"{base}-{ax}"))
    if include_corr:
        cxy, cxz, cyz = correlation3(sig_xyz[:, 0], sig_xyz[:, 1], sig_xyz[:, 2])
        feats[f"{base}-correlation(X,Y)"] = float(cxy)
        feats[f"{base}-correlation(X,Z)"] = float(cxz)
        feats[f"{base}-correlation(Y,Z)"] = float(cyz)
    if include_sma:
        feats[f"{base}-sma"] = float(sma(sig_xyz))
    return feats

def mag_stats(sig_mag, base):
    return basic_stats_1d(sig_mag, base)

def freq_stats_1d(sig, prefix):
    sig = np.asarray(sig, dtype=float)
    # compute FFT magnitudes (rfft works similarly; use rfft for real input)
    mag = np.abs(rfft(sig))
    freqs = rfftfreq(len(sig), d=DT)
    feats = {}
    feats[f"{prefix}-mean"] = float(np.mean(mag))
    feats[f"{prefix}-std"]  = float(np.std(mag))
    feats[f"{prefix}-mad"]  = float(np.median(np.abs(mag - np.median(mag))))
    feats[f"{prefix}-max"]  = float(np.max(mag))
    feats[f"{prefix}-min"]  = float(np.min(mag))
    feats[f"{prefix}-energy"] = float(np.sum(mag**2) / len(mag)) if len(mag) > 0 else 0.0
    feats[f"{prefix}-iqr"]  = float(iqr(mag)) if len(mag) > 0 else 0.0
    feats[f"{prefix}-entropy"] = float(entropy_custom(mag))
    feats[f"{prefix}-skewness"] = safe_skew(mag)
    feats[f"{prefix}-kurtosis"] = safe_kurtosis(mag)
    feats[f"{prefix}-maxInds"] = float(max_inds(mag))
    feats[f"{prefix}-meanFreq"] = float(mean_freq(mag, freqs))
    be = bands_energy(mag, freqs, bands=8, fmax=freqs[-1] if len(freqs)>0 else FS/2)
    for i, e in enumerate(be, 1):
        feats[f"{prefix}-bandsEnergy({i})"] = float(e)
    return feats

def freq_stats_3d(sig_xyz, base):
    feats = {}
    axes = ['X', 'Y', 'Z']
    for i, ax in enumerate(axes):
        feats.update(freq_stats_1d(sig_xyz[:, i], f"{base}-{ax}"))
    return feats

def freq_mag_stats(sig_mag, base):
    return freq_stats_1d(sig_mag, base)

# -------------------- Full 561-feature extractor -------------------- #
def extract_uci_har_561(acc_xyz, gyro_xyz):
    """
    Generate 561 features from 128 accelerometer and 128 gyroscope samples 
    similar to UCI HAR Dataset.
    Returns:
        np.array(features) shape (561,), and list(feature_names) in that same order.
    """
    # Input validation
    acc_xyz = np.asarray(acc_xyz, dtype=float)
    gyro_xyz = np.asarray(gyro_xyz, dtype=float)
    if acc_xyz.shape != (WINDOW_N, 3) or gyro_xyz.shape != (WINDOW_N, 3):
        raise ValueError(f"acc and gyro must be shape ({WINDOW_N},3). Got {acc_xyz.shape} and {gyro_xyz.shape}")

    # 1) gravity/body split on accelerometer
    tBodyAcc, tGravityAcc = gravity_body_split(acc_xyz)

    # 2) jerk signals
    tBodyAccJerk  = jerk(tBodyAcc)
    tBodyGyro     = gyro_xyz.copy()
    tBodyGyroJerk = jerk(tBodyGyro)

    # 3) magnitudes
    tBodyAccMag        = magnitude(tBodyAcc)
    tGravityAccMag     = magnitude(tGravityAcc)
    tBodyAccJerkMag    = magnitude(tBodyAccJerk)
    tBodyGyroMag       = magnitude(tBodyGyro)
    tBodyGyroJerkMag   = magnitude(tBodyGyroJerk)

    feats = {}
    # Time-domain 3D blocks
    feats.update(basic_stats_3d(tBodyAcc,       "tBodyAcc",       include_corr=True))
    feats.update(basic_stats_3d(tGravityAcc,    "tGravityAcc",    include_corr=False))
    feats.update(basic_stats_3d(tBodyAccJerk,   "tBodyAccJerk",   include_corr=True))
    feats.update(basic_stats_3d(tBodyGyro,      "tBodyGyro",      include_corr=True))
    feats.update(basic_stats_3d(tBodyGyroJerk,  "tBodyGyroJerk",  include_corr=False))

    # Time-domain magnitudes
    feats.update(mag_stats(tBodyAccMag,      "tBodyAccMag"))
    feats.update(mag_stats(tGravityAccMag,   "tGravityAccMag"))
    feats.update(mag_stats(tBodyAccJerkMag,  "tBodyAccJerkMag"))
    feats.update(mag_stats(tBodyGyroMag,     "tBodyGyroMag"))
    feats.update(mag_stats(tBodyGyroJerkMag, "tBodyGyroJerkMag"))

    # Frequency-domain: using original signals (as UCI does)
    fBodyAcc        = tBodyAcc
    fBodyAccJerk    = tBodyAccJerk
    fBodyGyro       = tBodyGyro
    fBodyAccMag     = tBodyAccMag
    fBodyAccJerkMag = tBodyAccJerkMag
    fBodyGyroMag    = tBodyGyroMag
    fBodyGyroJerkMag= tBodyGyroJerkMag

    feats.update(freq_stats_3d(fBodyAcc,        "fBodyAcc"))
    feats.update(freq_stats_3d(fBodyAccJerk,    "fBodyAccJerk"))
    feats.update(freq_stats_3d(fBodyGyro,       "fBodyGyro"))

    feats.update(freq_mag_stats(fBodyAccMag,     "fBodyAccMag"))
    feats.update(freq_mag_stats(fBodyAccJerkMag, "fBodyAccJerkMag"))
    feats.update(freq_mag_stats(fBodyGyroMag,    "fBodyGyroMag"))
    feats.update(freq_mag_stats(fBodyGyroJerkMag,"fBodyGyroJerkMag"))
    
    # Angle features
    def mean_vec(sig_xyz):
        return np.array([np.mean(sig_xyz[:,0]), np.mean(sig_xyz[:,1]), np.mean(sig_xyz[:,2])])

    def angle(v1, v2, eps=1e-12):
        a = v1 / (np.linalg.norm(v1) + eps)
        b = v2 / (np.linalg.norm(v2) + eps)
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.arccos(dot))

    gravity_mean      = mean_vec(tGravityAcc)
    bodyacc_mean      = mean_vec(tBodyAcc)
    bodyaccjerk_mean  = mean_vec(tBodyAccJerk)
    bodygyro_mean     = mean_vec(tBodyGyro)
    bodygyrojerk_mean = mean_vec(tBodyGyroJerk)

    feats["angle(tBodyAccMean,gravity)"]       = angle(bodyacc_mean, gravity_mean)
    feats["angle(tBodyAccJerkMean,gravity)"]   = angle(bodyaccjerk_mean, gravity_mean)
    feats["angle(tBodyGyroMean,gravity)"]      = angle(bodygyro_mean, gravity_mean)
    feats["angle(tBodyGyroJerkMean,gravity)"]  = angle(bodygyrojerk_mean, gravity_mean)
    feats["angle(X,gravityMean)"]              = angle(np.array([1,0,0]), gravity_mean)
    feats["angle(Y,gravityMean)"]              = angle(np.array([0,1,0]), gravity_mean)
    feats["angle(Z,gravityMean)"]              = angle(np.array([0,0,1]), gravity_mean)

    # produce ordered lists (the original UCI features.txt order is followed by the blocks above)
    names = list(feats.keys())
    vals  = [feats[n] for n in names]

    # Final check: UCI HAR expects 561 features
    if len(vals) != 561:
        raise ValueError(f"Feature count is {len(vals)}; expected 561. Please review feature blocks.")

    features_array = np.array(vals, dtype=float)
    print(features_array)
    # Replace NaN/infs with numbers (0)
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

    return features_array, names

# -------------------- Flask routes -------------------- #
@app.route('/')
def home():
    return {"message": "API is working"}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected JSON object."}), 400

        acc = np.array(data.get("accelerometer"))
        gyr = np.array(data.get("gyroscope"))

        # Basic validation
        if not (isinstance(acc, np.ndarray) and isinstance(gyr, np.ndarray)):
            return jsonify({"error": "accelerometer and gyroscope must be arrays"}), 400
        if acc.ndim != 2 or gyr.ndim != 2:
            return jsonify({"error": "accelerometer and gyroscope must be 2D arrays"}), 400
        if acc.shape[1] != 3 or gyr.shape[1] != 3:
            return jsonify({"error": "Each sample must have 3 axes"}), 400
        if acc.shape[0] != gyr.shape[0]:
            return jsonify({"error": "Accelerometer and gyroscope length mismatch"}), 400
        if acc.shape[0] < 64:
            return jsonify({"error": "Too few samples (need ~128)"}), 400

        # Window/pad to fixed size 128
        WINDOW_N = 128
        if acc.shape[0] > WINDOW_N:
            acc = acc[-WINDOW_N:, :]
            gyr = gyr[-WINDOW_N:, :]
        elif acc.shape[0] < WINDOW_N:
            pad_size = WINDOW_N - acc.shape[0]
            acc = np.pad(acc, ((0, pad_size), (0, 0)), mode="constant")
            gyr = np.pad(gyr, ((0, pad_size), (0, 0)), mode="constant")

        # Extract 561 features
        X_561, names = extract_uci_har_561(acc, gyr)

        # Reorder features to match training dataset
        feats_dict = dict(zip(names, X_561))
        ordered_feats = np.array([feats_dict.get(f, 0.0) for f in feature_names], dtype=float)
        ordered_feats = np.nan_to_num(ordered_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Sanity check
        if ordered_feats.shape[0] != len(feature_names):
            return jsonify({"error": f"Feature mismatch: got {ordered_feats.shape[0]}, expected {len(feature_names)}"}), 500

        # Apply PCA + Predict
        X_pca = pca.transform(ordered_feats.reshape(1, -1))
        y_pred = model.predict(X_pca)
        label = label_encoder.inverse_transform(y_pred)[0]

        return jsonify({"prediction": label})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    

@app.post("/WalkingPredict")
def predict_walking():
    data = request.get_json(force=True)
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return {"Predicted Action": prediction[0]}