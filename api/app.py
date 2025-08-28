from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis as _kurtosis, skew as _skew, iqr
from numpy.fft import rfft, rfftfreq
import math

# -------------------- Initialize FastAPI -------------------- #
app = FastAPI(title="HAR Prediction API")

# -------------------- Load trained artifacts -------------------- #
model = joblib.load("svc_model.pkl")
pca = joblib.load("pca_transform.pkl")            # trained on 561 features
label_encoder = joblib.load("label_encoder.pkl")  # encodes activity labels

# -------------------- HAR constants -------------------- #
FS = 50.0      # Hz (UCI HAR)
DT = 1.0 / FS
FC = 0.3       # Hz (cutoff for gravity separation)
WINDOW_N = 128 # samples per window (≈2.56s)

# -------------------- Helpers -------------------- #
def safe_skew(x):
    x = np.asarray(x)
    if np.std(x) == 0:
        return 0.0
    v = float(_skew(x, bias=False))
    if not np.isfinite(v): v = 0.0
    return v

def safe_kurtosis(x):
    x = np.asarray(x)
    if np.std(x) == 0:
        return 0.0
    v = float(_kurtosis(x, bias=False))
    if not np.isfinite(v): v = 0.0
    return v

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    wn = cutoff / nyq
    return butter(order, wn, btype='low', analog=False)

def gravity_body_split(acc_xyz):
    b, a = butter_lowpass(FC, FS, order=4)
    g = np.zeros_like(acc_xyz)
    for i in range(3):
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
    return np.sum(sig**2) / len(sig)

def entropy(sig, bins=30, eps=1e-12):
    hist, _ = np.histogram(sig, bins=bins, density=True)
    p = hist + eps
    p = p / p.sum()
    return -np.sum(p * np.log2(p))

def ar_coeff(sig, order=4):
    sig = sig - np.mean(sig)
    r = np.correlate(sig, sig, mode='full')
    r = r[r.size // 2:] / len(sig)
    R = np.array([[r[abs(i-j)] for j in range(order)] for i in range(order)])
    r_vec = r[1:order+1]
    try:
        a = np.linalg.solve(R, r_vec)
    except np.linalg.LinAlgError:
        a = np.zeros(order)
    return a

def correlation3(x, y, z):
    def corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    return corr(x, y), corr(x, z), corr(y, z)

def mean_freq(mag, freqs):
    p = mag**2
    denom = np.sum(p)
    if denom == 0:
        return 0.0
    return float(np.sum(freqs * p) / denom)

def max_inds(mag):
    return int(np.argmax(mag))

def bands_energy(mag, freqs, bands=8, fmax=None):
    if fmax is None:
        fmax = freqs[-1] if len(freqs) > 0 else FS/2
    band_edges = np.linspace(0, fmax, bands + 1)
    energies = []
    p = mag**2
    for i in range(bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
        energies.append(float(np.sum(p[mask]) / (np.sum(mask) if np.sum(mask) > 0 else 1)))
    return energies


# -------------------- Feature blocks -------------------- #
def basic_stats_1d(sig, prefix):
    feats = {}
    feats[f"{prefix}-mean"] = float(np.mean(sig))
    feats[f"{prefix}-std"]  = float(np.std(sig))
    feats[f"{prefix}-mad"]  = float(np.median(np.abs(sig - np.median(sig))))
    feats[f"{prefix}-max"]  = float(np.max(sig))
    feats[f"{prefix}-min"]  = float(np.min(sig))
    feats[f"{prefix}-energy"] = float(energy(sig))
    feats[f"{prefix}-iqr"]  = float(iqr(sig))
    feats[f"{prefix}-entropy"] = float(entropy(sig))
    feats[f"{prefix}-skewness"] = safe_skew(sig)
    feats[f"{prefix}-kurtosis"] = safe_kurtosis(sig)
    # AR(4)
    ac = ar_coeff(sig, order=4)
    for k in range(4):
        feats[f"{prefix}-arCoeff{(k+1)}"] = float(ac[k])
    return feats

def basic_stats_3d(sig_xyz, base, include_corr=True, include_sma=True):
    """
    Per-axis 1D stats (+ optional correlations + SMA).
    We set include_corr=False for tGravityAcc and tBodyGyroJerk to match 561 total features.
    """
    feats = {}
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
    """ FFT magnitude stats for 1D signal. """
    mag = np.abs(rfft(sig))
    freqs = rfftfreq(len(sig), d=DT)
    feats = {}
    feats[f"{prefix}-mean"] = float(np.mean(mag))
    feats[f"{prefix}-std"]  = float(np.std(mag))
    feats[f"{prefix}-mad"]  = float(np.median(np.abs(mag - np.median(mag))))
    feats[f"{prefix}-max"]  = float(np.max(mag))
    feats[f"{prefix}-min"]  = float(np.min(mag))
    feats[f"{prefix}-energy"] = float(np.sum(mag**2) / len(mag))
    feats[f"{prefix}-iqr"]  = float(iqr(mag))
    feats[f"{prefix}-entropy"] = float(entropy(mag))
    feats[f"{prefix}-skewness"] = safe_skew(mag)
    feats[f"{prefix}-kurtosis"] = safe_kurtosis(mag)
    feats[f"{prefix}-maxInds"] = float(max_inds(mag))
    feats[f"{prefix}-meanFreq"] = float(mean_freq(mag, freqs))
    # bandsEnergy(1..8)
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
    acc_xyz, gyro_xyz: np.array shape (N,3), N≈128
    Returns: ordered list of 561 features and names
    """
    # 1) body/gravity split
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

    # 4) time-domain features
    feats = {}
    # include_corr=True for these three:
    feats.update(basic_stats_3d(tBodyAcc,       "tBodyAcc",       include_corr=True))
    feats.update(basic_stats_3d(tGravityAcc,    "tGravityAcc",    include_corr=False))  # <-- drop 3 correlations
    feats.update(basic_stats_3d(tBodyAccJerk,   "tBodyAccJerk",   include_corr=True))
    feats.update(basic_stats_3d(tBodyGyro,      "tBodyGyro",      include_corr=True))
    feats.update(basic_stats_3d(tBodyGyroJerk,  "tBodyGyroJerk",  include_corr=False))  # <-- drop 3 correlations

    feats.update(mag_stats(tBodyAccMag,      "tBodyAccMag"))
    feats.update(mag_stats(tGravityAccMag,   "tGravityAccMag"))
    feats.update(mag_stats(tBodyAccJerkMag,  "tBodyAccJerkMag"))
    feats.update(mag_stats(tBodyGyroMag,     "tBodyGyroMag"))
    feats.update(mag_stats(tBodyGyroJerkMag, "tBodyGyroJerkMag"))

    # 5) frequency-domain (FFT of body signals & magnitudes)
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
    
    # 6) angles (exact 7)
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

    # order and pack
    names = list(feats.keys())
    vals  = [feats[n] for n in names]

    # Expect exactly 561 features
    if len(vals) != 561:
        raise ValueError(f"Feature count is {len(vals)}; expected 561. Please review feature blocks.")

    return np.array(vals, dtype=float), names




# -------------------- FastAPI Routes -------------------- #
@app.get("/")
def home():
    return {"message": "API is working"}

@app.post("/predict")
async def predict(request: Request):
    """
    Expected JSON:
    {
      "accelerometer": [[ax, ay, az], ... 128 rows ...],
      "gyroscope":     [[gx, gy, gz], ... 128 rows ...]
    }
    """
    try:
        data = await request.json()
        acc = np.array(data["accelerometer"], dtype=float)
        gyr = np.array(data["gyroscope"], dtype=float)

        if acc.shape[1] != 3 or gyr.shape[1] != 3:
            return JSONResponse({"error": "Each sample must have 3 axes for accelerometer and gyroscope."}, status_code=400)
        if len(acc) != len(gyr):
            return JSONResponse({"error": "Accelerometer and gyroscope must have same number of samples."}, status_code=400)
        if len(acc) < 64:
            return JSONResponse({"error": "Too few samples; provide a ~128-sample window at 50Hz."}, status_code=400)

        if len(acc) > WINDOW_N:
            acc = acc[-WINDOW_N:, :]
            gyr = gyr[-WINDOW_N:, :]

        X_561, names = extract_uci_har_561(acc, gyr)
        X_pca = pca.transform(X_561.reshape(1, -1))
        y_pred = model.predict(X_pca)
        label  = label_encoder.inverse_transform(y_pred)[0]

        return {"prediction": label}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



