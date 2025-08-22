import numpy as np
from typing import List, Tuple, Optional
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import joblib
from .ml_handler import extract_features

# --- STA/LTA wrappers ---

def run_sta_lta_array(data: np.ndarray, sr: float, sta_sec: float, lta_sec: float):
    nsta = max(1, int(sta_sec * sr))
    nlta = max(nsta + 1, int(lta_sec * sr))
    if nlta * 2 > data.size:
        raise ValueError("Trace too short for STA/LTA")
    cft = classic_sta_lta(data.astype(float), nsta, nlta)
    return cft

def detect_onset_indices(cft: np.ndarray, thr_on: float, thr_off: float) -> np.ndarray:
    if cft.size == 0:
        return np.empty((0,2), dtype=int)
    pairs = trigger_onset(cft, thr_on, thr_off)
    return np.array(pairs, dtype=int)

# --- ML sliding window detection ---

def detect_events_with_ml(trace, model_path: str = 'seismic_event_classifier.joblib', window_size_sec: int = 30, step_size_sec: int = 10):
    try:
        model = joblib.load(model_path)
    except Exception as e:  # noqa: BLE001
        print(f'Model load failed: {e}')
        return []
    sr = trace.stats.sampling_rate
    win = int(window_size_sec * sr)
    step = int(step_size_sec * sr)
    if win <= 0 or win > trace.stats.npts:
        print('Invalid window size for trace length')
        return []
    data = trace.data
    events: List[Tuple] = []
    for start in range(0, len(data) - win, step):
        segment = data[start:start+win]
        feats = extract_features(segment, sr)
        X = np.array(list(feats.values()), dtype=float).reshape(1,-1)
        pred = model.predict(X)
        if int(pred[0]) == 1:  # quake
            t0 = trace.stats.starttime + (start / sr)
            t1 = t0 + window_size_sec
            events.append((t0, t1))
    return events
