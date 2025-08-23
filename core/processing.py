import numpy as np
from typing import List, Tuple, Optional
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import joblib
from .ml_handler import extract_features
from typing import Any

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

def detect_events_with_ml(trace, model_path: str = 'seismic_event_classifier.joblib', window_size_sec: int = 30, step_size_sec: int = 10, probability: bool = True):
    """Sliding window inference returning list of (t0, t1, class_idx, class_label, prob).

    Supports artifact dictionary saved by updated train_model (model+meta) or raw sklearn model.
    """
    try:
        artifact = joblib.load(model_path)
    except Exception as e:  # noqa: BLE001
        print(f'Model load failed: {e}')
        return []
    if isinstance(artifact, dict) and 'model' in artifact:
        model: Any = artifact['model']  # type: ignore[assignment]
        label_map = artifact.get('label_map', {0: 'noise'})
        # invert mapping if stored in forward direction
        if any(isinstance(k,str) for k in label_map.keys()):
            inv_label_map = {v:k for k,v in label_map.items()}
        else:
            inv_label_map = label_map
        feature_order = artifact.get('feature_order', None)
    else:
        model = artifact  # type: ignore[assignment]
        inv_label_map = {0: 'noise', 1: 'event'}
        feature_order = None

    sr = trace.stats.sampling_rate
    win = int(window_size_sec * sr)
    step = int(step_size_sec * sr)
    if win <= 0 or win > trace.stats.npts:
        print('Invalid window size for trace length')
        return []
    data = trace.data
    detections: List[Tuple] = []
    for start in range(0, len(data) - win, step):
        segment = data[start:start+win]
        feats = extract_features(segment, sr)
        # Reorder features if artifact provided order
        if feature_order:
            feat_vec = [feats.get(k, 0.0) for k in feature_order]
        else:
            feat_vec = list(feats.values())
        X = np.array(feat_vec, dtype=float).reshape(1,-1)
        pred = model.predict(X)
        cls_idx = int(pred[0])
        # obtain probability if available
        prob_val = None
        if probability and hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(X)
                if probs.shape[1] > cls_idx:
                    prob_val = float(probs[0, cls_idx])
            except Exception:
                prob_val = None
        label = inv_label_map.get(cls_idx, str(cls_idx))
        if cls_idx != 0:  # skip noise windows
            t0 = trace.stats.starttime + (start / sr)
            t1 = t0 + window_size_sec
            detections.append((t0, t1, cls_idx, label, prob_val))
    return detections
