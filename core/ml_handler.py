import numpy as np
from scipy.fft import rfft, rfftfreq
from typing import Dict, Any, List

# Fixed feature order to guarantee consistent vector layout when using dict.values().
_FEATURE_ORDER: List[str] = [
    'mean', 'median', 'min', 'max', 'range', 'std_dev', 'mean_abs', 'rms', 'energy',
    'skewness', 'kurtosis', 'zero_crossing_rate', 'dominant_freq', 'spectral_centroid',
    'peak_to_rms', 'crest_factor'
]

def extract_features(data_window: np.ndarray, sampling_rate: float) -> Dict[str, Any]:
    """Extract deterministic, ordered features from a 1D seismic window.

    Returns dict with keys in _FEATURE_ORDER (missing values filled with 0.0).
    Additional notes:
      - kurtosis here is raw 4th standardized moment (not excess -> subtract 3 if desired).
      - zero_crossing_rate normalized by window duration (approx len/sr).
    """
    feats: Dict[str, Any] = {k: 0.0 for k in _FEATURE_ORDER}
    n = int(data_window.size)
    if n == 0 or sampling_rate <= 0:
        return feats

    x = data_window.astype(float, copy=False)
    mean = float(np.mean(x))
    median = float(np.median(x))
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    rng = x_max - x_min
    std = float(np.std(x))
    mean_abs = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    energy = float(np.sum(x**2))
    if std > 0:
        z = (x - mean) / std
        skewness = float(np.mean(z**3))
        kurt = float(np.mean(z**4))
    else:
        skewness = 0.0
        kurt = 0.0
    # Zero crossing rate
    if n > 1:
        zc = float(((x[:-1] * x[1:]) < 0).sum())
        duration = n / sampling_rate
        zcr = zc / duration if duration > 0 else 0.0
    else:
        zcr = 0.0
    peak_to_rms = (x_max / rms) if rms > 0 else 0.0
    crest_factor = (x_max / mean_abs) if mean_abs > 0 else 0.0

    # Frequency domain
    if n > 1:
        # Cast to ndarray to avoid potential typing issues with scipy stubs
        spec = np.asarray(rfft(x - mean))
        freqs = rfftfreq(n, 1.0 / sampling_rate)
        power = np.abs(spec)**2
        total_power = float(power.sum())
        if total_power > 0:
            dominant_freq = float(freqs[np.argmax(power)])
            spectral_centroid = float(np.sum(freqs * power) / total_power)
        else:
            dominant_freq = 0.0
            spectral_centroid = 0.0
    else:
        dominant_freq = 0.0
        spectral_centroid = 0.0

    feats.update({
        'mean': mean,
        'median': median,
        'min': x_min,
        'max': x_max,
        'range': rng,
        'std_dev': std,
        'mean_abs': mean_abs,
        'rms': rms,
        'energy': energy,
        'skewness': skewness,
        'kurtosis': kurt,
        'zero_crossing_rate': zcr,
        'dominant_freq': dominant_freq,
        'spectral_centroid': spectral_centroid,
        'peak_to_rms': peak_to_rms,
        'crest_factor': crest_factor,
    })
    return feats
