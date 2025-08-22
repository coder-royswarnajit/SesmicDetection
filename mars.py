import os
import argparse
from typing import List, Optional, Tuple
import kagglehub 
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DATASET = "ahmedsamir1598/space-apps-2024-seismic-detection"
KAGGLE_NOTEBOOK_BASE = "/kaggle/input/space-apps-2024-seismic-detection/space_apps_2024_seismic_detection/data/mars/training/data"  # path used inside Kaggle kernels


def download_dataset(dataset: str = DEFAULT_DATASET) -> str:
    """Download (or reuse cached) Kaggle dataset via kagglehub and return local path."""
    if kagglehub is None:
        raise RuntimeError("kagglehub is not installed. Add it to requirements or install manually.")
    path = kagglehub.dataset_download(dataset)
    print(f"Dataset downloaded / located at: {path}")
    return path


def guess_data_root(user_root: Optional[str], force_download: bool, dataset: str) -> str:
    """Resolve a root directory containing .mseed files.

    Priority:
      1. user_root if provided and exists
      2. Kaggle notebook default path (if exists – when running in Kaggle env)
      3. Download via kagglehub
    """
    if user_root and os.path.isdir(user_root):
        return user_root

    if os.path.isdir(KAGGLE_NOTEBOOK_BASE): 
        return KAGGLE_NOTEBOOK_BASE

    if force_download or True:  # always try to ensure we have data locally
        return download_dataset(dataset)


def collect_mseed_files(root: str, limit: Optional[int] = None, planet: str = "mars") -> List[str]:
    """Collect MiniSEED files under *root* optionally filtering for a specific planet substring.

    The combined dataset 'ahmedsamir1598/space-apps-2024-seismic-detection' contains both
    Mars and Moon (lunar) data under nested directories. We filter by default for 'mars'
    so this script only operates on Martian traces. Adjust *planet* if reuse is desired.
    """
    planet_lower = planet.lower() if planet else ""
    matches: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        # Skip directories that don't contain the planet substring if planet specified
        if planet_lower and planet_lower not in dirpath.replace("\\", "/").lower():
            continue
        for f in filenames:
            if f.lower().endswith((".mseed", ".miniseed")):
                full = os.path.join(dirpath, f)
                matches.append(full)
                if limit and len(matches) >= limit:
                    return sorted(matches)
    return sorted(matches)


def choose_file(files: List[str], index: int) -> str:
    if not files:
        raise FileNotFoundError("No .mseed files found.")
    if index < 0 or index >= len(files):
        raise IndexError(f"File index {index} out of range (0..{len(files)-1}).")
    return files[index]


def load_stream(mseed_path: str):
    print(f"Reading MiniSEED: {mseed_path}")
    st = read(mseed_path)
    print(st)
    return st


def apply_bandpass(st, freqmin: float, freqmax: float):
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    return st_filt


def run_sta_lta(tr, sta_sec: float, lta_sec: float, thr_on: float, thr_off: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute classic STA/LTA characteristic function and trigger on/off indices.

    Returns (cft, triggers) where triggers is Nx2 array of [on_index, off_index].
    """
    from obspy.signal.trigger import classic_sta_lta, trigger_onset  # local import

    df = tr.stats.sampling_rate
    nsta = int(sta_sec * df)
    nlta = int(lta_sec * df)
    if nsta <= 0 or nlta <= 0:
        raise ValueError("STA/LTA window lengths must be > 0.")
    if nsta >= nlta:
        raise ValueError("STA window must be shorter than LTA window.")
    if nlta * 2 > tr.stats.npts:
        raise ValueError("Trace too short relative to LTA window.")

    data = tr.data.astype(float)
    cft = classic_sta_lta(data, nsta, nlta)
    on_off = np.array(trigger_onset(cft, thr_on, thr_off)) if cft.size else np.empty((0, 2), dtype=int)
    return cft, on_off


def plot_sta_lta(tr, cft: np.ndarray, on_off: np.ndarray, out: Optional[str] = None, thr_on: float = 0.0, thr_off: float = 0.0):
    """Plot STA/LTA characteristic function with trigger lines (optional save)."""
    times = tr.times()
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(times, cft, lw=0.8)
    ax.set_xlim(times.min(), times.max())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Characteristic Function')
    ax.set_title(f'STA/LTA (STA={thr_on if thr_on else "?"} / LTA={thr_off if thr_off else "?"})')
    # Trigger thresholds (horizontal lines)
    if thr_on:
        ax.axhline(thr_on, color='red', linestyle='--', label='thr_on')
    if thr_off:
        ax.axhline(thr_off, color='purple', linestyle='--', label='thr_off')
    # Mark on/off verticals
    for idx, pair in enumerate(on_off):
        on_t = times[pair[0]]
        off_t = times[pair[1]] if pair[1] < len(times) else times[-1]
        ax.axvline(on_t, color='red', alpha=0.6)
        ax.axvline(off_t, color='purple', alpha=0.4)
    if thr_on or thr_off or len(on_off):
        ax.legend(loc='upper right')
    fig.tight_layout()
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f'STA/LTA plot saved to {out}')
        plt.close(fig)
    else:
        plt.show()


def export_triggers(tr, on_off: np.ndarray, catalog_out: str, original_filename: str):
    """Export trigger on-times to a CSV catalog (start of events only)."""
    if on_off.size == 0:
        print('No triggers to export.')
        return
    start_dt = tr.stats.starttime.datetime
    times = tr.times()
    on_times_abs = []
    on_times_rel = []
    filenames = []
    for row in on_off:
        on_idx = row[0]
        if on_idx >= len(times):
            continue
        rel_sec = float(times[on_idx])
        abs_dt = start_dt + pd.to_timedelta(rel_sec, unit='s')
        on_times_rel.append(rel_sec)
        on_times_abs.append(abs_dt.strftime('%Y-%m-%dT%H:%M:%S.%f'))
        filenames.append(os.path.basename(original_filename))
    df = pd.DataFrame({
        'filename': filenames,
        'time_abs(%Y-%m-%dT%H:%M:%S.%f)': on_times_abs,
        'time_rel(sec)': on_times_rel,
    })
    os.makedirs(os.path.dirname(catalog_out) or '.', exist_ok=True)
    df.to_csv(catalog_out, index=False)
    print(f'Exported {len(df)} trigger(s) to {catalog_out}')


def plot_trace(st, trace_index: int = 0, show: bool = True, out: Optional[str] = None, abs_arrival: Optional[str] = None,
               bandpass: bool = False, freqmin: float = 0.5, freqmax: float = 1.0, spectrogram: bool = False,
               spec_out: Optional[str] = None, sta_lta: bool = False, sta_sec: float = 120.0, lta_sec: float = 600.0,
               thr_on: float = 4.0, thr_off: float = 1.5, trigger_plot_out: Optional[str] = None,
               catalog_out: Optional[str] = None):
    from scipy import signal  # local import

    if trace_index < 0 or trace_index >= len(st):
        raise IndexError(f"Trace index {trace_index} out of range (0..{len(st)-1}).")

    base_st = st
    filt_label = ""
    if bandpass:
        base_st = apply_bandpass(st, freqmin, freqmax)
        filt_label = f" (BP {freqmin}-{freqmax} Hz)"

    tr = base_st[trace_index]

    # STA/LTA detection (before potential plotting of waveform/spectrogram)
    cft = None
    on_off = np.empty((0, 2), dtype=int)
    if sta_lta:
        try:
            cft, on_off = run_sta_lta(tr, sta_sec, lta_sec, thr_on, thr_off)
            if trigger_plot_out or spectrogram is False and not out:
                # If user will look at waveform only (no spectrogram) and no explicit trigger plot file
                # we can later overlay triggers on waveform.
                pass
            if trigger_plot_out:
                plot_sta_lta(tr, cft, on_off, out=trigger_plot_out, thr_on=thr_on, thr_off=thr_off)
            if catalog_out:
                export_triggers(tr, on_off, catalog_out, tr.stats._format if hasattr(tr.stats, '_format') else tr.id)
        except Exception as exc:  # noqa: BLE001
            print(f'STA/LTA computation failed: {exc}')

    arrival_seconds = None
    if abs_arrival:
        try:
            arrival_utc = UTCDateTime(abs_arrival)
            arrival_seconds = arrival_utc - tr.stats.starttime
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not parse abs_arrival '{abs_arrival}': {exc}")

    times = tr.times()
    data = tr.data

    if spectrogram:
        fig = plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(times, data, lw=0.8)
        if arrival_seconds is not None:
            ax1.axvline(arrival_seconds, color='red', label=f'Arrival {arrival_seconds:.2f}s')
        if sta_lta and on_off.size > 0:
            for pair in on_off:
                on_t = times[pair[0]]
                ax1.axvline(on_t, color='orange', alpha=0.7)
        if arrival_seconds is not None or (sta_lta and on_off.size > 0):
            ax1.legend(loc='upper right')
        ax1.set_xlim(times.min(), times.max())
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f"{tr.id}{filt_label} | Start {tr.stats.starttime.isoformat()}", fontweight='bold')

        f, t, Sxx = signal.spectrogram(data, tr.stats.sampling_rate)
        ax2 = plt.subplot(2, 1, 2)
        mesh = ax2.pcolormesh(t, f, Sxx, shading='auto', cmap='viridis')
        if arrival_seconds is not None:
            ax2.axvline(arrival_seconds, color='red')
        if sta_lta and on_off.size > 0:
            for pair in on_off:
                ax2.axvline(times[pair[0]], color='orange', alpha=0.6)
        ax2.set_ylabel('Freq (Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylim(0, min(freqmax * 2 if bandpass else f.max(), f.max()))
        cbar = plt.colorbar(mesh, ax=ax2, orientation='horizontal', pad=0.15)
        cbar.set_label('Power', fontweight='bold')
        fig.tight_layout()

        if spec_out:
            os.makedirs(os.path.dirname(spec_out) or '.', exist_ok=True)
            fig.savefig(spec_out, dpi=150)
            print(f"Spectrogram figure saved to {spec_out}")
            plt.close(fig)
        elif show:
            plt.show()
        return

    # Simple trace only
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(times, data, lw=0.8)
    if arrival_seconds is not None:
        ax.axvline(arrival_seconds, color='red', label=f'Arrival {arrival_seconds:.2f}s')
    if sta_lta and on_off.size > 0:
        for pair in on_off:
            ax.axvline(times[pair[0]], color='orange', alpha=0.7)
    if arrival_seconds is not None or (sta_lta and on_off.size > 0):
        ax.legend(loc='upper right')
    ax.set_xlim(times.min(), times.max())
    ax.set_xlabel('Time since start (s)')
    ax.set_ylabel('Amplitude')
    det_suffix = ""
    if sta_lta and on_off.size > 0:
        det_suffix = f" | {on_off.shape[0]} evt(s)"
    ax.set_title(f"{tr.id}{filt_label}{det_suffix} | Start {tr.stats.starttime.isoformat()}", fontweight='bold')
    fig.tight_layout()
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Trace plot saved to {out}")
        plt.close(fig)
    elif show:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Marsquake MiniSEED exploration utility")
    p.add_argument("--data_root", default="", help="Root directory containing .mseed files (optional)")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset identifier for kagglehub download")
    p.add_argument("--list", action="store_true", help="List available .mseed files and exit")
    p.add_argument("--index", type=int, default=0, help="Index of file to read (after sorting)")
    p.add_argument("--trace", type=int, default=0, help="Trace index within the stream to plot")
    p.add_argument("--limit", type=int, default=None, help="Limit number of files when listing")
    p.add_argument("--no_plot", action="store_true", help="Skip plotting MiniSEED trace")
    p.add_argument("--plot_out", default="", help="If set, save MiniSEED trace plot to this path instead of interactive display")
    p.add_argument("--abs_arrival", default="", help="Absolute arrival timestamp (ISO)")
    p.add_argument("--bandpass", action="store_true", help="Apply bandpass filter before plotting")
    p.add_argument("--freqmin", type=float, default=0.5, help="Bandpass minimum frequency (Hz)")
    p.add_argument("--freqmax", type=float, default=1.0, help="Bandpass maximum frequency (Hz)")
    p.add_argument("--spectrogram", action="store_true", help="Also produce spectrogram")
    p.add_argument("--spec_out", default="", help="If set, save spectrogram figure to this path")
    # STA/LTA arguments
    p.add_argument("--sta_lta", action="store_true", help="Run STA/LTA detection")
    p.add_argument("--sta_sec", type=float, default=120.0, help="STA window length (s)")
    p.add_argument("--lta_sec", type=float, default=600.0, help="LTA window length (s)")
    p.add_argument("--thr_on", type=float, default=4.0, help="Trigger on threshold")
    p.add_argument("--thr_off", type=float, default=1.5, help="Trigger off threshold")
    p.add_argument("--trigger_plot_out", default="", help="Save STA/LTA characteristic function plot")
    p.add_argument("--catalog_out", default="", help="Save trigger on-time catalog CSV")
    p.add_argument("--ml_detect", action="store_true", help="Run ML ensemble detection (requires trained model)")
    p.add_argument("--model_path", default="seismic_event_classifier.joblib", help="Path to trained ML model")
    return p.parse_args()


def main():
    args = parse_args()

    root = guess_data_root(args.data_root or None, force_download=True, dataset=args.dataset)
    print(f"Using data root: {root}")

    files = collect_mseed_files(root, limit=args.limit if args.list else None, planet="mars")

    if args.list:
        if not files:
            print("No .mseed files found.")
        else:
            print("Found .mseed files (sorted):")
            for i, f in enumerate(files):
                print(f"[{i}] {f}")
        return

    # Re-collect without limit if selection index invalid after initial (possibly limited) list
    if not files or (args.limit and len(files) < args.index + 1):
        files = collect_mseed_files(root, limit=None, planet="mars")

    target_file = choose_file(files, args.index)
    st = load_stream(target_file)
    tr_for_ml = st[0] if len(st) else None

    if not args.no_plot:
        out = args.plot_out or None
        plot_trace(
            st,
            trace_index=args.trace,
            show=not out,
            out=out,
            abs_arrival=(args.abs_arrival or None),
            bandpass=args.bandpass,
            freqmin=args.freqmin,
            freqmax=args.freqmax,
            spectrogram=args.spectrogram,
            spec_out=(args.spec_out or None),
            sta_lta=args.sta_lta,
            sta_sec=args.sta_sec,
            lta_sec=args.lta_sec,
            thr_on=args.thr_on,
            thr_off=args.thr_off,
            trigger_plot_out=(args.trigger_plot_out or None),
            catalog_out=(args.catalog_out or None),
        )

    if args.ml_detect and tr_for_ml is not None:
        try:
            from core.processing import detect_events_with_ml
            print("Running ML-based detection (Mars)...")
            events = detect_events_with_ml(tr_for_ml, model_path=args.model_path)
            if events:
                print(f"ML detected {len(events)} event window(s):")
                for (s, e) in events:
                    print(f"  - {s.isoformat()} -> {e.isoformat()}")
            else:
                print("ML detected no events.")
        except Exception as e:  # noqa: BLE001
            print(f"ML detection failed: {e}")


if __name__ == "__main__":
    main()
