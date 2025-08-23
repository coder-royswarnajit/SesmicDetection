# 🌍 Seismic Detection (Mars & Moon)

## 🎯 **Professional Seismic Analysis Platform**

A comprehensive Python toolkit for Mars and Moon seismic data analysis featuring:

### **🚀 Professional Dashboard** *(Recommended)*
- **Production-ready web interface** with complete workflows
- **Interactive visualizations** and real-time analysis
- **Professional export capabilities** for results and reports
- **Instant demo mode** with synthetic seismic data

**Quick Start:**
```bash
python launch_professional_dashboard.py
```
📖 **Complete Guide:** [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)

### **🔧 Core Analysis Toolkit**
- MiniSEED waveform processing (Mars & Moon)
- CSV (lunar) waveform + feature extraction
- STA/LTA event detection algorithms
- Batch feature catalog export
- Pure ObsPy implementation (no TensorFlow dependency)
- Dockerized runtime environment
- Automatic Kaggle dataset download via kagglehub

## Key Datasets (Kaggle)
Primary combined dataset (Mars & Moon):
- `ahmedsamir1598/space-apps-2024-seismic-detection`

Alternate full-system dataset:
- `tylerdurden73/seismic-detection-across-the-solar-system`

Download (script auto if needed):
```python
import kagglehub
path = kagglehub.dataset_download("ahmedsamir1598/space-apps-2024-seismic-detection")
```

## 📁 Project Structure
```
SesmicDetection/
├── 🌍 PROFESSIONAL DASHBOARD
│   ├── professional_dashboard.py          # Main dashboard application
│   ├── launch_professional_dashboard.py   # Dashboard launcher
│   ├── test_professional_dashboard.py     # Test suite
│   └── DASHBOARD_GUIDE.md                 # Complete user guide
├── 🔧 CORE ANALYSIS MODULES
│   ├── mars.py                            # Mars MiniSEED processing & STA/LTA
│   ├── moon.py                            # Moon MiniSEED processing
│   ├── planet_unified.py                  # Unified Mars/Moon processor
│   ├── train_model.py                     # ML model training
│   └── core/
│       ├── processing.py                  # Signal processing utilities
│       ├── ml_handler.py                  # ML model handling
│       └── phase_picker.py                # Phase picking algorithms
├── 📋 PROJECT FILES
│   ├── requirements.txt                   # Python dependencies
│   ├── Dockerfile                         # Container setup
│   └── README.md                          # This overview
```

## 🎯 **Recommended: Professional Dashboard**

The **Professional Seismic Detection Dashboard** provides a complete, production-ready interface for seismic analysis:

### ✅ **Fully Functional Features**
- **Interactive waveform visualization** with zoom/pan capabilities
- **Real-time STA/LTA detection** with parameter tuning
- **Comprehensive statistical analysis** with professional metrics
- **High-quality export capabilities** (CSV, plots, reports)
- **Instant demo mode** with synthetic seismic data
- **Professional presentation** suitable for stakeholders

### 🚀 **Quick Start**
```bash
python launch_professional_dashboard.py
```
**Dashboard URL:** `http://localhost:8502`

📖 **Complete Guide:** [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)

---

## 🔧 **Core Toolkit Features**
- List & plot MiniSEED traces
- Absolute arrival → relative marking
- Bandpass filtering
- STA/LTA characteristic function + triggers
- Spectral & statistical feature extraction (energy, RMS, skewness, centroid, etc.)
- Batch processing (all files) with per-trace PNG and CSV export
- CSV (lunar) ingestion with legacy column normalization
- Docker image for reproducible execution

## Installation
```bash
pip install -r requirements.txt
```

(Optional virtual environment)
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

## Core Scripts & Usage

### Mars (MiniSEED)
```bash
python mars.py --list --limit 10
python mars.py --index 0 --sta_lta --plot_out trace0.png --trigger_plot_out cft0.png
python mars.py --index 0 --bandpass --freqmin 0.3 --freqmax 2.0 --sta_lta --catalog_out mars_triggers.csv
```

### Moon (MiniSEED subset)
```bash
python moon.py --list --limit 10
python moon.py --index 2 --sta_lta --bandpass --freqmin 0.5 --freqmax 1.5 --plot_out moon_trace.png
```

### Unified (Both Planets)
```bash
# Mars only
python planet_unified.py --planet mars --sta_lta --plot_waveform --features_out mars_feats.csv
# Moon only (limit first 5)
python planet_unified.py --planet moon --sta_lta --moon_limit 5 --features_out moon_feats.csv
# Both with bandpass + feature export
python planet_unified.py --planet both --bandpass --freqmin 0.3 --freqmax 1.0 --sta_lta --features_out all_feats.csv
```

### Lunar CSV (Heuristic Event Picks)
```bash
python moon_cnn.py --train_root data/lunar/train --plot --catalog_out lunar_events.csv --limit 10
```

## Common Flags (Overview)
- `--list`              : List available files then exit
- `--index N`           : Pick file by sorted index
- `--all` (unified)     : Batch over every file
- `--bandpass` + `--freqmin --freqmax`
- `--sta_lta` + `--sta_sec --lta_sec --thr_on --thr_off`
- `--plot_out / --spec_out / --trigger_plot_out`
- `--catalog_out`       : Append trigger on-times to CSV
- `--features_out`      : Feature CSV (unified script)
- `--abs_arrival ISO`   : Mark absolute arrival (converted to relative seconds)

## STA/LTA Parameters
| Param    | Meaning                      | Typical |
|----------|------------------------------|---------|
| sta_sec  | Short-term window (s)        | 60–120  |
| lta_sec  | Long-term window (s)         | 300–600 |
| thr_on   | Trigger-on ratio threshold   | 3–5     |
| thr_off  | Trigger-off ratio threshold  | 1–2     |

Tune to control false positives (raise thr_on) or sensitivity (lower thr_on).

## Feature Set (Unified Extraction)
- Time: arrival_time (if supplied), relative sampling stats
- Amplitude: mean, median, std, min, max, range, RMS, energy
- Distribution: skewness, kurtosis
- Dynamics: derivative mean/std, zero-crossing rate
- Spectral: dominant_frequency, spectral_centroid
- STA/LTA: max, value at arrival
- Window (±5 s by default): mean_window, max_window
(Extendable in `planet_unified.py` or lunar CSV pipeline.)

## Batch Feature Export
```bash
python planet_unified.py --planet mars --all --sta_lta --features_out mars_features.csv
python planet_unified.py --planet moon --all --features_out moon_features.csv
```

## Docker
Build:
```bash
docker build -t seismic-tool .
```
Run (Mars list):
```bash
docker run --rm seismic-tool --list --limit 5
```
Run (Moon STA/LTA):
```bash
docker run --rm -e APP_TARGET=moon.py seismic-tool --sta_lta --index 0
```
Save outputs:
```bash
docker run --rm -v %cd%:/out seismic-tool --sta_lta --index 0 --plot_out /out/trace.png
```

## Kaggle Download Notes
Requires an authenticated kagglehub environment (ensure credentials configured). Scripts attempt dataset download if local root not provided.

## Extending
Planned / optional:
- Parallel batch processing
- JSON export
- ML classifier integration (scikit-learn)
Open an issue / request for additions.

## Troubleshooting
| Issue | Fix |
|-------|-----|
| ObsPy import error | Ensure correct Python version (<=3.11) or rebuild Docker image |
| No files listed | Confirm dataset downloaded; use `--dataset` override |
| Empty feature CSV | Check index/filters; run without `--limit` |

## License
Add a license file (e.g., MIT) if distributing.

## Citation
If using dataset(s), cite the respective Kaggle sources.

---
Concise, modular, reproducible seismic detection workflow for Mars
