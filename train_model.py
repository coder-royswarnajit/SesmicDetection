import os
import argparse
import pandas as pd
from obspy import read, UTCDateTime
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib
from core.ml_handler import extract_features

WINDOW_SIZE_SEC = 30

def derive_label_map(labels_df: pd.DataFrame, noise_tokens: Tuple[str,...] = ('noise', 'background', 'bg')) -> Dict[str,int]:
    """Create deterministic label->int mapping with 'noise' class (if present) forced to 0.

    Any token in *noise_tokens* (case-insensitive) appearing among labels becomes index 0.
    Remaining labels sorted alphabetically for stability.
    """
    unique_labels = sorted({str(l).strip() for l in labels_df['label'].unique()})
    noise_label = None
    for cand in unique_labels:
        if cand.lower() in noise_tokens:
            noise_label = cand
            break
    ordered: List[str] = []
    if noise_label is not None:
        ordered.append(noise_label)
    for l in unique_labels:
        if l != noise_label:
            ordered.append(l)
    return {lab: idx for idx, lab in enumerate(ordered)}


def build_feature_dataset(labels_df: pd.DataFrame, label_map: Dict[str,int]):
    rows = []
    for raw_fname, group in labels_df.groupby('filename'):
        fname = str(raw_fname)
        if not os.path.isfile(fname):
            print(f'Skip missing file: {fname}')
            continue
        try:
            st = read(fname)
            tr = st[0]
        except Exception as e:  # noqa: BLE001
            print(f'Read fail {fname}: {e}')
            continue
        for _idx, row in group.iterrows():
            label_str = str(row['label']).strip()
            try:
                et = UTCDateTime(row['event_time'])
            except Exception:
                print(f'Bad event_time: {row["event_time"]}')
                continue
            try:
                seg = tr.copy().trim(starttime=et - WINDOW_SIZE_SEC/2, endtime=et + WINDOW_SIZE_SEC/2).data
            except Exception as e:  # noqa: BLE001
                print(f'Trim fail {fname}: {e}')
                continue
            sr = tr.stats.sampling_rate
            expected = int(WINDOW_SIZE_SEC * sr * 0.9)
            if seg.size < expected:
                continue
            feats = extract_features(seg, sr)
            feats['label'] = label_map.get(label_str, 0)  # default to noise index
            rows.append(feats)
    return pd.DataFrame(rows)

def parse_args():
    ap = argparse.ArgumentParser(description='Train multi-class seismic event classifier (Voting RF+SVM)')
    ap.add_argument('--labels', default='labels.csv', help='Path to labels CSV')
    ap.add_argument('--window', type=int, default=30, help='Window size seconds')
    ap.add_argument('--model_out', default='seismic_event_classifier.joblib', help='Output model filename')
    ap.add_argument('--balance', action='store_true', help='Use class_weight="balanced" for RandomForest & SVM')
    ap.add_argument('--print_map', action='store_true', help='Print derived label mapping and exit (dry run)')
    return ap.parse_args()

def main():
    global WINDOW_SIZE_SEC
    args = parse_args()
    WINDOW_SIZE_SEC = args.window
    if not os.path.isfile(args.labels):
        raise SystemExit(f'{args.labels} not found.')
    labels = pd.read_csv(args.labels)
    if labels.empty:
        raise SystemExit('Labels file empty.')
    # Derive multi-class mapping
    label_map = derive_label_map(labels)
    inv_map = {v:k for k,v in label_map.items()}
    print(f'Derived label map: {label_map}')
    if args.print_map:
        return
    print('Building feature dataset...')
    feat_df = build_feature_dataset(labels, label_map)
    if feat_df.empty:
        raise SystemExit('No feature rows built.')
    # Stable feature column ordering: use ml_handler._FEATURE_ORDER if present else alphabetical
    feature_cols = [c for c in feat_df.columns if c != 'label']
    # Keep order deterministic: sort to avoid dependency on dict insertion if mismatch
    feature_cols = sorted(feature_cols)
    X = feat_df[feature_cols]
    y = feat_df['label']
    print(f'Features: {len(feature_cols)} columns -> {feature_cols}')
    print(f'Class distribution: {y.value_counts().to_dict()}')
    if len(label_map) < 2:
        print('Warning: Only one class present; classifier training may fail.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    rf_params: Dict[str, Any] = { 'n_estimators': 160, 'random_state': 42 }
    svm_params: Dict[str, Any] = { 'gamma': 'scale', 'probability': True, 'random_state': 42 }
    if args.balance:
        rf_params['class_weight'] = 'balanced'
        svm_params['class_weight'] = 'balanced'
    clf_rf = make_pipeline(StandardScaler(), RandomForestClassifier(**rf_params))
    clf_svm = make_pipeline(StandardScaler(), SVC(**svm_params))
    ensemble = VotingClassifier(estimators=[('rf', clf_rf), ('svm', clf_svm)], voting='soft')
    print('Training ensemble...')
    ensemble.fit(X_train, y_train)
    print('Evaluating...')
    y_pred = ensemble.predict(X_test)
    target_names = [inv_map[i] for i in sorted(inv_map.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    artifact = {
        'model': ensemble,
        'feature_order': feature_cols,
        'label_map': label_map,
        'window_size_sec': WINDOW_SIZE_SEC,
    }
    joblib.dump(artifact, args.model_out)
    print(f'Model artifact (model+meta) saved -> {args.model_out}')

if __name__ == '__main__':
    main()
