import os
import argparse
import pandas as pd
from obspy import read, UTCDateTime
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib
from core.ml_handler import extract_features

WINDOW_SIZE_SEC = 30

def build_feature_dataset(labels_df: pd.DataFrame):
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
            label = row['label']
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
            feats['label'] = 1 if label == 'quake' else 0
            rows.append(feats)
    return pd.DataFrame(rows)

def parse_args():
    ap = argparse.ArgumentParser(description='Train seismic event classifier (Voting RF+SVM)')
    ap.add_argument('--labels', default='labels.csv', help='Path to labels CSV')
    ap.add_argument('--window', type=int, default=30, help='Window size seconds')
    ap.add_argument('--model_out', default='seismic_event_classifier.joblib', help='Output model filename')
    ap.add_argument('--balance', action='store_true', help='Use class_weight="balanced" for RandomForest & SVM')
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
    print('Building feature dataset...')
    feat_df = build_feature_dataset(labels)
    if feat_df.empty:
        raise SystemExit('No feature rows built.')
    # Ensure only binary labels 0/1
    if not set(feat_df['label'].unique()).issubset({0,1}):
        raise SystemExit('Non-binary labels encountered (expected 0/1).')
    # Stable feature column ordering (alphabetical) for reproducibility
    feature_cols = sorted([c for c in feat_df.columns if c != 'label'])
    X = feat_df[feature_cols]
    y = feat_df['label']
    print(f'Features: {len(feature_cols)} columns -> {feature_cols}')
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
    print(classification_report(y_test, y_pred, target_names=['noise','quake']))
    joblib.dump(ensemble, args.model_out)
    print(f'Model saved -> {args.model_out}')

if __name__ == '__main__':
    main()
