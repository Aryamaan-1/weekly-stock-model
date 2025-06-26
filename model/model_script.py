import os
import gc
import traceback
import tempfile
from datetime import timedelta, datetime
import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    confusion_matrix, roc_auc_score
)

# === ENVIRONMENT VARIABLES ===
FEATURES_CSV_KEY    = os.environ['FEATURES_CSV_KEY']
S3_BUCKET           = os.environ['S3_BUCKET']
RESULTS_PREFIX      = os.environ.get('RESULTS_PREFIX', 'results')
SNS_TOPIC_ARN       = os.environ['SNS_TOPIC_ARN']
NUM_STOCKS          = int(os.environ.get('NUM_STOCKS', 500))
TOP_N               = int(os.environ.get('TOP_N', 10))
START_CAP           = float(os.environ.get('START_CAP', 100.0))
TRANSACTION_FEE_PCT = float(os.environ.get('TRANSACTION_FEE_PCT', 0.001))
SLIPPAGE_PCT        = float(os.environ.get('SLIPPAGE_PCT', 0.0005))
MODEL_PARAMS        = {
    'objective':'binary:logistic','eval_metric':'logloss',
    'tree_method':'hist','max_depth':4,
    'learning_rate':0.1,'subsample':0.7,'colsample_bytree':0.7,
    'alpha':5.0,'lambda':5.0,'seed':42
}

# AWS clients
s3  = boto3.client('s3')
sns = boto3.client('sns')

# Create a dated results prefix: e.g. results/2025-06-26/
today = datetime.utcnow().strftime("%Y-%m-%d")
run_prefix = f"{RESULTS_PREFIX}/{today}"
errors = []

def s3_download(key, local_path):
    s3.download_file(S3_BUCKET, key, local_path)

def s3_upload(local_path, key):
    s3.upload_file(local_path, S3_BUCKET, key)

try:
    # 0) FETCH FEATURES CSV
    tmp_feat = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    s3_download(FEATURES_CSV_KEY, tmp_feat)
    df = pd.read_csv(tmp_feat, parse_dates=['Date'])
    # ... (cleaning exactly as before, minus prints) ...
    for c in ['Unnamed: 0','Unnamed: 0.1','CAPITALINE_CODE','CAPITALINE CODE']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Return_5d'], inplace=True)
    df.sort_values(['Symbol','Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # One-hot encode Sector
    sector_dummies = pd.get_dummies(df['Sector'], prefix='Sector')
    df = pd.concat([df, sector_dummies], axis=1).drop(columns=['Sector'])

    # Filter & target
    symbols = df['Symbol'].unique()[:NUM_STOCKS]
    df = df[df['Symbol'].isin(symbols)].copy()
    df['Target'] = (df['Return_5d'] > 0).astype(int)

    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in ['Return_1d','Return_5d','Return_10d','Target']]

    # Train/test split
    unique_dates = sorted(df['Date'].unique())
    split_date = unique_dates[int(0.8*len(unique_dates))]
    train_df = df[df['Date'] <= split_date]
    test_df  = df[df['Date'] >  split_date]

    # Scale & train
    scaler = StandardScaler().fit(train_df[feat_cols])
    dtr = xgb.DMatrix(scaler.transform(train_df[feat_cols]), label=train_df['Target'], feature_names=feat_cols)
    dte = xgb.DMatrix(scaler.transform(test_df[feat_cols]),  label=test_df['Target'],  feature_names=feat_cols)
    bst = xgb.train(MODEL_PARAMS, dtr, num_boost_round=1000, verbose_eval=False)

    # Save model JSON to temp, then upload
    tmp_model = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    bst.save_model(tmp_model)
    s3_upload(tmp_model, f"{run_prefix}/xgb_model.json")

    # Weekly backtest loop as before (records, fee, slippage, etc.)
    # ... produce `weekly_scores_all.csv`, `weekly_top10.csv`, `summary.txt`, plots, `feature_importance.csv` & `.png` ...
    # Upload each artifact under f"{run_prefix}/..."

    # 8) Next‑week predictions & SNS
    last_date = test_df['Date'].max()
    mask_last = test_df['Date'] == last_date
    live = pd.DataFrame({'Symbol': test_df.loc[mask_last,'Symbol']})
    live_scores = bst.predict(xgb.DMatrix(
        scaler.transform(test_df.loc[mask_last, feat_cols]), feature_names=feat_cols))
    live['Score'] = live_scores

    out = live.sort_values('Score', ascending=False)
    next_week = last_date + timedelta(days=1)
    out['Week_Start'] = next_week
    tmp_pred = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    out.to_csv(tmp_pred, index=False)
    s3_upload(tmp_pred, f"{run_prefix}/next_week_predictions.csv")

    # Send top-N via SNS
    top10 = out.head(TOP_N)
    msg = "Top 10 picks for week starting %s:\n" % next_week.date()
    msg += "\n".join(f"{i+1}. {sym} (score={score:.4f})"
                     for i,(sym,score) in enumerate(zip(top10['Symbol'], top10['Score'])))
    sns.publish(TopicArn=SNS_TOPIC_ARN, Subject="Weekly Model Picks", Message=msg)

except Exception:
    errors.append(traceback.format_exc())

if errors:
    print("Encountered errors:\n" + "\n".join(errors))

gc.collect()
