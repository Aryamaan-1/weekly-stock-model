import os
import gc
import time
import traceback
from io import BytesIO
from datetime import timedelta, datetime, timezone

import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# === ENVIRONMENT VARIABLES ===
AWS_REGION       = os.environ.get('AWS_REGION', 'ap-south-1')
FEATURES_CSV_KEY = os.environ['FEATURES_CSV_KEY']
S3_BUCKET        = os.environ['S3_BUCKET']
RESULTS_PREFIX   = os.environ.get('RESULTS_PREFIX', 'results')
SNS_TOPIC_ARN    = os.environ.get('SNS_TOPIC_ARN')   # unset to disable SNS
TOP_N            = int(os.environ.get('TOP_N', 10))

MODEL_PARAMS = {
    'objective':        'binary:logistic',
    'eval_metric':      'logloss',
    'tree_method':      'hist',
    'max_depth':        4,
    'learning_rate':    0.1,
    'subsample':        0.7,
    'colsample_bytree': 0.7,
    'alpha':            5.0,
    'lambda':           5.0,
    'seed':             42
}

# AWS clients
s3  = boto3.client('s3',   region_name=AWS_REGION)
sns = boto3.client('sns',  region_name=AWS_REGION) if SNS_TOPIC_ARN else None

def notify_error(subject, message):
    now = datetime.now(timezone.utc)
    print(f"[{now}] ERROR: {subject}\n{message}")
    if sns:
        sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message)

def main():
    start_time = time.time()
    try:
        now = datetime.now(timezone.utc)
        print(f"[{now}] === STARTING MODEL RUN ===")

        # 1) Stream & clean CSV in chunks (no Return_5d drop here)
        now = datetime.now(timezone.utc)
        print(f"[{now}] 1) Streaming feature CSV in chunks from s3://{S3_BUCKET}/{FEATURES_CSV_KEY}")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=FEATURES_CSV_KEY)
        stream = BytesIO(obj['Body'].read())
        del obj
        gc.collect()

        chunks = []
        for chunk in pd.read_csv(stream, parse_dates=['Date'], chunksize=200_000):
            # Drop unwanted columns
            for c in ['Unnamed: 0','Unnamed: 0.1','CAPITALINE_CODE','CAPITALINE CODE']:
                if c in chunk:
                    chunk.drop(columns=c, inplace=True)
            # Replace infs and NaNs in features, but keep rows missing Return_5d for prediction
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            # One-hot encode Sector
            chunk = pd.concat([chunk, pd.get_dummies(chunk['Sector'], prefix='Sector')],
                              axis=1).drop(columns=['Sector'])
            chunks.append(chunk)

        df = pd.concat(chunks, axis=0)
        del chunks
        gc.collect()

        # 2) Sort & prepare
        now = datetime.now(timezone.utc)
        print(f"[{now}] 2) Sorting data and preparing target field")
        df.sort_values(['Symbol','Date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['Target'] = (df['Return_5d'] > 0).astype(int)

        # 3) Determine dates
        last_date   = df['Date'].max()
        cutoff_date = last_date - timedelta(days=5)   # all dates ≤ this have valid 5d returns
        next_week   = last_date + timedelta(days=1)

        # 4) Split into training vs. “recent” (last 5 days) for prediction
        train_df  = df[df['Date'] <= cutoff_date].dropna(subset=['Return_5d']).copy()
        predict_df = df[df['Date'] == last_date].copy()

        # 5) Train on train_df
        now = datetime.now(timezone.utc)
        print(f"[{now}] 5) Training XGBoost on data through {cutoff_date.date()} "
              f"(train rows: {len(train_df)})")
        feat_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns
                     if c not in ['Return_1d','Return_5d','Return_10d','Target']]
        scaler   = StandardScaler().fit(train_df[feat_cols])
        X_train  = scaler.transform(train_df[feat_cols])
        dtrain   = xgb.DMatrix(X_train, label=train_df['Target'], feature_names=feat_cols)
        bst      = xgb.train(MODEL_PARAMS, dtrain, num_boost_round=1000, verbose_eval=False)

        # 6) Save & upload model JSON into next_week folder
        now = datetime.now(timezone.utc)
        print(f"[{now}] 6) Saving and uploading model JSON for week {next_week.date()}")
        local_model = "xgb_model.json"
        bst.save_model(local_model)
        run_prefix = f"{RESULTS_PREFIX}/{next_week.strftime('%Y-%m-%d')}"
        s3.upload_file(local_model, S3_BUCKET, f"{run_prefix}/{local_model}")
        os.remove(local_model)
        del X_train, dtrain, train_df
        gc.collect()

        # 7) Generate predictions for next_week
        now = datetime.now(timezone.utc)
        print(f"[{now}] 7) Generating predictions for week starting {next_week.date()} "
              f"(prediction rows: {len(predict_df)})")
        X_live  = scaler.transform(predict_df[feat_cols])
        syms    = predict_df['Symbol'].values
        scores  = bst.predict(xgb.DMatrix(X_live, feature_names=feat_cols))

        out = pd.DataFrame({
            'Symbol':     syms,
            'Score':      scores,
            'Week_Start': next_week
        }).sort_values('Score', ascending=False)

        # 8) Write & upload predictions CSV
        now = datetime.now(timezone.utc)
        print(f"[{now}] 8) Writing and uploading next_week_predictions.csv")
        local_out = "next_week_predictions.csv"
        out.to_csv(local_out, index=False)
        s3.upload_file(local_out, S3_BUCKET, f"{run_prefix}/{local_out}")
        os.remove(local_out)

        # 9) Publish top-N via SNS (include run time & last_date)
        if sns:
            elapsed = time.time() - start_time
            now = datetime.now(timezone.utc)
            print(f"[{now}] 9) Publishing top {TOP_N} via SNS (run time: {elapsed:.1f}s)")
            topn = out.head(TOP_N)
            msg  = (
                f"Data last date:    {last_date.date()}\n"
                f"Training through:   {cutoff_date.date()}\n"
                f"Predicting week:    {next_week.date()}\n"
                f"Run time:           {elapsed:.1f}s\n\n"
                f"Top {TOP_N} picks:\n"
            )
            msg += "\n".join(
                f"{i+1}. {sym} (score={score:.4f})"
                for i, (sym, score) in enumerate(zip(topn['Symbol'], topn['Score']))
            )
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"Weekly Picks – {next_week.date()}",
                Message=msg
            )

        # 10) Final clean-up
        del df, predict_df, out, scaler, bst, X_live, syms, scores
        gc.collect()

        total_time = time.time() - start_time
        now = datetime.now(timezone.utc)
        print(f"[{now}] === MODEL RUN COMPLETE in {total_time:.1f}s ===")

    except Exception:
        tb = traceback.format_exc()
        notify_error("Model Pipeline Error", tb)
        raise

if __name__ == "__main__":
    main()
