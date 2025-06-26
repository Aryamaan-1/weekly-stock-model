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
    'objective':       'binary:logistic',
    'eval_metric':     'logloss',
    'tree_method':     'hist',
    'max_depth':       4,
    'learning_rate':   0.1,
    'subsample':       0.7,
    'colsample_bytree':'0.7',
    'alpha':           5.0,
    'lambda':          5.0,
    'seed':            42
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

        # 1) Stream & clean CSV in chunks
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
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(subset=['Return_5d'], inplace=True)
            # One-hot encode Sector
            chunk = pd.concat([chunk, pd.get_dummies(chunk['Sector'], prefix='Sector')],
                              axis=1).drop(columns=['Sector'])
            chunks.append(chunk)

        # concatenate all cleaned chunks
        df = pd.concat(chunks, axis=0)
        del chunks
        gc.collect()

        # 2) Sort & create target
        now = datetime.now(timezone.utc)
        print(f"[{now}] 2) Sorting data and creating target")
        df.sort_values(['Symbol','Date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['Target'] = (df['Return_5d'] > 0).astype(int)

        # determine last available date and next week's start
        last_date = df['Date'].max()
        next_week = last_date + timedelta(days=1)

        # 3) Train on full data
        now = datetime.now(timezone.utc)
        print(f"[{now}] 3) Training XGBoost on data through {last_date.date()}")
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in ['Return_1d','Return_5d','Return_10d','Target']]
        scaler = StandardScaler().fit(df[feat_cols])
        X_all  = scaler.transform(df[feat_cols])
        dmat   = xgb.DMatrix(X_all, label=df['Target'], feature_names=feat_cols)
        bst    = xgb.train(MODEL_PARAMS, dmat, num_boost_round=1000, verbose_eval=False)

        # 3a) Save model JSON locally and upload into next_week folder
        now = datetime.now(timezone.utc)
        print(f"[{now}] 3a) Saving and uploading model JSON for week {next_week.date()}")
        local_model = "xgb_model.json"
        bst.save_model(local_model)
        run_prefix = f"{RESULTS_PREFIX}/{next_week.strftime('%Y-%m-%d')}"
        s3.upload_file(local_model, S3_BUCKET, f"{run_prefix}/{local_model}")
        os.remove(local_model)
        del X_all, dmat
        gc.collect()

        # 4) Generate next-week predictions
        now = datetime.now(timezone.utc)
        print(f"[{now}] 4) Generating predictions for week starting {next_week.date()}")
        mask_last = df['Date'] == last_date
        X_live    = scaler.transform(df.loc[mask_last, feat_cols])
        syms_live = df.loc[mask_last, 'Symbol'].values
        scores    = bst.predict(xgb.DMatrix(X_live, feature_names=feat_cols))

        out = pd.DataFrame({
            'Symbol':     syms_live,
            'Score':      scores,
            'Week_Start': next_week
        }).sort_values('Score', ascending=False)

        # 5) Write & upload predictions CSV
        now = datetime.now(timezone.utc)
        print(f"[{now}] 5) Writing and uploading next_week_predictions.csv")
        local_out = "next_week_predictions.csv"
        out.to_csv(local_out, index=False)
        s3.upload_file(local_out, S3_BUCKET, f"{run_prefix}/{local_out}")
        os.remove(local_out)

        # 6) Publish top-N via SNS, including run time and last_date
        if sns:
            elapsed = time.time() - start_time
            now = datetime.now(timezone.utc)
            print(f"[{now}] 6) Publishing top {TOP_N} via SNS (run time: {elapsed:.1f}s)")
            topn = out.head(TOP_N)
            msg  = (
                f"Data last date: {last_date.date()}\n"
                f"Week starting:  {next_week.date()}\n"
                f"Run time:       {elapsed:.1f}s\n\n"
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

        # 7) Clean up in-memory objects
        del df, out, scaler, bst, X_live, syms_live, scores
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
