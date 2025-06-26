import os
import gc
import traceback
from io import BytesIO
from datetime import timedelta, datetime

import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# === ENVIRONMENT VARIABLES ===
AWS_REGION       = os.environ.get('AWS_REGION',     'us-east-1')
FEATURES_CSV_KEY = os.environ['FEATURES_CSV_KEY']
S3_BUCKET        = os.environ['S3_BUCKET']
RESULTS_PREFIX   = os.environ.get('RESULTS_PREFIX','results')
SNS_TOPIC_ARN    = os.environ.get('SNS_TOPIC_ARN')   # unset to disable SNS
NUM_STOCKS       = int(os.environ.get('NUM_STOCKS',500))
TOP_N            = int(os.environ.get('TOP_N',10))
MODEL_PARAMS     = {
    'objective':'binary:logistic',
    'eval_metric':'logloss',
    'tree_method':'hist',
    'max_depth':4,
    'learning_rate':0.1,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'alpha':5.0,
    'lambda':5.0,
    'seed':42
}

# AWS clients
s3  = boto3.client('s3',   region_name=AWS_REGION)
sns = boto3.client('sns',  region_name=AWS_REGION) if SNS_TOPIC_ARN else None

# Date‑stamped S3 prefix
today      = datetime.utcnow().strftime("%Y-%m-%d")
run_prefix = f"{RESULTS_PREFIX}/{today}"

def notify_error(subject, message):
    print(f"[{datetime.utcnow()}] ERROR: {subject}\n{message}")
    if sns:
        sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message)

def main():
    try:
        print(f"[{datetime.utcnow()}] === STARTING MODEL RUN ===")

        # 1) Stream features CSV from S3
        print(f"[{datetime.utcnow()}] 1) Streaming feature CSV from s3://{S3_BUCKET}/{FEATURES_CSV_KEY}")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=FEATURES_CSV_KEY)
        df = pd.read_csv(BytesIO(obj['Body'].read()), parse_dates=['Date'])
        del obj
        gc.collect()

        # 2) Clean & encode
        print(f"[{datetime.utcnow()}] 2) Cleaning data")
        drop_cols = ['Unnamed: 0','Unnamed: 0.1','CAPITALINE_CODE','CAPITALINE CODE']
        df.drop(columns=[c for c in drop_cols if c in df], inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['Return_5d'], inplace=True)
        df.sort_values(['Symbol','Date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"[{datetime.utcnow()}] 2a) One‑hot encoding Sector")
        df = pd.concat([df, pd.get_dummies(df['Sector'], prefix='Sector')],
                       axis=1).drop(columns=['Sector'])

        # 3) Filter & target
        print(f"[{datetime.utcnow()}] 3) Filtering top {NUM_STOCKS} symbols")
        symbols = df['Symbol'].unique()[:NUM_STOCKS]
        df      = df[df['Symbol'].isin(symbols)].copy()
        df['Target'] = (df['Return_5d'] > 0).astype(int)

        # 4) Train on full data
        print(f"[{datetime.utcnow()}] 4) Training XGBoost on full dataset")
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in ['Return_1d','Return_5d','Return_10d','Target']]
        scaler = StandardScaler().fit(df[feat_cols])
        X_all  = scaler.transform(df[feat_cols])
        dall   = xgb.DMatrix(X_all, label=df['Target'], feature_names=feat_cols)
        bst    = xgb.train(MODEL_PARAMS, dall, num_boost_round=1000, verbose_eval=False)

        # 4a) Save model JSON locally and upload
        print(f"[{datetime.utcnow()}] 4a) Saving and uploading model JSON")
        local_model = "xgb_model.json"
        bst.save_model(local_model)
        s3.upload_file(local_model, S3_BUCKET, f"{run_prefix}/{local_model}")
        os.remove(local_model)

        del X_all, dall
        gc.collect()

        # 5) Next‑week predictions
        print(f"[{datetime.utcnow()}] 5) Generating next‑week predictions")
        last_date   = df['Date'].max()
        mask_last   = df['Date'] == last_date
        X_live      = scaler.transform(df.loc[mask_last, feat_cols])
        syms_live   = df.loc[mask_last, 'Symbol'].values
        scores_live = bst.predict(xgb.DMatrix(X_live, feature_names=feat_cols))
        next_week   = last_date + timedelta(days=1)

        out = pd.DataFrame({
            'Symbol':     syms_live,
            'Score':      scores_live,
            'Week_Start': next_week
        }).sort_values('Score', ascending=False)

        # 6) Write & upload predictions CSV
        print(f"[{datetime.utcnow()}] 6) Writing and uploading next_week_predictions.csv")
        local_out = "next_week_predictions.csv"
        out.to_csv(local_out, index=False)
        s3.upload_file(local_out, S3_BUCKET, f"{run_prefix}/{local_out}")
        os.remove(local_out)

        # 7) Publish top‑N via SNS
        if sns:
            print(f"[{datetime.utcnow()}] 7) Publishing top {TOP_N} via SNS")
            top10 = out.head(TOP_N)
            msg   = f"Top {TOP_N} picks for week starting {next_week.date()}:\n"
            msg  += "\n".join(
                f"{i+1}. {sym} (score={score:.4f})"
                for i,(sym,score) in enumerate(zip(top10['Symbol'], top10['Score']))
            )
            sns.publish(TopicArn=SNS_TOPIC_ARN,
                        Subject="Weekly Model Picks", Message=msg)

        print(f"[{datetime.utcnow()}] === MODEL RUN COMPLETE ===")

    except Exception:
        tb = traceback.format_exc()
        notify_error("Model Pipeline Error", tb)
        raise

if __name__ == "__main__":
    main()
