# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
import os

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_forecast(국적_입력, 목적_입력, 예측연도, 예측월리스트):
    파일경로 = "./data/외국인입국자_전처리완료_딥러닝용.csv"
    df = pd.read_csv(파일경로)
    df['년월'] = pd.to_datetime(df['연도'].astype(str) + '-' + df['월'].astype(str).str.zfill(2))
    df['입국자수'] = df['입국자수'].astype(int)
    df['연도'] = df['년월'].dt.year
    df['월'] = df['년월'].dt.month
    df['연도편차'] = df['연도'] - df['연도'].min()
    df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
    df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
    df['성수기여부'] = df['월'].isin([2, 7, 8]).astype(int)
    df['방학여부'] = df['월'].isin([1, 7, 8, 12]).astype(int)
    df['학기여부'] = df['월'].isin([3, 9]).astype(int)

    유효 = df.groupby(['국적', '목적']).size().reset_index(name='count')
    유효 = 유효[유효['count'] >= 24]

    if 국적_입력:
        유효 = 유효[유효['국적'] == 국적_입력]
    if 목적_입력:
        유효 = 유효[유효['목적'] == 목적_입력]

    if 유효.empty:
        return {"error": "⛔ 해당 조건에 맞는 데이터가 없습니다."}

    os.makedirs("./model", exist_ok=True)
    results = []

    for idx, row in 유효.iterrows():
        국적, 목적 = row['국적'], row['목적']
        print(f"[{idx+1}/{len(유효)}] ⏳ 모델 생성 중: {국적} / {목적}...")

        try:
            모델명 = f"./model/{국적}_{목적}"
            data = df[(df['국적'] == 국적) & (df['목적'] == 목적)].copy()

            feature_cols = ['연도편차', '월', '월_cos', '월_sin', '성수기여부']
            if 목적 == '관광':
                feature_cols += ['방학여부']
            elif 목적 == '유학연수':
                feature_cols += ['학기여부']

            use_log = 목적 in ['공용', '관광', '상용']
            gru_epochs = 250 if 목적 == '유학연수' else (200 if 목적 == '공용' else 150)

            X = data[feature_cols]
            y_raw = data['입국자수']
            y = np.log1p(y_raw) if use_log else y_raw

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, y_train = X_scaled[:-6], y[:-6]
            X_all = scaler.transform(X)

            param_dict = {
                '공용': {'n_estimators': 400, 'max_depth': 3},
                '관광': {'n_estimators': 350, 'max_depth': 4},
                '상용': {'n_estimators': 250, 'max_depth': 3},
                '유학연수': {'n_estimators': 450, 'max_depth': 4}
            }
            params = param_dict.get(목적, {'n_estimators': 250, 'max_depth': 2})

            xgb = XGBRegressor(**params)
            xgb.fit(X_train, y_train)
            joblib.dump(xgb, f"{모델명}_xgb.pkl")
            pred_xgb = xgb.predict(X_train)
            residuals = y_train.values - pred_xgb

            model_rnn, scaler_resid = None, None
            if len(residuals) >= 18:
                seq_len = 18
                scaler_resid = StandardScaler()
                resid_scaled = scaler_resid.fit_transform(residuals.reshape(-1, 1)).flatten()
                X_seq = np.array([resid_scaled[i:i+seq_len] for i in range(len(resid_scaled) - seq_len)])
                y_seq = np.array([resid_scaled[i+seq_len] for i in range(len(resid_scaled) - seq_len)])
                X_seq = X_seq.reshape(-1, seq_len, 1)

                model_rnn = Sequential()
                model_rnn.add(GRU(64, return_sequences=True, input_shape=(seq_len, 1)))
                model_rnn.add(Dropout(0.2))
                model_rnn.add(GRU(32))
                model_rnn.add(Dense(1))
                model_rnn.compile(optimizer='adam', loss='mse')
                model_rnn.fit(X_seq, y_seq, epochs=gru_epochs + 100, verbose=0)
                model_rnn.save(f"{모델명}_gru.h5")
                joblib.dump(scaler_resid, f"{모델명}_scaler_resid.pkl")

            joblib.dump(scaler, f"{모델명}_scaler.pkl")

        except Exception as e:
            print(f"🚫 오류 발생: {국적}/{목적} - {e}")
            continue

    print("✅ 전체 모델 학습 완료.")
    return

if __name__ == "__main__":
    국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
    목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
    연도 = input("예측 연도 입력 (예: 2026): ").strip()
    월입력 = input("예측할 월 입력 (예: 3,6,9 또는 1~12): ").strip()

    if '~' in 월입력:
        start, end = map(int, 월입력.split('~'))
        예측월 = list(range(start, end + 1))
    else:
        예측월 = list(map(int, 월입력.split(',')))

    run_forecast(국적 or None, 목적 or None, int(연도), 예측월)
