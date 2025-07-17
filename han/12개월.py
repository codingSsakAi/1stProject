import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dateutil.relativedelta import relativedelta
import concurrent.futures
import time
import os

def predict_one_combo(args):
    # args: (국적, 목적, group 데이터프레임, today)
    nation, purpose, group, today = args
    try:
        result = {}
        if group.shape[0] < 24:
            return None

        group = group.sort_values('ds').reset_index(drop=True)
        group['성수기'] = group['ds'].dt.month.isin([7,8,12]).astype(int)
        group['명절'] = group['ds'].dt.month.isin([1,2,9,10]).astype(int)

        # 예측 구간: 오늘 기준 다음 달부터 12개월
        start_month = (today + relativedelta(months=1)).replace(day=1)
        future_dates = pd.date_range(start_month, periods=12, freq='MS')
        test_df = pd.DataFrame({'ds': future_dates})
        test_df['성수기'] = test_df['ds'].dt.month.isin([7,8,12]).astype(int)
        test_df['명절']  = test_df['ds'].dt.month.isin([1,2,9,10]).astype(int)

        # Prophet
        try:
            holidays = pd.DataFrame({
                'holiday': '명절일',
                'ds': pd.to_datetime([
                    '2020-01-25', '2020-09-30', '2021-02-12', '2021-09-21',
                    '2022-02-01', '2022-09-10', '2023-01-22', '2023-09-29',
                    '2024-02-10', '2024-09-17', '2025-01-29', '2025-10-06'
                ]),
                'lower_window': -1,
                'upper_window': 2
            })
            prophet = Prophet(
                yearly_seasonality=True,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=3.5,
                seasonality_prior_scale=20,
                holidays_prior_scale=20,
                holidays=holidays
            )
            prophet.add_regressor('성수기')
            prophet.add_regressor('명절')
            prophet.fit(group.rename(columns={'입국자수':'y'}))
            future = test_df[['ds', '성수기', '명절']]
            prophet_forecast = prophet.predict(future)
            test_df['prophet_pred'] = prophet_forecast['yhat'].values
        except Exception as e:
            test_df['prophet_pred'] = np.nan

        # XGBoost
        def add_volatility_features(df, window=3):
            df = df.copy()
            df['rolling_mean'] = df['입국자수'].rolling(window=window, min_periods=1).mean()
            df['rolling_std'] = df['입국자수'].rolling(window=window, min_periods=1).std().fillna(0)
            df['diff'] = df['입국자수'].diff().fillna(0)
            lower, upper = df['입국자수'].quantile([0.05, 0.95])
            df['clipped'] = df['입국자수'].clip(lower=lower, upper=upper)
            return df

        def create_xgb_features(df, window=12):
            df = df.copy()
            for lag in range(1, window+1):
                df[f'lag_{lag}'] = df['입국자수'].shift(lag)
            df = add_volatility_features(df, window=3)
            df = df.dropna().reset_index(drop=True)
            return df

        try:
            xgb_window = 12
            train_xgb = create_xgb_features(group, xgb_window)
            last_vals = group.tail(xgb_window).copy()
            preds = []
            for i in range(12):
                temp = pd.concat([last_vals, pd.DataFrame({'입국자수':[np.nan]}, index=[0])], ignore_index=True)
                temp = create_xgb_features(temp, xgb_window)
                xgb = XGBRegressor(n_estimators=100, random_state=42)
                xgb.fit(train_xgb[[c for c in train_xgb.columns if c.startswith('lag_')] + ['rolling_mean','rolling_std','diff','clipped']], train_xgb['입국자수'])
                X_pred = temp.iloc[-1][[c for c in temp.columns if c.startswith('lag_')] + ['rolling_mean','rolling_std','diff','clipped']].values.reshape(1, -1)
                pred = xgb.predict(X_pred)[0]
                preds.append(pred)
                last_vals = pd.concat([last_vals, pd.DataFrame({'입국자수':[pred]}, index=[0])], ignore_index=True)
                last_vals = last_vals.iloc[1:]
            test_df['xgb_pred'] = preds
        except Exception as e:
            test_df['xgb_pred'] = np.nan

        # LSTM
        try:
            seq_len = 12
            group_features = group[['입국자수', '성수기', '명절']].values
            scaler = StandardScaler()
            scaled = scaler.fit_transform(group_features)
            X_lstm = []
            for i in range(seq_len, len(group)):
                X_lstm.append(scaled[i-seq_len:i])
            X_lstm = np.array(X_lstm)
            y_lstm = scaled[seq_len:, 0]
            model_lstm = Sequential([
                LSTM(80, input_shape=(seq_len, 3), return_sequences=True),
                LSTM(40),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model_lstm.compile(loss='mse', optimizer='adam')
            model_lstm.fit(X_lstm, y_lstm, epochs=30, batch_size=8, verbose=0, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
            last_seq = scaled[-seq_len:]
            preds = []
            for i in range(12):
                feat = np.array([[0, test_df.iloc[i]['성수기'], test_df.iloc[i]['명절']]])
                X_pred = last_seq.reshape(1, seq_len, 3)
                pred_scaled = model_lstm.predict(X_pred, verbose=0)[0,0]
                inv = scaler.inverse_transform(np.array([[pred_scaled, test_df.iloc[i]['성수기'], test_df.iloc[i]['명절']]]))[0,0]
                preds.append(inv)
                last_seq = np.vstack([last_seq[1:], scaler.transform([[inv, test_df.iloc[i]['성수기'], test_df.iloc[i]['명절']]])])
            test_df['lstm_pred'] = preds
        except Exception as e:
            test_df['lstm_pred'] = np.nan

        # Stacking (단순 평균)
        test_df['stacking_pred'] = test_df[['prophet_pred', 'xgb_pred', 'lstm_pred']].mean(axis=1)
        test_df['국적'] = nation
        test_df['목적'] = purpose
        return test_df[['국적','목적','ds','prophet_pred','xgb_pred','lstm_pred','stacking_pred']]
    except Exception as e:
        print(f"예외 발생 [{nation}/{purpose}] : {e}")
        return None

if __name__ == "__main__":
    # 데이터 로딩
    df = pd.read_csv("data/목적별국적별입국소계제거.csv", encoding='cp949')
    df = df.dropna()
    df['입국자수'] = df['입국자수'].astype(float)
    df['ds'] = pd.to_datetime(df['년'].astype(str) + df['월'].astype(str).str.zfill(2), format='%Y%m')

    combos = df.groupby(['국적','목적']).size().reset_index()[['국적','목적']]
    total = len(combos)
    print(f"\n[INFO] 예측 대상 조합: {total}개\n")
    start_time = time.time()
    today = datetime.today()

    # 병렬 처리 인자 준비
    job_args = []
    for idx, row in combos.iterrows():
        nation, purpose = row['국적'], row['목적']
        group = df[(df['국적']==nation)&(df['목적']==purpose)].copy()
        job_args.append((nation, purpose, group, today))

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # 진행률 모니터링
        future_to_combo = {executor.submit(predict_one_combo, args): args[:2] for args in job_args}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_combo)):
            nation, purpose = future_to_combo[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
                    print(f"[{idx+1}/{total}] {nation}-{purpose} 완료")
                else:
                    print(f"[{idx+1}/{total}] {nation}-{purpose} 스킵(데이터 부족)")
            except Exception as exc:
                print(f"[{idx+1}/{total}] {nation}-{purpose} 예외: {exc}")

    elapsed = time.time() - start_time
    if results:
        result_df = pd.concat(results, ignore_index=True)
        result_df.to_csv("12개월_국적목적별_향후12개월_예측.csv", index=False, encoding='utf-8-sig')
        print(f"\n[INFO] 전체 예측 완료! → 12개월_국적목적별_향후12개월_예측.csv")
    else:
        print("\n[INFO] 예측 결과 없음(모든 조합 데이터 부족 등)")

    print(f"[INFO] 총 소요 시간: {elapsed/60:.1f}분")
