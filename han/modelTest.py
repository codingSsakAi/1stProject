import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

# 1. 데이터 로드 및 날짜 생성
df = pd.read_csv("data/목적별국적별입국소계제거.csv", encoding='cp949')
df['날짜'] = pd.to_datetime(df['년'].astype(str) + '-' + df['월'].astype(str).str.zfill(2) + '-01')

# 2. 조합별 row count 집계
min_row = 24  # 최소 24개월(2년) 데이터 기준
combo_counts = df.groupby(['국적','목적']).size().reset_index(name='row_count')
valid_combos = combo_counts[combo_counts['row_count'] >= min_row].reset_index(drop=True)

# 3. 결과 저장용 리스트
results = []

for idx, row in valid_combos.iterrows():
    nation, purpose = row['국적'], row['목적']
    print(f"[{idx+1}/{len(valid_combos)}] {nation} - {purpose} 시작")
    group_df = df[(df['국적']==nation) & (df['목적']==purpose)].sort_values('날짜').reset_index(drop=True)
    group_df = group_df[['날짜', '입국자수']].rename(columns={'날짜':'ds','입국자수':'y'})
    
    # 결측치 제거 및 최소 row 확인
    group_df = group_df.dropna()
    if group_df.shape[0] < min_row:
        continue

    # 데이터 분할
    train_size = int(len(group_df)*0.8)
    train, test = group_df.iloc[:train_size], group_df.iloc[train_size:]

    if train.shape[0] < 6 or test.shape[0] < 6:  # 최소 train/test 분리
        continue

    # Prophet
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(train)
        future = m.make_future_dataframe(periods=len(test), freq='MS')
        forecast = m.predict(future)
        prophet_pred = forecast['yhat'].values[-len(test):]
    except Exception as e:
        print(f"[PROPHET ERROR][{nation}/{purpose}] {e}")
        prophet_pred = np.full(len(test), np.nan)

    # XGBoost
    def make_lag_features(df, lags=[1,2,3]):
        for lag in lags:
            df[f'y_lag{lag}'] = df['y'].shift(lag)
        return df.dropna().reset_index(drop=True)

    xgb_df = make_lag_features(group_df.copy())
    if xgb_df.shape[0] < train_size:
        continue
    xgb_train = xgb_df.iloc[:train_size-3]
    xgb_test = xgb_df.iloc[train_size-3:]
    xgb_features = [f'y_lag{i}' for i in [1,2,3]]

    try:
        xgb_model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        xgb_model.fit(xgb_train[xgb_features], xgb_train['y'])
        xgb_pred = xgb_model.predict(xgb_test[xgb_features])
    except Exception as e:
        xgb_pred = np.full(len(xgb_test), np.nan)

    # LSTM
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(group_df[['y']])
    def make_lstm_seq(arr, window=3):
        X, y = [], []
        for i in range(window, len(arr)):
            X.append(arr[i-window:i, 0])
            y.append(arr[i, 0])
        return np.array(X), np.array(y)
    X, y_ = make_lstm_seq(scaled, window=3)
    split = train_size-3
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_[:split], y_[split:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    try:
        lstm_model = Sequential([
            LSTM(32, input_shape=(X_train.shape[1],1), return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        lstm_model.compile(loss='mae', optimizer='adam')
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0, 
                        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
        lstm_pred_scaled = lstm_model.predict(X_test)
        lstm_pred = scaler.inverse_transform(np.concatenate([lstm_pred_scaled, np.zeros_like(lstm_pred_scaled)], axis=1))[:,0]
    except Exception as e:
        lstm_pred = np.full(len(X_test), np.nan)

    # stacking 데이터프레임 정렬(길이 맞추기)
    min_len = min(len(test.iloc[3:]), len(xgb_pred), len(lstm_pred), len(prophet_pred[3:]))
    stacking_df = pd.DataFrame({
        "prophet": prophet_pred[3:3+min_len],
        "xgb": xgb_pred[:min_len],
        "lstm": lstm_pred[:min_len],
        "target": test['y'].values[3:3+min_len]
    })
    stacking_df = stacking_df.dropna().reset_index(drop=True)
    if len(stacking_df) < 5:  # 최소 5개 이하면 스킵
        continue

    # 메타 모델 - Linear Regression
    meta_lr = LinearRegression()
    meta_lr.fit(stacking_df[['prophet','xgb','lstm']], stacking_df['target'])
    meta_pred_lr = meta_lr.predict(stacking_df[['prophet','xgb','lstm']])
    mae_lr = mean_absolute_error(stacking_df['target'], meta_pred_lr)

    # 메타 모델 - XGBoost
    meta_xgb = XGBRegressor(n_estimators=100, max_depth=2)
    meta_xgb.fit(stacking_df[['prophet','xgb','lstm']], stacking_df['target'])
    meta_pred_xgb = meta_xgb.predict(stacking_df[['prophet','xgb','lstm']])
    mae_xgb = mean_absolute_error(stacking_df['target'], meta_pred_xgb)

    results.append({
        "국적": nation, "목적": purpose,
        "test_samples": min_len,
        "mae_linear": mae_lr,
        "mae_xgb": mae_xgb
    })

# 4. 결과 저장 및 요약 출력
result_df = pd.DataFrame(results)
result_df = result_df.sort_values('mae_linear')
result_df.to_csv("stacking_예측성능_결과.csv", index=False, encoding='utf-8-sig')
print(result_df.head(10))  # 상위 10개 조합 성능 요약
