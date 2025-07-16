import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from utils import fill_covid_with_mean

def preprocess_lstm_data(df, country=None, purpose=None):
    if country: df = df[df['국적'] == country]
    if purpose: df = df[df['목적'] == purpose]
    df = fill_covid_with_mean(df)
    df['ds'] = pd.to_datetime(df['연도'].astype(str) + df['월'].astype(str).str.zfill(2), format='%Y%m')
    df = df.groupby('ds')['입국자수'].sum().reset_index()
    return df

def predict_lstm(df, country, purpose, predict_ym):
    data = preprocess_lstm_data(df.copy(), country, purpose)
    scaler = MinMaxScaler()
    data['입국자수_scaled'] = scaler.fit_transform(data[['입국자수']])
    window = 12

    # 학습 데이터
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data['입국자수_scaled'].values[i-window:i])
        y.append(data['입국자수_scaled'].values[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], 1), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=8, verbose=0)

    # 미래 예측
    last_window = data['입국자수_scaled'].values[-window:].tolist()
    last_date = data['ds'].iloc[-1]
    predict_end = pd.to_datetime(predict_ym, format='%Y%m')
    n_pred = (predict_end.year - last_date.year) * 12 + (predict_end.month - last_date.month)
    future_dates = []
    future_preds = []

    print("[LSTM TEST] 시작")
    print("  n_pred (예측 개월):", n_pred)
    print("  기존 window:", window, ", 데이터 길이:", len(data))

    for i in range(n_pred):
        x_input = np.array(last_window[-window:]).reshape(1, window, 1)
        y_pred = model.predict(x_input, verbose=0)[0, 0]
        last_window.append(y_pred)
        future_preds.append(y_pred)
        # 날짜
        month = (last_date.month + i + 1 - 1) % 12 + 1
        year = last_date.year + ((last_date.month + i) // 12)
        future_dates.append(pd.Timestamp(year=year, month=month, day=1))
        print(f"  [{i+1}/{n_pred}] future_pred={y_pred:.4f}, future_date={year}-{month:02d}")

    # 미래 예측만 역변환
    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
    pred_value = future_preds_inv[-1] if len(future_preds_inv) > 0 else None

    # 전체 시계열(실측)
    actual_df = data[['ds','입국자수']]
    pred_df = pd.DataFrame({'ds': future_dates, '입국자수': future_preds_inv})

    # 검증 오차
    y_true = scaler.inverse_transform(y.reshape(-1,1)).flatten()
    y_pred = scaler.inverse_transform(model.predict(X).flatten().reshape(-1,1)).flatten()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        '예측값': pred_value,
        '실제_시계열': actual_df,
        '예측_시계열': pred_df,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
