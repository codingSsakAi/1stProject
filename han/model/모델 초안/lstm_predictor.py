import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def preprocess_lstm_data(df, country=None, purpose=None):
    if country: df = df[df['국적'] == country]
    if purpose: df = df[df['목적'] == purpose]
    df['ds'] = pd.to_datetime(df['연도'].astype(str) + df['월'].astype(str).str.zfill(2), format='%Y%m')
    df = df.groupby('ds')['입국자수'].sum().reset_index()
    return df

def predict_lstm(df, country, purpose, predict_ym):
    data = preprocess_lstm_data(df.copy(), country, purpose)
    if len(data) < 15:
        print("LSTM 데이터 구간이 너무 짧아 예측을 진행할 수 없습니다.")
        return {
            '예측값': None,
            '실제_시계열': data,
            '예측_시계열': None,
            'RMSE': None,
            'MAE': None,
            'MAPE': None
        }
    scaler = MinMaxScaler()
    data['입국자수_scaled'] = scaler.fit_transform(data[['입국자수']])
    window = 12
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

    # 미래 예측 구간 계산
    full_preds = list(data['입국자수_scaled'].values)
    last_window = full_preds[-window:]
    last_date = data['ds'].iloc[-1]
    predict_end = pd.to_datetime(predict_ym, format='%Y%m')
    n_pred = (predict_end.year - last_date.year) * 12 + (predict_end.month - last_date.month)
    future_dates = []
    for i in range(n_pred):
        x_input = np.array(last_window[-window:]).reshape(1, window, 1)
        y_pred = model.predict(x_input, verbose=0)[0, 0]
        full_preds.append(y_pred)
        last_window.append(y_pred)
        last_window = last_window[-window:]
        next_month = (last_date.month + i + 1 - 1) % 12 + 1
        next_year = last_date.year + ((last_date.month + i) // 12)
        future_dates.append(pd.Timestamp(year=next_year, month=next_month, day=1))
    all_dates = list(data['ds']) + future_dates

    # 길이 mismatch 보정
    all_preds = scaler.inverse_transform(np.array(full_preds).reshape(-1, 1)).flatten()
    min_len = min(len(all_dates), len(all_preds))
    all_dates = all_dates[:min_len]
    all_preds = all_preds[:min_len]

    pred_value = all_preds[-1] if n_pred > 0 else None

    # 오차 (실측 구간)
    y_true = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_pred_arr = scaler.inverse_transform(model.predict(X).flatten().reshape(-1, 1)).flatten()
    rmse = mean_squared_error(y_true, y_pred_arr, squared=False)
    mae = mean_absolute_error(y_true, y_pred_arr)
    mape = mean_absolute_percentage_error(y_true, y_pred_arr)

    pred_df = pd.DataFrame({'ds': all_dates, '입국자수': all_preds})
    actual_df = data[['ds', '입국자수']]
    return {
        '예측값': pred_value,
        '실제_시계열': actual_df,
        '예측_시계열': pred_df,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
