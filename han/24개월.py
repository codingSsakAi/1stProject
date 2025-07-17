import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform

# 한글 폰트 설정
if platform.system() == "Windows":
    font_family = "Malgun Gothic"
elif platform.system() == "Darwin":
    font_family = "AppleGothic"
else:
    font_family = "NanumGothic"

mpl.rcParams['font.family'] = font_family
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = font_family
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로딩
df = pd.read_csv("data/목적별국적별입국소계제거.csv", encoding='cp949')
df = df.dropna()
df['입국자수'] = df['입국자수'].astype(float)

# 2. 월별 집계 및 피처
df['ds'] = pd.to_datetime(df['년'].astype(str) + df['월'].astype(str).str.zfill(2), format='%Y%m')
df_month = df.groupby('ds')['입국자수'].sum().reset_index().sort_values('ds')
df_month['성수기'] = df_month['ds'].dt.month.isin([7,8,12]).astype(int)
df_month['명절']  = df_month['ds'].dt.month.isin([1,2,9,10]).astype(int)

# 3. Train/Test split
train_end = pd.to_datetime('2024-05-01')
test_start = pd.to_datetime('2024-06-01')
test_end = pd.to_datetime('2025-05-01')
train_df = df_month[(df_month['ds'] <= train_end)].copy()
test_df  = df_month[(df_month['ds'] >= test_start) & (df_month['ds'] <= test_end)].copy()

# 4. Prophet: 명절/성수기/holiday 적용
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
prophet.fit(train_df.rename(columns={'입국자수':'y'}))
future = test_df[['ds', '성수기', '명절']].copy()
prophet_forecast = prophet.predict(future)
test_df['prophet_pred'] = prophet_forecast['yhat'].values

# 5. XGBoost (동일 방식: feature 추가)
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

xgb_window = 12
train_xgb = create_xgb_features(train_df, xgb_window)
test_xgb = create_xgb_features(pd.concat([train_df.tail(xgb_window), test_df], ignore_index=True), xgb_window)
features = [col for col in train_xgb.columns if col.startswith('lag_')] \
            + ['rolling_mean','rolling_std','diff','clipped']
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(train_xgb[features], train_xgb['입국자수'])
test_df['xgb_pred'] = xgb.predict(test_xgb[features])

# 6. LSTM: '성수기','명절' 다중 feature 활용 (24개월 시퀀스)
seq_len = 24  # ★ 24개월 시퀀스로 수정

all_df = pd.concat([train_df, test_df], ignore_index=True)
all_features = all_df[['입국자수', '성수기', '명절']].values
scaler = StandardScaler()
scaled_train = scaler.fit_transform(all_features[:len(train_df)])
scaled_all = scaler.transform(all_features)

X_lstm, y_lstm = [], []
for i in range(seq_len, len(train_df)):
    X_lstm.append(scaled_train[i-seq_len:i])
    y_lstm.append(scaled_train[i, 0])
X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

model_lstm = Sequential([
    LSTM(80, input_shape=(seq_len, 3), return_sequences=True),
    LSTM(40),
    Dense(16, activation='relu'),
    Dense(1)
])
model_lstm.compile(loss='mse', optimizer='adam')
model_lstm.fit(
    X_lstm, y_lstm,
    epochs=60,
    batch_size=8,
    verbose=0,
    callbacks=[EarlyStopping(patience=6, restore_best_weights=True)]
)

# Test 예측 (실제 미래 feature 사용)
X_lstm_test = []
for i in range(len(train_df), len(all_df)):
    X_lstm_test.append(scaled_all[i-seq_len:i])
X_lstm_test = np.array(X_lstm_test)
lstm_preds_scaled = model_lstm.predict(X_lstm_test, verbose=0)
lstm_pred_full = np.hstack([
    lstm_preds_scaled.reshape(-1,1),
    test_df[['성수기', '명절']].values
])
test_df['lstm_pred'] = scaler.inverse_transform(lstm_pred_full)[:,0]

# 7. Stacking (Meta Learner)
stack_train = pd.DataFrame({
    'prophet': test_df['prophet_pred'],
    'xgb': test_df['xgb_pred'],
    'lstm': test_df['lstm_pred']
})
meta = LinearRegression()
meta.fit(stack_train, test_df['입국자수'])
test_df['stacking_pred'] = meta.predict(stack_train)

# 8. 평가 및 시각화
def eval_metric(y_true, y_pred):
    return {
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }
print("\n[성능 비교]")
for model in ['prophet_pred','xgb_pred','lstm_pred','stacking_pred']:
    score = eval_metric(test_df['입국자수'], test_df[model])
    print(f"{model}: RMSE={score['RMSE']:.1f}, MAPE={score['MAPE']:.4f}")

# test_df를 csv로 저장 (모든 예측 결과 포함)
test_df.to_csv("data/24개월_test_df.csv", index=False, encoding='utf-8-sig')