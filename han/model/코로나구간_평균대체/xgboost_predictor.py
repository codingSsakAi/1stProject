import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def create_features(df):
    df = df.sort_values('ds')
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df['입국자수'].shift(lag)
    for w in [3, 6, 12]:
        df[f'roll_mean_{w}'] = df['입국자수'].rolling(w).mean()
    df['last_year'] = df['입국자수'].shift(12)
    df['월'] = df['ds'].dt.month
    df['연도'] = df['ds'].dt.year
    df['성수기'] = df['월'].isin([7,8,12]).astype(int)
    df['명절'] = df['월'].isin([1,2,9,10]).astype(int)
    df['코로나'] = (df['연도'] >= 2020).astype(int)
    df['연말'] = (df['월'] == 12).astype(int)
    df['연초'] = (df['월'] == 1).astype(int)
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def preprocess_data_xgb(df, country=None, purpose=None):
    if country: df = df[df['국적'] == country]
    if purpose: df = df[df['목적'] == purpose]
    df['ds'] = pd.to_datetime(df['연도'].astype(str) + df['월'].astype(str).str.zfill(2), format='%Y%m')
    df = df.groupby('ds', as_index=False)['입국자수'].sum()
    df = create_features(df)
    df = df.iloc[12:].reset_index(drop=True)
    X = df.drop(['ds','입국자수'], axis=1)
    y = df['입국자수']
    feature_cols = X.columns.tolist()
    return X, y, df, feature_cols

def predict_xgb(df, country, purpose, predict_ym):
    X, y, base_df, feature_cols = preprocess_data_xgb(df, country, purpose)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1)).flatten()
    split_idx = int(len(X_scaled) * 0.85)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.07, random_state=42)
    model.fit(X_train, y_train)
    # 검증 예측
    val_pred = model.predict(X_val)
    val_pred_rescaled = scaler_y.inverse_transform(val_pred.reshape(-1,1)).flatten()
    y_val_rescaled = scaler_y.inverse_transform(y_val.reshape(-1,1)).flatten()
    rmse = mean_squared_error(y_val_rescaled, val_pred_rescaled, squared=False)
    mape = mean_absolute_percentage_error(y_val_rescaled, val_pred_rescaled)
    r2 = r2_score(y_val_rescaled, val_pred_rescaled)

    # 미래 예측 시계열 (실측 마지막 ~ 예측달까지)
    last_ds = base_df['ds'].max()
    pred_date = pd.to_datetime(predict_ym, format='%Y%m')
    pred_months = pd.date_range(last_ds, pred_date, freq='MS')
    preds = []
    cur_base = base_df.copy()
    for ds in pred_months[1:]:
        last_row = cur_base.iloc[-1:].copy()
        month, year = ds.month, ds.year
        for lag in [1,3,6,12]:
            last_row[f'lag_{lag}'] = cur_base['입국자수'].iloc[-lag]
        for w in [3,6,12]:
            last_row[f'roll_mean_{w}'] = cur_base['입국자수'].iloc[-w:].mean()
        last_row['last_year'] = cur_base['입국자수'].iloc[-12]
        last_row['월'] = month
        last_row['연도'] = year
        last_row['성수기'] = int(month in [7,8,12])
        last_row['명절'] = int(month in [1,2,9,10])
        last_row['코로나'] = int(year >= 2020)
        last_row['연말'] = int(month == 12)
        last_row['연초'] = int(month == 1)
        feat_df = last_row.drop(['ds','입국자수'], axis=1)
        feat_df = feat_df[feature_cols]
        feat_scaled = scaler_x.transform(feat_df)
        pred_scaled = model.predict(feat_scaled)[0]
        pred = scaler_y.inverse_transform([[pred_scaled]])[0,0]
        preds.append({'ds': ds, '입국자수': pred})
        # 추가: 다음 달 예측을 위해 concat
        cur_base = pd.concat([cur_base, pd.DataFrame({'ds':[ds],'입국자수':[pred]})], ignore_index=True)
    pred_df = pd.DataFrame(preds)
    # 시계열 반환: 실측 마지막달~예측달까지
    result_series = pd.concat([
        base_df[['ds','입국자수']].iloc[-1:],
        pred_df
    ], ignore_index=True)
    return {
        '예측값': pred_df['입국자수'].iloc[-1] if not pred_df.empty else np.nan,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        '예측_시계열': result_series
    }
