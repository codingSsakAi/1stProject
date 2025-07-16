import pandas as pd
from prophet import Prophet

def preprocess_data(df, country=None, purpose=None):
    if country: df = df[df['국적'] == country]
    if purpose: df = df[df['목적'] == purpose]
    df['성수기'] = df['월'].isin([7, 8, 12]).astype(int)
    df['명절'] = df['월'].isin([1, 2, 9, 10]).astype(int)
    df['코로나'] = (df['연도'] >= 2020).astype(int)
    df['연초'] = (df['월'] == 1).astype(int)
    df['연말'] = (df['월'] == 12).astype(int)
    df['ds'] = pd.to_datetime(df['연도'].astype(str) + df['월'].astype(str).str.zfill(2), format='%Y%m')
    df = df.groupby(['ds', '성수기', '명절', '코로나', '연초', '연말'])['입국자수'].sum().reset_index()
    df.rename(columns={'입국자수': 'y'}, inplace=True)
    return df

def get_holidays():
    dates = [
        '2020-01-25', '2020-09-30', '2021-02-12', '2021-09-21', '2022-02-01', '2022-09-10', 
        '2023-01-22', '2023-09-29', '2024-02-10', '2024-09-17', '2025-01-29', '2025-10-06'
    ]
    holidays = pd.DataFrame({
        'holiday': '명절',
        'ds': pd.to_datetime(dates),
        'lower_window': -1,
        'upper_window': 2,
    })
    return holidays

def predict_prophet(df, country, purpose, predict_ym):
    data = preprocess_data(df.copy(), country, purpose)
    holidays = get_holidays()
    model = Prophet(
        yearly_seasonality=True,
        holidays=holidays,
        holidays_prior_scale=20,
        seasonality_prior_scale=10,
        changepoint_prior_scale=0.15
    )
    for reg in ['성수기', '명절', '코로나', '연초', '연말']:
        if reg != '명절':
            model.add_regressor(reg)
    model.fit(data)
    last = data['ds'].max()
    if predict_ym is not None:
        periods = (int(predict_ym[:4]) - last.year) * 12 + int(predict_ym[4:]) - last.month
    else:
        periods = 0
    if periods < 0: periods = 0
    future = model.make_future_dataframe(periods=periods, freq='MS')
    def add_regressors(future):
        future['월'] = future['ds'].dt.month
        future['연도'] = future['ds'].dt.year
        future['성수기'] = future['월'].isin([7, 8, 12]).astype(int)
        future['명절'] = future['월'].isin([1, 2, 9, 10]).astype(int)
        future['코로나'] = (future['연도'] >= 2020).astype(int)
        future['연초'] = (future['월'] == 1).astype(int)
        future['연말'] = (future['월'] == 12).astype(int)
        return future
    future = add_regressors(future)
    forecast = model.predict(future)
    pred_row = forecast[forecast['ds'] == pd.to_datetime(predict_ym, format='%Y%m')] if predict_ym is not None else pd.DataFrame()
    pred = pred_row['yhat'].values[0] if not pred_row.empty else None
    actual_df = data.rename(columns={'y': '입국자수'})
    forecast = forecast.rename(columns={'yhat': 'yhat'})
    return {
        '예측값': pred,
        '실제_시계열': actual_df[['ds', '입국자수']],
        '예측_시계열': forecast[['ds', 'yhat']]
    }
