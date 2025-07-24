# -*- coding: utf-8 -*-
import os
import glob
import re
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from prophet import Prophet

# 데이터 및 모델 경로 설정
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'data', '외국인입국자_전처리완료_딥러닝용.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# 특수문자·공백 제거 후 소문자화

def normalize(text):
    return re.sub(r'[^0-9a-z가-힣]', '', text.lower())

# 예측 함수
def run_forecast(country_input, purpose_input, predict_year, predict_months):
    # 데이터 로드
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['년월'] = pd.to_datetime(df['연도'].astype(str) + '-' + df['월'].astype(str).str.zfill(2))
    df_group = df[(df['국적']==country_input)&(df['목적']==purpose_input)].copy()
    if df_group.empty:
        return {'error': f'{country_input}/{purpose_input} 데이터가 없습니다.'}
    df_group = df_group.sort_values('년월').reset_index(drop=True)

    # 파생 변수 생성
    df_group['연도']    = df_group['년월'].dt.year
    df_group['월']      = df_group['년월'].dt.month
    df_group['lag1']    = df_group['입국자수'].shift(1).fillna(method='bfill')
    df_group['diff1']   = df_group['입국자수'].pct_change().fillna(0)
    df_group['ma3']     = df_group['입국자수'].rolling(3, min_periods=1).mean()
    df_group['ma6']     = df_group['입국자수'].rolling(6, min_periods=1).mean()
    df_group['yoy']     = df_group['입국자수'].pct_change(12).fillna(0)
    df_group['연도편차'] = df_group['연도'] - df_group['연도'].min()
    df_group['월_cos']  = np.cos(2*np.pi*df_group['월']/12)
    df_group['월_sin']  = np.sin(2*np.pi*df_group['월']/12)
    df_group['성수기여부'] = df_group['월'].isin([2,7,8]).astype(int)
    df_group['방학여부']   = df_group['월'].isin([1,7,8,12]).astype(int)
    df_group['학기여부']   = df_group['월'].isin([3,9]).astype(int)

    # 피처 정의
    base_cols = ['연도편차','월','월_cos','월_sin','성수기여부']
    extra_cols = ['lag1','diff1','ma3','ma6','yoy']
    feature_cols = base_cols + extra_cols
    if purpose_input=='관광': feature_cols.append('방학여부')
    if purpose_input=='유학연수': feature_cols.append('학기여부')

    # 모델 파일 로드
    def find_file(suf):
        files = glob.glob(os.path.join(MODEL_DIR, f"*{suf}"))
        if not files: return None
        nc, np_ = normalize(country_input), normalize(purpose_input)
        for f in files:
            if nc in normalize(os.path.basename(f)) and np_ in normalize(os.path.basename(f)):
                return f
        return files[0]

    scaler_path = find_file('_scaler.pkl'); xgb_path = find_file('_xgb.pkl')
    if not scaler_path or not xgb_path:
        return {'error':'모델 파일을 찾을 수 없습니다.'}
    scaler = joblib.load(scaler_path)
    xgb    = joblib.load(xgb_path)

    # 활성 피처 결정
    n_feats = getattr(scaler,'n_features_in_',None)
    if n_feats==len(base_cols):
        active_cols = base_cols
    elif n_feats==len(base_cols)+1:
        active_cols = base_cols + (['방학여부'] if purpose_input=='관광' else ['학기여부'])
    else:
        active_cols = feature_cols

    # Hold-out: XGB & Prophet 예측
    hist = df_group.copy(); val_size = min(6,len(hist)//3)
    X_hold = hist[active_cols].values[-val_size:]
    y_true = hist['입국자수'].values[-val_size:]
    y_xgb  = xgb.predict(scaler.transform(X_hold))
    if purpose_input in ['공용','관광','상용']:
        y_xgb = np.expm1(y_xgb)
    # Prophet hold-out
    prop_train = hist.iloc[:-val_size][['년월','입국자수']].rename(columns={'년월':'ds','입국자수':'y'})
    m_eval = Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
    m_eval.fit(prop_train)
    future_eval = m_eval.make_future_dataframe(periods=val_size,freq='MS')
    y_prop = m_eval.predict(future_eval)['yhat'].values[-val_size:]
    # 메타 학습: LinearRegression으로 최적 가중치 학습
    meta_X = np.vstack([y_xgb,y_prop]).T
    lr = LinearRegression(fit_intercept=False).fit(meta_X,y_true)
    w_xgb, w_prop = lr.coef_
    # 음수 방지 및 정규화
    w_xgb, w_prop = max(w_xgb,0), max(w_prop,0)
    s = w_xgb + w_prop
    if s>0:
        w_xgb, w_prop = w_xgb/s, w_prop/s
    else:
        w_xgb = w_prop = 0.5
    y_hybrid = w_xgb*y_xgb + w_prop*y_prop
    r2   = r2_score(y_true,y_hybrid)
    mape = mean_absolute_percentage_error(y_true,y_hybrid)*100
    rmse = np.sqrt(mean_squared_error(y_true,y_hybrid))

    # 미래 예측: XGB & Prophet
    last = df_group.iloc[-1]
    feats=[]
    for m in predict_months:
        d={'연도편차':predict_year-df_group['연도'].min(),'월':m}
        d['월_cos'],d['월_sin'] = np.cos(2*np.pi*m/12),np.sin(2*np.pi*m/12)
        d['성수기여부'] = int(m in [2,7,8]); d['lag1']=last['입국자수']
        d['diff1'],d['ma3'],d['ma6'],d['yoy'] = last['diff1'],last['ma3'],last['ma6'],last['yoy']
        if purpose_input=='관광': d['방학여부']=int(m in [1,7,8,12])
        if purpose_input=='유학연수': d['학기여부']=int(m in [3,9])
        feats.append(d)
    Xf = pd.DataFrame(feats)[active_cols].values
    xgb_pred = xgb.predict(scaler.transform(Xf));
    if purpose_input in ['공용','관광','상용']:
        xgb_pred = np.expm1(xgb_pred)
    # Prophet 미래
    pro_df = df_group[['년월','입국자수']].rename(columns={'년월':'ds','입국자수':'y'})
    m_prop = Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
    m_prop.fit(pro_df)
    future_prop = m_prop.make_future_dataframe(periods=len(predict_months),freq='MS')
    prop_vals = m_prop.predict(future_prop)['yhat'].values[-len(predict_months):]
    # 메타 가중치 적용
    hybrid = w_xgb*xgb_pred + w_prop*prop_vals
    preds = [int(round(v)) for v in hybrid]
    predictions = [{'ym':f"{predict_year}-{m:02d}",'predicted':preds[i]} for i,m in enumerate(predict_months)]

    return {
        'predictions': predictions,
        'r2': round(r2,4),
        'mape': round(mape,2),
        'rmse': round(rmse,2)
    }
