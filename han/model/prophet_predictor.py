import pandas as pd
from prophet import Prophet

def preprocess_data(df, country=None, purpose=None):
    # 필터링
    if country: df = df[df['국적'] == country]
    if purpose: df = df[df['목적'] == purpose]
    # 월 컬럼 생성
    df['ds'] = pd.to_datetime(df['연도'].astype(str) + df['월'].astype(str).str.zfill(2), format='%Y%m')
    df = df.groupby('ds')['입국자수'].sum().reset_index()
    df.rename(columns={'입국자수':'y'}, inplace=True)
    return df

def predict_prophet(df, country, purpose, predict_ym):
    # 데이터 전처리
    data = preprocess_data(df, country, purpose)
    # 모델 학습
    model = Prophet()
    model.fit(data)
    # 예측용 future 생성
    last = data['ds'].max()
    periods = (int(predict_ym[:4]) - last.year)*12 + int(predict_ym[4:]) - last.month
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    # 예측값 추출
    pred_row = forecast[forecast['ds']==pd.to_datetime(predict_ym, format='%Y%m')]
    pred = pred_row['yhat'].values[0] if not pred_row.empty else None
    # 결과 리턴: (여기서 실제 오차/정확도 계산 로직 추가)
    return {
        '예측값': pred,
        '예측_시계열': forecast[['ds','yhat']],
        '적용특이사항': [], # 추가 Feature 엔지니어링 필요
        '정확도': None,    # 테스트셋 있으면 계산
        '오차': None
    }


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv')
    # 예시 입력
    country = 'JAPAN'
    purpose = '관광'
    date = '202407'
    result = predict_prophet(df, country, purpose, date)   # prophet_predictor.py
    # result = predict_xgb(df, country, purpose, date)     # xgboost_predictor.py
    print(result)
