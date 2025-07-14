import pandas as pd
from xgboost import XGBRegressor

def preprocess_data_xgb(df, country=None, purpose=None):
    # 필터링
    if country: df = df[df['국적'] == country]
    if purpose: df = df[df['목적'] == purpose]
    # 특이사항 Feature 추가
    df['성수기'] = df['월'].isin([7,8,12]).astype(int)
    df['명절'] = df['월'].isin([1,2,9,10]).astype(int)
    df['코로나'] = (df['연도']>=2020).astype(int)
    # 날짜 인코딩
    df['ym'] = df['연도']*100 + df['월']
    X = df[['ym','성수기','명절','코로나']]
    y = df['입국자수']
    return X, y

def predict_xgb(df, country, purpose, predict_ym):
    X, y = preprocess_data_xgb(df, country, purpose)
    model = XGBRegressor()
    model.fit(X, y)
    # 예측 row 생성
    ym = int(predict_ym)
    year, month = ym//100, ym%100
    features = pd.DataFrame([{
        'ym': ym,
        '성수기': int(month in [7,8,12]),
        '명절': int(month in [1,2,9,10]),
        '코로나': int(year>=2020)
    }])
    pred = model.predict(features)[0]
    # 결과 리턴(정확도/오차 등 추가)
    return {
        '예측값': pred,
        '예측_피쳐': features,
        '정확도': None,
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
