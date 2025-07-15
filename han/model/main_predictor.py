import pandas as pd
from prophet_predictor import predict_prophet
from xgboost_predictor import predict_xgb

# 데이터 불러오기
df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv')

def valid_country(input_str, df):
    countries = df['국적'].unique()
    if input_str == "": return None
    if input_str in countries: return input_str
    return False

def valid_purpose(input_str, df):
    purposes = df['목적'].unique()
    if input_str == "": return None
    if input_str in purposes: return input_str
    return False

def valid_ym(input_str, df):
    # 200501 ~ 203012(최대 5년 뒤)
    try:
        if len(input_str)!=6: return False
        y, m = int(input_str[:4]), int(input_str[4:])
        if not (2005 <= y <= 2030 and 1 <= m <= 12): return False
        return input_str
    except: return False

def get_user_input():
    while True:
        c = input("국가 입력(없으면 Enter): ")
        country = valid_country(c, df)
        if country is not False: break
        print("❗존재하지 않는 국가입니다. 다시 입력하세요.")
    while True:
        p = input("목적 입력(없으면 Enter): ")
        purpose = valid_purpose(p, df)
        if purpose is not False: break
        print("❗존재하지 않는 목적입니다. 다시 입력하세요.")
    while True:
        d = input("예측 날짜 입력(YYYYMM): ")
        date = valid_ym(d, df)
        if date: break
        print("❗날짜는 200501~203012(6자리, 예: 202501)로 입력하세요.")
    return country, purpose, date

if __name__ == "__main__":
    print("⚡외국인 입국자 예측 프로그램")
    country, purpose, date = get_user_input()
    result_prophet = predict_prophet(df, country, purpose, date)
    result_xgb = predict_xgb(df, country, purpose, date)
    print("[Prophet] 예측값:", result_prophet['예측값'])
    print("[XGBoost] 예측값:", result_xgb['예측값'])
    # 시각화 및 해석 등 추가…
