# predictor_interface.py
from prophet_predictor import predict_prophet
from prophet_xgboost import predict_xgboost

def get_user_input(country_list, purpose_list, date_min, date_max, country_map):
    # 국가 입력
    while True:
        country = input("국가명(영문/한글) 입력 [Enter: 전체]: ").strip()
        if not country:
            country = None
            break
        # 대소문자 변환, 영어 → 한글 변환
        country = country.lower()
        if country in country_map:
            country = country_map[country]
        if country in country_list:
            break
        print("올바른 국가명을 입력하세요. (예: korea, 일본, usa 등)")
    
    # 목적 입력
    while True:
        purpose = input("방문 목적 입력 [Enter: 전체]: ").strip()
        if not purpose:
            purpose = None
            break
        if purpose in purpose_list:
            break
        print(f"올바른 목적을 입력하세요. (선택지: {', '.join(purpose_list)})")
    
    # 날짜 입력
    while True:
        date = input(f"예측할 날짜(YYYYMM) 입력 [{date_min}~{date_max}]: ").strip()
        if not date.isdigit() or not (date_min <= date <= date_max):
            print(f"올바른 날짜(YYYYMM) 범위를 입력하세요. ({date_min}~{date_max})")
        else:
            break
    
    return country, purpose, date

def main_predict():
    # 데이터 불러오기, 국가/목적 목록, 국가맵핑 추출
    df = load_data("../data/외국인입국자_전처리완료_딥러닝용.csv")
    country_list = sorted(df['국적'].unique())
    purpose_list = sorted(df['목적'].unique())
    date_min = "200501"
    date_max = "203006"
    country_map = {'korea': '한국', 'japan': '일본', 'china': '중국', 'usa': '미국', ...}  # 실제 매핑 필요

    # 입력 받기
    country, purpose, date = get_user_input(country_list, purpose_list, date_min, date_max, country_map)
    
    # Feature Engineering (내부 함수화)
    features = make_features(date, country, purpose)
    
    # Prophet 예측
    result_prophet = predict_prophet(df, country, purpose, date, features)
    # XGBoost 예측
    result_xgb = predict_xgboost(df, country, purpose, date, features)
    
    # 결과 해석, 그래프, 오차 등 통합 출력
    print_results(result_prophet, result_xgb)
    plot_results(result_prophet, result_xgb, features)
