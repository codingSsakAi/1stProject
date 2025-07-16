import re

def normalize_korean_input(input_str):
    """띄어쓰기 제거 + 앞뒤 공백 제거"""
    return re.sub(r"\s+", "", str(input_str).strip())

def valid_country(input_str, df):
    input_str = normalize_korean_input(input_str)
    if input_str == "":
        return None
    countries = df['국적'].unique()
    korean_countries = [c for c in countries if re.fullmatch(r"[가-힣]+", c)]
    if input_str in korean_countries:
        return input_str
    return False

def valid_purpose(input_str, df):
    input_str = normalize_korean_input(input_str)
    if input_str == "":
        return None
    purposes = df['목적'].unique()
    if input_str in purposes:
        return input_str
    return False

def valid_ym(input_str, df):
    input_str = normalize_korean_input(input_str)
    try:
        if len(input_str) != 6: return False
        y, m = int(input_str[:4]), int(input_str[4:])
        if not (2005 <= y <= 2030 and 1 <= m <= 12): return False
        return input_str
    except: return False

def get_user_input(df):
    print("\n[입력 안내]")
    print("▶ 국가명은 반드시 한글로 입력 (예: 일본, 중국, 미국). 띄어쓰기는 자동 제거됩니다.")
    print("▶ 목적은 데이터 기준 한글명(예: 관광, 상용 등). 띄어쓰기는 자동 제거됩니다.")
    print("▶ 예측 날짜는 200501~203012 범위의 6자리(예: 202501)로 입력.")
    while True:
        c = input("\n국가 입력(한글, 없으면 Enter): ")
        country = valid_country(c, df)
        if country is not False: break
        print("❗존재하지 않거나 한글이 아닌 국가입니다. 다시 입력하세요.")
    while True:
        p = input("목적 입력(한글, 없으면 Enter): ")
        purpose = valid_purpose(p, df)
        if purpose is not False: break
        print("❗존재하지 않는 목적입니다. 다시 입력하세요.")
    while True:
        d = input("예측 날짜 입력(YYYYMM): ")
        date = valid_ym(d, df)
        if date: break
        print("❗날짜는 200501~203012(6자리, 예: 202501)로 입력하세요.")
    return country, purpose, date
