from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
import random

app = Flask(__name__, static_folder='../frontend', static_url_path='')

### 1. 실제 데이터 로드 및 연월 컬럼 생성
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', '실제데이터.csv')
df = pd.read_csv(DATA_PATH, encoding='cp949')
df['년'] = df['년'].astype(int)
df['월'] = df['월'].astype(int)
df['연월'] = df['년'].astype(str) + '-' + df['월'].astype(str).str.zfill(2)

min_ym = df['연월'].min()
max_ym = df['연월'].max()

actual_dict = {
    (row['국적'], row['목적'], row['연월']): int(row['입국자수'])
    for _, row in df.iterrows()
}

unique_countries = sorted(df['국적'].unique())
unique_purposes = sorted(df['목적'].unique())

def make_ym_list(start_ym, end_ym):
    return pd.date_range(start=start_ym+'-01', end=end_ym+'-01', freq='MS').strftime('%Y-%m').tolist()

def calc_sum(country, purpose, ym):
    """전체/합계용: 인자로 None이 오면 전체를 의미"""
    if country == "전체" and purpose == "전체":
        v = df[df['연월'] == ym]['입국자수'].sum()
    elif country == "전체":
        v = df[(df['목적'] == purpose) & (df['연월'] == ym)]['입국자수'].sum()
    elif purpose == "전체":
        v = df[(df['국적'] == country) & (df['연월'] == ym)]['입국자수'].sum()
    else:
        v = actual_dict.get((country, purpose, ym), 0)
    return int(v) if not pd.isna(v) else 0

def get_values(country, purpose, start_ym, end_ym):
    if (start_ym < min_ym) or (end_ym > max_ym):
        return [], [], [], f'데이터 범위는 {min_ym} ~ {max_ym}입니다.'
    yms = make_ym_list(start_ym, end_ym)
    values, is_actual = [], []
    for ym in yms:
        v = calc_sum(country, purpose, ym)
        values.append(v)
        is_actual.append(True)
    return yms, values, is_actual, None

iso_map = {
    "대한민국": "KOR", "일본": "JPN", "미국": "USA", "중국": "CHN",
    # 실제데이터에 존재하는 국가만 추가 (필요시 확장)
}

@app.route('/')
def root():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/countries')
def api_countries():
    # 1번 드롭다운에는 "전체" 반드시 포함, 2번에도 포함
    return jsonify(["전체"] + unique_countries)

@app.route('/api/purposes')
def api_purposes():
    # 1번 드롭다운에는 "전체" 반드시 포함, 2번에도 포함
    return jsonify(["전체"] + unique_purposes)

@app.route('/api/iso3')
def api_iso3():
    return jsonify({k: iso_map.get(k, '') for k in unique_countries})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    req = request.get_json()
    combos = req['combos']    # [{"country":..., "purpose":...}, ...]
    start_ym = req['start_ym']
    end_ym = req['end_ym']
    results = []
    for combo in combos:
        country = combo['country']
        purpose = combo['purpose']
        if country == "미선택" or purpose == "미선택":
            results.append(None)
            continue
        yms, values, is_actual, err = get_values(country, purpose, start_ym, end_ym)
        if err:
            results.append({"error": err})
        else:
            results.append({
                "country": country,
                "purpose": purpose,
                "yms": yms,
                "values": values,
                "is_actual": is_actual
            })
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
