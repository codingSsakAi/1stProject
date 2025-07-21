from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# .env 환경변수 불러오기
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
NAVER_CLIENT_ID = os.getenv('CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('CLIENT_SECRET')

app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates'
)

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
    if pd.isna(v): return 0
    try: return int(v)
    except: return 0

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
def index():
    return render_template('index.html')

@app.route('/api/countries')
def api_countries():
    return jsonify(["전체"] + unique_countries)

@app.route('/api/purposes')
def api_purposes():
    return jsonify(["전체"] + unique_purposes)

@app.route('/api/iso3')
def api_iso3():
    return jsonify({k: iso_map.get(k, '') for k in unique_countries})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    req = request.get_json()
    combos = req['combos']
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

# 뉴스 API 영역 (생략 시 기존 코드 그대로 두면 됩니다)
BAD_WORDS = [
    "사망", "사고", "사건", "범죄", "폭력", "논란", "사기", "불법", "피해", "징역", "재판",
    "폭우", "화재", "감염", "확진", "부상", "부정", "문제", "논란",
    "총선", "대선", "정당", "의원", "국회", "대통령", "정치", "정책", "청와대", "선거",
    "야당", "여당", "국회의원", "보수", "진보"
]
def naver_news_search(query, display=50, sort='date'):
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {
        "query": query,
        "display": display,
        "start": 1,
        "sort": sort
    }
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        return res.json()['items']
    else:
        print('Naver API Error:', res.status_code, res.text)
        return []

def filter_after_date_and_badwords(items, min_date):
    result = []
    for item in items:
        try:
            pub_date = datetime.strptime(item['pubDate'], "%a, %d %b %Y %H:%M:%S %z")
            text = (item['title'] or '') + ' ' + (item['description'] or '')
            if any(bad in text for bad in BAD_WORDS):
                continue
            if pub_date >= min_date:
                result.append({
                    "title": item['title'],
                    "description": item['description'],
                    "link": item['link'],
                    "pubDate": pub_date.strftime("%Y-%m-%d")
                })
        except Exception as e:
            continue
    return result

@app.route('/api/news')
def api_news():
    keywords = request.args.get('keywords')
    page = int(request.args.get('page', 1))
    page_size = 20
    if keywords:
        keywords = keywords.split(',')
    else:
        keywords = [
            "한국 축제", "한국 행사", "서울 전시회", "K-POP 콘서트", "외국인 체험 프로그램",
            "국제박람회", "국제컨퍼런스", "한국 예정 이벤트", "한국 대회", "한국 콘서트", "동계 축체", "봄 축제"
        ]
    min_date = datetime(2025, 5, 1, tzinfo=datetime.now().astimezone().tzinfo)
    all_news = []
    for kw in keywords:
        items = naver_news_search(kw, display=50, sort='date')
        filtered = filter_after_date_and_badwords(items, min_date)
        all_news.extend(filtered)
    seen = set()
    unique_news = []
    for n in all_news:
        if n['link'] not in seen:
            seen.add(n['link'])
            unique_news.append(n)
    unique_news = sorted(unique_news, key=lambda x: x['pubDate'], reverse=True)
    total = len(unique_news)
    start = (page-1)*page_size
    end = start + page_size
    paged_news = unique_news[start:end]
    return jsonify({"news": paged_news, "total": total, "page": page, "page_size": page_size})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
