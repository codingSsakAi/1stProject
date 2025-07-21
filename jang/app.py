from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from model import run_forecast
import pandas as pd

# .env 환경변수 불러오기
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
NAVER_CLIENT_ID = os.getenv('CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('CLIENT_SECRET')

app = Flask(__name__, static_folder='static', template_folder='templates')

# 데이터 경로 설정
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', '외국인입국자_전처리완료_딥러닝용.csv')
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
df.columns = df.columns.str.strip()

# 드롭다운용 리스트 생성
unique_countries = sorted(df['국적'].unique())
unique_purposes = sorted(df['목적'].unique())

@app.route('/api/countries')
def api_countries():
    return jsonify(["전체"] + unique_countries)

@app.route('/api/purposes')
def api_purposes():
    return jsonify(["전체"] + unique_purposes)

iso_map = {
    "대한민국": "KOR", "일본": "JPN", "미국": "USA", "중국": "CHN"
}

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return app.send_static_file(path)

@app.route('/api/iso3')
def api_iso3():
    return jsonify(iso_map)

@app.route('/api/predict', methods=['POST'])
def predict():
    req = request.get_json()
    combos = req['combos']
    start_ym = req['start_ym']
    end_ym = req['end_ym']

    start_year, start_month = map(int, start_ym.split('-'))
    end_year, end_month = map(int, end_ym.split('-'))

    all_results = []
    for combo in combos:
        country = combo['country']
        purpose = combo['purpose']

        for year in range(start_year, end_year + 1):
            month_start = start_month if year == start_year else 1
            month_end = end_month if year == end_year else 12
            예측월리스트 = list(range(month_start, month_end + 1))

            if purpose == '전체':
                purposes = df[df['국적'] == country]['목적'].unique()
                merged = []
                for p in purposes:
                    result = run_forecast(country, p, year, 예측월리스트)
                    if isinstance(result, list):
                        merged.extend(result)
                if not merged:
                    all_results.append({
                        'country': country,
                        'purpose': purpose,
                        'error': f'{country} 에 해당하는 목적별 예측 결과가 없습니다.'
                    })
                else:
                    all_results.extend(merged)
            else:
                result = run_forecast(country, purpose, year, 예측월리스트)
                if isinstance(result, list):
                    yms = [r['ym'] for r in result]
                    vals = [r['predicted'] for r in result]
                    all_results.append({
                        'country': country,
                        'purpose': purpose,
                        'yms': yms,
                        'values': vals,
                        'r2': '',
                        'mape': '',
                        'confidence': ''
                    })
                else:
                    all_results.append({'country': country, 'purpose': purpose, 'error': result['error']})

    return jsonify({'results': all_results})

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
        except:
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
            "국제박람회", "국제컨퍼런스", "한국 예정 이벤트", "한국 대회", "한국 콘서트",
            "동계 축제", "봄 축제"
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
    start = (page - 1) * page_size
    end = start + page_size
    paged_news = unique_news[start:end]

    return jsonify({"news": paged_news, "total": total, "page": page, "page_size": page_size})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
