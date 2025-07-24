# app.py
# [외국인 방문객 예측 서비스] 백엔드 (2025-07-24: 국가/목적/기간 기준 로직 완전 반영)
from flask import Flask, request, jsonify, render_template
import pandas as pd, os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__, static_folder='static', template_folder='templates')

# 데이터 준비 (국가/목적/기간별 실제·예측값)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', '실제데이터.csv')
FORECAST_PATH = os.path.join(os.path.dirname(__file__), 'data', '2025-06_2026-12_예측값.csv')
df_real = pd.read_csv(DATA_PATH, encoding='cp949')
df_real['년'] = df_real['년'].astype(int); df_real['월'] = df_real['월'].astype(int)
df_real['연월'] = df_real['년'].astype(str) + '-' + df_real['월'].astype(str).str.zfill(2)
df_fore = pd.read_csv(FORECAST_PATH, encoding='utf-8-sig'); df_fore['연월'] = df_fore['연월'].astype(str)

# 국가별 입국자 합계 기준 내림차순
country_total = df_real.groupby('국적')['입국자수'].sum().sort_values(ascending=False)
unique_countries = list(country_total.index)
# 목적 지정 순서로
purpose_order = ['전체', '관광', '유학연수', '상용', '공용']
real_purposes = [p for p in df_real['목적'].unique() if p in purpose_order]
unique_purposes = [p for p in purpose_order if p in real_purposes]

# 연월 범위 세팅 (예측가능 구간: 2025-06 ~ 2026-12)
PREDICT_START = '2025-06'
PREDICT_END = '2026-12'
def make_ym_list(start_ym, end_ym):
    return pd.date_range(start=start_ym+'-01', end=end_ym+'-01', freq='MS').strftime('%Y-%m').tolist()

@app.route('/')
def index(): return render_template('index.html')
@app.route('/predict')
def predict(): return render_template('predict.html')
@app.route('/about')
def about(): return render_template('about.html')

@app.route('/api/countries')
def api_countries():
    # '러시아(연방)' 제외, ‘전체’는 프론트에서 추가
    filtered = [c for c in unique_countries if c != "러시아(연방)"]
    return jsonify(filtered)

@app.route('/api/purposes')
def api_purposes():
    # ‘전체’는 프론트에서 추가
    return jsonify([p for p in unique_purposes if p != "기타"])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    req = request.get_json()
    combos = req['combos']
    start_ym = req['start_ym']
    end_ym = req['end_ym']
    
    # 실제 구간: 2021-06 ~ 2024-06(3개월 단위), 2024-06~2025-05(월별), 2025-06~2026-12(예측)
    yms = []
    # 1. 2021-06 ~ 2024-06, 3개월 단위
    q3_months = pd.date_range(start='2021-06-01', end='2024-06-01', freq='3MS').strftime('%Y-%m').tolist()
    # 2. 2024-06 ~ 2025-05, 월별
    monthly = pd.date_range(start='2024-06-01', end='2025-05-01', freq='MS').strftime('%Y-%m').tolist()
    # 3. 2025-06 ~ 2026-12, 월별(예측) - 전체 예측 구간
    all_pred_months = pd.date_range(start='2025-06-01', end='2026-12-01', freq='MS').strftime('%Y-%m').tolist()
    
    # 그래프 x축: 항상 2021-06 ~ 검색 종료월까지
    all_yms = q3_months + [x for x in monthly if x not in q3_months] + [x for x in all_pred_months if x not in q3_months and x not in monthly]
    # 실제 검색 범위
    graph_end = end_ym
    yms = [ym for ym in all_yms if ym <= graph_end]
    
    # ===== 수정된 부분: 전체 예측 구간을 사용자 검색 범위와 분리 =====
    # 사용자가 검색한 예측 범위: 2025-06~2026-12 중 사용자가 선택한 시작~끝
    user_predict_yms = [ym for ym in all_pred_months if start_ym <= ym <= end_ym]
    # 전체 예측 범위: 2025-06~그래프 끝까지 (중간 구간 포함)
    all_predict_yms = [ym for ym in all_pred_months if ym <= graph_end]

    real_pivot = df_real.pivot_table(index='연월', columns=['국적', '목적'], values='입국자수', aggfunc='sum', fill_value=0)
    fore_pivot = df_fore.pivot_table(index='연월', columns=['국가', '목적'], values='예측입국자수', aggfunc='sum', fill_value=0)
    
    results = []
    for combo in combos:
        c, p = combo['country'], combo['purpose']
        vals, is_actual, r2, mape, conf = [], [], [], [], []
        
        for ym in yms:
            # ===== 수정된 조건: 전체 예측 구간에서 데이터 로드 =====
            if ym in all_predict_yms:  # 2025-06 이후 모든 예측 구간
                # 예측값 로드
                if c == "전체" and p == "전체": 
                    v = fore_pivot.loc[ym].sum() if ym in fore_pivot.index else 0
                elif c == "전체": 
                    v = sum(val for (cc, pp), val in (fore_pivot.loc[ym] if ym in fore_pivot.index else {}).items() if pp == p)
                elif p == "전체": 
                    v = sum(val for (cc, pp), val in (fore_pivot.loc[ym] if ym in fore_pivot.index else {}).items() if cc == c)
                else: 
                    v = fore_pivot.loc[ym][(c, p)] if ym in fore_pivot.index and (c, p) in fore_pivot.columns else 0
                
                is_actual.append(False)
                
                # 통계값은 사용자 검색 구간만 적용
                if ym in user_predict_yms:
                    row = df_fore[(df_fore['국가'] == c) & (df_fore['목적'] == p) & (df_fore['연월'] == ym)]
                    r2.append(float(row.iloc[0]['r2']) if len(row) and not pd.isna(row.iloc[0]['r2']) else None)
                    mape.append(float(row.iloc[0]['mape']) if len(row) and not pd.isna(row.iloc[0]['mape']) else None)
                    conf.append(float(row.iloc[0]['confidence']) if len(row) and not pd.isna(row.iloc[0]['confidence']) else None)
                else:
                    # 중간 구간은 통계값 없음
                    r2.append(None)
                    mape.append(None)
                    conf.append(None)
                    
            else:
                # 실제값 로드 (2025-05까지)
                if c == "전체" and p == "전체": 
                    v = real_pivot.loc[ym].sum() if ym in real_pivot.index else 0
                elif c == "전체": 
                    v = sum(val for (cc, pp), val in (real_pivot.loc[ym] if ym in real_pivot.index else {}).items() if pp == p)
                elif p == "전체": 
                    v = sum(val for (cc, pp), val in (real_pivot.loc[ym] if ym in real_pivot.index else {}).items() if cc == c)
                else: 
                    v = real_pivot.loc[ym][(c, p)] if ym in real_pivot.index and (c, p) in real_pivot.columns else 0
                
                is_actual.append(True)
                # 실제값은 통계값 없음
                r2.append(None)
                mape.append(None)
                conf.append(None)
            
            vals.append(int(v) if not pd.isna(v) else 0)
        
        results.append({
            "country": c, 
            "purpose": p, 
            "yms": yms, 
            "values": vals, 
            "is_actual": is_actual, 
            "r2": r2, 
            "mape": mape, 
            "confidence": conf, 
            "predict_yms": user_predict_yms  # 사용자 검색 구간만 전달
        })
    
    return jsonify({"results": results})

# 뉴스 API (필터·정렬·중복제거)
BAD_WORDS = ["사망","사고","사건","범죄","폭력","논란","사기","불법","피해","징역","재판","폭우","화재","감염","확진","부상","부정","문제","논란",
"총선","대선","정당","의원","국회","대통령","정치","정책","청와대","선거","야당","여당","국회의원","보수","진보"]
def naver_news_search(query, display=50, sort='date'):
    import requests
    url, cid, csec = "https://openapi.naver.com/v1/search/news.json", os.getenv("CLIENT_ID"), os.getenv("CLIENT_SECRET")
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}
    res = requests.get(url, headers=headers, params={"query": query, "display": display, "start": 1, "sort": sort})
    return res.json()['items'] if res.status_code == 200 else []

def filter_after_date_and_badwords(items, min_date):
    result = []
    for item in items:
        try:
            pub_date = datetime.strptime(item['pubDate'], "%a, %d %b %Y %H:%M:%S %z")
            text = (item['title'] or '') + ' ' + (item['description'] or '')
            if any(bad in text for bad in BAD_WORDS): continue
            if pub_date >= min_date:
                result.append({"title": item['title'], "description": item['description'], "link": item['link'], "pubDate": pub_date.strftime("%Y-%m-%d")})
        except: continue
    return result

@app.route('/api/news')
def api_news():
    keywords = (request.args.get('keywords') or "").split(',') if request.args.get('keywords') else [
        "한국 축제", "한국 행사", "서울 전시회", "K-POP 콘서트", "외국인 체험 프로그램",
        "국제박람회", "국제컨퍼런스", "한국 예정 이벤트", "한국 대회", "한국 콘서트", "동계 축제", "봄 축제"
    ]
    page = int(request.args.get('page', 1)); page_size = 20
    min_date = datetime(2025, 5, 1, tzinfo=datetime.now().astimezone().tzinfo)
    all_news = [n for kw in keywords for n in filter_after_date_and_badwords(naver_news_search(kw, 50, 'date'), min_date)]
    # 중복 제거 및 정렬
    seen, unique_news = set(), []
    for n in sorted(all_news, key=lambda x: x['pubDate'], reverse=True):
        if n['link'] not in seen:
            seen.add(n['link']); unique_news.append(n)
    start, end = (page-1)*page_size, (page)*page_size
    return jsonify({"news": unique_news[start:end], "total": len(unique_news), "page": page, "page_size": page_size})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
