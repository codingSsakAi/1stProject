from flask import Flask, render_template, request, jsonify
import random
from datetime import datetime

app = Flask(__name__)

def generate_data(country, purpose, start_ym, end_ym):
    yms = []
    values = []
    is_actual = []
    ym = datetime.strptime("2025-01", "%Y-%m")
    last_actual = datetime.strptime("2025-03", "%Y-%m")
    while ym <= last_actual:
        yms.append(ym.strftime("%Y-%m"))
        values.append(random.randint(300, 1200))
        is_actual.append(True)
        ym = ym.replace(day=1)
        ym = ym.replace(month=ym.month + 1 if ym.month < 12 else 1, year=ym.year if ym.month < 12 else ym.year + 1)
    ym = datetime.strptime("2025-04", "%Y-%m")
    last_predict = datetime.strptime("2025-12", "%Y-%m")
    while ym <= last_predict:
        yms.append(ym.strftime("%Y-%m"))
        values.append(random.randint(800, 1800))
        is_actual.append(False)
        ym = ym.replace(day=1)
        ym = ym.replace(month=ym.month + 1 if ym.month < 12 else 1, year=ym.year if ym.month < 12 else ym.year + 1)
    # filter
    start_idx = yms.index(start_ym)
    end_idx = yms.index(end_ym) + 1
    return {
        "yms": yms[start_idx:end_idx],
        "values": values[start_idx:end_idx],
        "is_actual": is_actual[start_idx:end_idx]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    country = data.get("country", "전체")
    purpose = data.get("purpose", "전체")
    start_ym = data.get("start_ym", "2025-01")
    end_ym = data.get("end_ym", "2025-12")
    result = generate_data(country, purpose, start_ym, end_ym)
    return jsonify(result)

@app.route('/api/countries')
def api_countries():
    countries = ["전체", "일본", "중국", "미국", "베트남", "필리핀", "태국", "인도네시아", "러시아", "몽골",
                 "말레이시아", "홍콩", "싱가포르", "캄보디아", "인도", "호주", "프랑스", "독일", "영국", "캐나다"]
    return jsonify(countries)

@app.route('/api/purposes')
def api_purposes():
    purposes = ["전체", "관광", "상용", "교육", "공용", "기타"]
    return jsonify(purposes)

if __name__ == '__main__':
    app.run(debug=True)
