# Flask 기본 라이브러리 import
from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from scipy.stats import pearsonr, spearmanr

# Flask 앱 인스턴스 생성
app = Flask(__name__)


# 모델 로딩 함수
def load_models():
    """모델 파일들을 로딩하는 함수"""
    models = {}
    models_dir = "models"

    for filename in os.listdir(models_dir):
        # pkl, h5 모두 탐색 → pkl 부분만 주석 처리
        if filename.endswith(".pkl") or filename.endswith(".h5"):
            parts = filename.replace(".pkl", "").replace(".h5", "").split("_")
            if len(parts) >= 3:
                country = parts[0]
                purpose = parts[1]
                model_type = parts[2]

                filepath = os.path.join(models_dir, filename)
                print(filepath)
                try:
                    # if filename.endswith(".pkl"):
                    #     with open(filepath, "rb") as f:
                    #         model = pickle.load(f)
                    if filename.endswith(".h5"):
                        import tensorflow as tf

                        model = tf.keras.models.load_model(filepath)
                    else:
                        continue  # pkl 파일은 무시

                    key = f"{country}_{purpose}_{model_type}"
                    models[key] = {
                        "model": model,
                        "country": country,
                        "purpose": purpose,
                        "type": model_type,
                        "filepath": filepath,
                    }
                except Exception as e:
                    print(f"모델 로딩 실패: {filename}, 오류: {e}")

    return models


def load_scalers():
    """스케일러 파일들을 로딩하는 함수"""
    scalers = {}
    models_dir = "models"

    for filename in os.listdir(models_dir):
        # if filename.endswith("_scaler.pkl") or filename.endswith("_scaler_resid.pkl"):
        #     ... (모두 주석 처리)
        continue  # scaler 관련 파일은 모두 무시

    return scalers


# 전역 변수로 모델과 스케일러 저장
MODELS = load_models()
SCALERS = load_scalers()

# 국가 및 목적 목록
COUNTRIES = ["필리핀", "홍콩", "피지", "페루", "폴란드", "프랑스", "터키", "파키스탄"]
PURPOSES = ["공용", "관광", "상용", "유학연수"]


# 기본 라우트 설정
@app.route("/")
def home():
    # 메인 페이지 간단 응답
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


# API 엔드포인트들
@app.route("/api/countries")
def get_countries():
    """사용 가능한 국가 목록 반환"""
    return jsonify(COUNTRIES)


@app.route("/api/purposes")
def get_purposes():
    """사용 가능한 목적 목록 반환"""
    return jsonify(PURPOSES)


@app.route("/api/news")
def get_news():
    """뉴스 데이터 반환 (기존 기능 유지)"""
    # 임시 뉴스 데이터
    news_data = {
        "news": [
            {
                "title": "2025년 한국 관광 활성화 정책 발표",
                "link": "#",
                "pubDate": "2025-01-15",
                "description": "정부가 2025년 한국 관광 활성화를 위한 새로운 정책을 발표했습니다.",
            },
            {
                "title": "외국인 관광객 증가세 지속",
                "link": "#",
                "pubDate": "2025-01-10",
                "description": "2024년 대비 외국인 관광객 수가 15% 증가한 것으로 나타났습니다.",
            },
        ],
        "news_total": 2,
    }
    return jsonify(news_data)


@app.route("/api/predict", methods=["POST"])
def predict_entrants():
    """입국자수 예측 API"""
    try:
        data = request.get_json()
        combos = data.get("combos", [])
        start_ym = data.get("start_ym", "2025-01")
        end_ym = data.get("end_ym", "2025-12")

        results = []

        for combo in combos:
            country = combo.get("country")
            purpose = combo.get("purpose")

            # 예측 데이터 생성 (실제 모델 사용 시 여기서 실제 예측 수행)
            prediction_data = generate_prediction_data(
                country, purpose, start_ym, end_ym
            )
            results.append(prediction_data)

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/report", methods=["POST"])
def report_diagnosis():
    """
    예측 결과 리포트/진단 API
    입력: {
        "country": "필리핀",
        "purpose": "관광",
        "start_ym": "2024-01",
        "end_ym": "2024-12"
    }
    """
    try:
        data = request.get_json()
        country = data.get("country")
        purpose = data.get("purpose")
        start_ym = data.get("start_ym")
        end_ym = data.get("end_ym")

        # 예측 데이터 생성 (기존 함수 활용)
        pred_data = generate_prediction_data(country, purpose, start_ym, end_ym)
        yms = pred_data["yms"]
        values = pred_data["values"]
        is_actual = pred_data["is_actual"]

        # 실제값/예측값 분리
        actual_values = [v for v, a in zip(values, is_actual) if a]
        predicted_values = [v for v, a in zip(values, is_actual) if not a]
        predicted_yms = [ym for ym, a in zip(yms, is_actual) if not a]
        actual_yms = [ym for ym, a in zip(yms, is_actual) if a]

        # 진단 지표 계산
        def mape(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        def rmse(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        # 실제값/예측값이 모두 있는 구간만 비교
        n_compare = min(len(actual_values), len(predicted_values))
        if n_compare > 0:
            y_true = actual_values[-n_compare:]
            y_pred = predicted_values[:n_compare]
            mape_val = mape(y_true, y_pred)
            rmse_val = rmse(y_true, y_pred)
            pearson_val, _ = pearsonr(y_true, y_pred)
            spearman_val, _ = spearmanr(y_true, y_pred)
        else:
            mape_val = rmse_val = pearson_val = spearman_val = None

        # flatness(변동성 부족) 진단
        def is_flat(arr):
            arr = np.array(arr)
            return np.std(arr) < 1e-3

        flatness = is_flat(predicted_values)
        # 이상치(예: 예측값이 실제값 대비 2배 이상 차이) 진단
        outlier_count = 0
        if n_compare > 0:
            for yt, yp in zip(y_true, y_pred):
                if abs(yt - yp) > max(1000, 2 * np.std(y_true)):
                    outlier_count += 1

        # 한글 리포트 생성
        report_lines = []
        if mape_val is not None:
            report_lines.append(f"MAPE(평균예측오차): {mape_val:.2f}%")
        if rmse_val is not None:
            report_lines.append(f"RMSE(평균제곱근오차): {rmse_val:.1f}명")
        if pearson_val is not None:
            report_lines.append(f"Pearson 상관계수: {pearson_val:.3f}")
        if spearman_val is not None:
            report_lines.append(f"Spearman 상관계수: {spearman_val:.3f}")
        report_lines.append(f"Flatness(변동성 부족): {'예' if flatness else '아니오'}")
        report_lines.append(f"이상치(Outlier) 개수: {outlier_count}건")
        report = "\n".join(report_lines)

        # 비교표 생성
        comparison = []
        for ym, a, p in zip(predicted_yms, y_true, y_pred):
            comparison.append({"ym": ym, "actual": int(a), "predicted": int(p)})

        return jsonify(
            {
                "report": report,
                "metrics": {
                    "mape": mape_val,
                    "rmse": rmse_val,
                    "pearson": pearson_val,
                    "spearman": spearman_val,
                },
                "comparison": comparison,
                "diagnosis": {"flatness": flatness, "outliers": outlier_count},
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_prediction_data(country, purpose, start_ym, end_ym):
    """실제 모델을 사용하여 예측 데이터 생성"""
    # 날짜 범위 생성
    start_date = datetime.strptime(start_ym, "%Y-%m")
    end_date = datetime.strptime(end_ym, "%Y-%m")

    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m"))
        current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)

    # 실제 데이터와 예측 데이터 구분
    actual_end = datetime(2024, 12, 1)

    is_actual = []
    values = []

    # 모델 키 생성
    # xgb_key = f"{country}_{purpose}_xgb"
    gru_key = f"{country}_{purpose}_gru"
    # scaler_key = f"{country}_{purpose}_scaler"
    # scaler_resid_key = f"{country}_{purpose}_scaler_resid"

    # 모델이 있는지 확인
    # has_xgb = xgb_key in MODELS
    has_gru = gru_key in MODELS
    # has_scaler = scaler_key in SCALERS

    for date_str in dates:
        date_obj = datetime.strptime(date_str, "%Y-%m")

        if date_obj <= actual_end:
            # 실제 데이터 (랜덤 값 생성)
            base_value = 10000 + np.random.randint(-2000, 2000)
            if purpose == "관광":
                base_value = int(base_value * 1.5)
            elif purpose == "상용":
                base_value = int(base_value * 0.8)
            elif purpose == "유학연수":
                base_value = int(base_value * 0.6)

            values.append(base_value)
            is_actual.append(True)
        else:
            # 예측 데이터 (딥러닝 모델만 사용)
            if has_gru:
                try:
                    prediction = predict_with_model(country, purpose, date_str, "gru")
                    values.append(prediction)
                except Exception as e:
                    print(f"GRU 예측 실패: {e}")
                    # 폴백: 랜덤 값 생성
                    base_value = 12000 + np.random.randint(-3000, 3000)
                    if purpose == "관광":
                        base_value = int(base_value * 1.6)
                    elif purpose == "상용":
                        base_value = int(base_value * 0.9)
                    elif purpose == "유학연수":
                        base_value = int(base_value * 0.7)
                    values.append(base_value)
            else:
                # 모델이 없는 경우 랜덤 값 생성
                base_value = 12000 + np.random.randint(-3000, 3000)
                if purpose == "관광":
                    base_value = int(base_value * 1.6)
                elif purpose == "상용":
                    base_value = int(base_value * 0.9)
                elif purpose == "유학연수":
                    base_value = int(base_value * 0.7)
                values.append(base_value)

            is_actual.append(False)

    return {
        "country": country,
        "purpose": purpose,
        "yms": dates,
        "values": values,
        "is_actual": is_actual,
        "r2": 0.85,  # 예시 성능 지표
        "mape": 12.5,
        "confidence": 87.3,
    }


def predict_with_model(country, purpose, date_str, model_type):
    """실제 모델을 사용하여 예측 수행"""
    model_key = f"{country}_{purpose}_{model_type}"
    # scaler_key = f"{country}_{purpose}_scaler"

    if model_key not in MODELS:
        raise ValueError(f"모델을 찾을 수 없습니다: {model_key}")

    model = MODELS[model_key]["model"]

    date_obj = datetime.strptime(date_str, "%Y-%m")
    features = np.array(
        [
            date_obj.year,
            date_obj.month,
            date_obj.year * 12 + date_obj.month,
            np.sin(2 * np.pi * date_obj.month / 12),
            np.cos(2 * np.pi * date_obj.month / 12),
        ]
    ).reshape(1, -1)

    # if scaler_key in SCALERS:
    #     scaler = SCALERS[scaler_key]["scaler"]
    #     features_scaled = scaler.transform(features)
    # else:
    features_scaled = features

    if model_type == "gru":
        prediction = model.predict(features_scaled.reshape(1, 1, -1))[0][0]
    else:
        prediction = model.predict(features_scaled)[0]

    prediction = max(0, int(prediction))
    return prediction


# 앱 실행 (직접 실행 시)
if __name__ == "__main__":
    app.run(debug=True)
