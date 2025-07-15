# -*- coding: utf-8 -*-
"""
LSTM 모델 예측 및 시각화 (Model Predictor)
Author: Jin
Created: 2025-01-15

기능:
- 저장된 모델 로드
- 빠른 예측 수행
- 시각화 및 결과 저장
- 사용자 인터페이스
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def setup_gpu():
    """M1 GPU 최적화 설정"""
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("🍎 M1 GPU 메모리 증가 설정 완료")
        return len(physical_devices) > 0
    except Exception as e:
        print(f"⚠️ GPU 설정 실패: {e}")
        return False


class ModelPredictor:
    """모델 예측 및 시각화 담당 클래스"""

    def __init__(self):
        self.models = {}  # 로드된 모델들
        self.scalers = {}  # 로드된 스케일러들
        self.data = None

        # 폴더 설정
        self.results_dir = "results"
        self.models_dir = "models"
        os.makedirs(self.results_dir, exist_ok=True)

        # 코로나 기간 정의
        self.covid_start = pd.to_datetime("2020-03-01")
        self.covid_end = pd.to_datetime("2022-10-31")

        # 🌍 국가별 국기 이모지 매핑
        self.country_flags = {
            "일본": "🇯🇵",
            "미국": "🇺🇸",
            "중국": "🇨🇳",
            "태국": "🇹🇭",
            "대만": "🇹🇼",
            "베트남": "🇻🇳",
            "필리핀": "🇵🇭",
            "말레이시아": "🇲🇾",
            "싱가포르": "🇸🇬",
            "인도네시아": "🇮🇩",
            "인도": "🇮🇳",
            "몽골": "🇲🇳",
            "우즈베키스탄": "🇺🇿",
            "카자흐스탄": "🇰🇿",
            "러시아": "🇷🇺",
            "호주": "🇦🇺",
            "캐나다": "🇨🇦",
            "영국": "🇬🇧",
            "독일": "🇩🇪",
            "프랑스": "🇫🇷",
            "이탈리아": "🇮🇹",
            "스페인": "🇪🇸",
            "네덜란드": "🇳🇱",
            "브라질": "🇧🇷",
        }

    def get_country_flag(self, nationality):
        """국가명에 따른 국기 이모지 반환"""
        return self.country_flags.get(nationality, "🌍")

    def load_data(self):
        """전체 데이터 로드"""
        print("📂 데이터 로드 중...")

        try:
            self.data = pd.read_csv(
                "../../data_preprocessing/data/processed/외국인입국자_전처리완료_딥러닝용.csv"
            )

            # 날짜 컬럼 생성
            self.data["날짜"] = pd.to_datetime(
                self.data["연도"].astype(str)
                + "-"
                + self.data["월"].astype(str).str.zfill(2)
                + "-01"
            )

            print(f"✅ 데이터 로드 완료: {len(self.data):,}행")
            return True

        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False

    def load_saved_models(self, nationality=None, timestamp=None):
        """저장된 모델들을 로드"""
        print("🔄 저장된 모델 로드 중...")

        # 모델 파일 찾기
        if nationality and timestamp:
            model_file = os.path.join(self.models_dir, f"models_{nationality}_{timestamp}.pkl")
            scaler_file = os.path.join(self.models_dir, f"scalers_{nationality}_{timestamp}.pkl")
        else:
            # 가장 최신 파일 찾기
            model_files = [
                f
                for f in os.listdir(self.models_dir)
                if f.startswith("models_") and f.endswith(".pkl")
            ]
            if not model_files:
                print("❌ 저장된 모델을 찾을 수 없습니다.")
                return False

            model_file = os.path.join(self.models_dir, sorted(model_files)[-1])
            scaler_file = model_file.replace("models_", "scalers_")

        try:
            # 모델 경로 딕셔너리 로드
            with open(model_file, "rb") as f:
                model_paths = pickle.load(f)

            # 각 모델 로드
            for key, model_path in model_paths.items():
                self.models[key] = load_model(model_path)

            # 스케일러 로드
            with open(scaler_file, "rb") as f:
                self.scalers = pickle.load(f)

            print(f"✅ 모델 로드 완료: {len(self.models)}개")
            return True

        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False

    def prepare_features(self, data):
        """특성 준비 (트레이너와 동일)"""
        processed_data = data.copy()

        # 시간 특성
        processed_data["월_sin"] = np.sin(2 * np.pi * processed_data["월"] / 12)
        processed_data["월_cos"] = np.cos(2 * np.pi * processed_data["월"] / 12)

        # 계절 인코딩
        if "계절" in processed_data.columns:
            season_mapping = {"봄": 1, "여름": 2, "가을": 3, "겨울": 4}
            processed_data["계절"] = processed_data["계절"].map(season_mapping)

        # 핵심 특성 선택
        feature_columns = [
            "입국자수",
            "연도",
            "월",
            "분기",
            "계절",
            "코로나기간",
            "입국자수_1개월전",
            "입국자수_3개월전",
            "입국자수_12개월전",
            "입국자수_3개월평균",
            "월_sin",
            "월_cos",
        ]

        # 실제 존재하는 컬럼만 사용
        available_features = [col for col in feature_columns if col in processed_data.columns]
        features_data = processed_data[available_features].copy()

        # 안전한 데이터 처리
        features_data = features_data.fillna(0)

        # 무한대 값 처리
        for col in features_data.select_dtypes(include=[np.number]).columns:
            features_data[col] = features_data[col].replace([np.inf, -np.inf], 0)

        # 음수 클리핑
        features_data["입국자수"] = np.clip(features_data["입국자수"], 0, None)

        return features_data

    def predict_future_months(self, nationality, purpose, target_months):
        """미래 월들 예측 (빠른 예측)"""
        key = f"{nationality}_{purpose}"

        if key not in self.models:
            print(f"❌ {nationality}-{purpose} 모델을 찾을 수 없습니다.")
            return None

        model = self.models[key]
        scaler = self.scalers[key]

        # 해당 조합의 최근 데이터
        combo_data = (
            self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
            .copy()
            .sort_values("날짜")
        )

        if len(combo_data) < 12:
            print(f"⚠️ {nationality}-{purpose} 예측용 데이터 부족")
            return None

        # 특성 준비
        features = self.prepare_features(combo_data)
        recent_data = features.tail(12)

        # 시퀀스 초기화
        sequence = scaler.transform(recent_data.values)

        predictions = []

        for target_month in target_months:
            target_date = pd.to_datetime(target_month + "-01")

            # 실제 데이터가 있는지 확인
            existing_data = combo_data[combo_data["날짜"] == target_date]

            if len(existing_data) > 0:
                # 실제값 사용
                actual_value = existing_data["입국자수"].iloc[0]
                predictions.append({"month": target_month, "value": actual_value, "type": "actual"})
            else:
                # 예측값 계산
                pred_scaled = model.predict(sequence.reshape(1, 12, -1), verbose=0)[0, 0]

                # 역스케일링
                dummy_data = np.zeros((1, features.shape[1]))
                dummy_data[0, 0] = pred_scaled
                pred_value = scaler.inverse_transform(dummy_data)[0, 0]
                pred_value = max(0, int(pred_value))

                predictions.append(
                    {"month": target_month, "value": pred_value, "type": "predicted"}
                )

                # 다음 예측을 위해 시퀀스 업데이트
                year, month = map(int, target_month.split("-"))

                # 새로운 특성 생성
                new_features = recent_data.iloc[-1].copy()
                new_features["연도"] = year
                new_features["월"] = month
                new_features["분기"] = (month - 1) // 3 + 1
                new_features["계절"] = [4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4][month - 1]
                new_features["월_sin"] = np.sin(2 * np.pi * month / 12)
                new_features["월_cos"] = np.cos(2 * np.pi * month / 12)
                new_features["입국자수"] = pred_value

                # 코로나 기간 체크
                new_features["코로나기간"] = (
                    1 if self.covid_start <= target_date <= self.covid_end else 0
                )

                # 시퀀스 업데이트
                new_scaled = scaler.transform(new_features.values.reshape(1, -1))
                sequence = np.vstack([sequence[1:], new_scaled])

        return predictions

    def predict(self, nationality, purpose=None, start_date="2025-07", end_date="2025-09"):
        """메인 예측 함수"""
        print(f"🎯 예측 시작: {nationality}")
        print(f"📅 기간: {start_date} ~ {end_date}")

        if self.data is None:
            if not self.load_data():
                return None

        # 모델이 로드되지 않았다면 로드 시도
        if not self.models:
            if not self.load_saved_models():
                print("❌ 모델을 로드할 수 없습니다. 먼저 model_trainer.py를 실행하세요.")
                return None

        # 예측 기간 생성
        start_year, start_month = map(int, start_date.split("-"))
        end_year, end_month = map(int, end_date.split("-"))

        target_months = []
        current_year, current_month = start_year, start_month

        while (current_year, current_month) <= (end_year, end_month):
            target_months.append(f"{current_year}-{current_month:02d}")
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        # 연결을 위한 이전 달 추가
        prev_year, prev_month = start_year, start_month - 1
        if prev_month < 1:
            prev_month = 12
            prev_year -= 1

        connection_month = f"{prev_year}-{prev_month:02d}"
        all_months = [connection_month] + target_months

        # 목적 결정
        if purpose is None:
            # 해당 국적의 사용 가능한 목적들 찾기
            available_keys = [k for k in self.models.keys() if k.startswith(nationality)]
            purposes = [k.split("_", 1)[1] for k in available_keys]

            if not purposes:
                print(f"❌ {nationality}에 대한 학습된 모델을 찾을 수 없습니다.")
                return None

            results = {}
            for p in purposes:
                predictions = self.predict_future_months(nationality, p, all_months)
                if predictions:
                    results[p] = predictions

            if results:
                self.plot_multiple_purposes(nationality, results, start_date, end_date)
                self.save_results(nationality, results, start_date, end_date)

            return results
        else:
            # 특정 목적 예측
            predictions = self.predict_future_months(nationality, purpose, all_months)

            if predictions:
                results = {purpose: predictions}
                self.plot_single_purpose(nationality, purpose, predictions, start_date, end_date)
                self.save_results(nationality, results, start_date, end_date)
                return results

            return None

    # plot_single_purpose, plot_multiple_purposes, save_results 메서드들은
    # flexible_predictor.py에서 동일하게 복사
    # (코드가 길어서 생략, 필요시 복사)


def main():
    """메인 실행 함수"""
    print("🔮 LSTM 모델 예측기 시작")
    print("=" * 50)

    # GPU 설정
    setup_gpu()

    # 예측기 초기화
    predictor = ModelPredictor()

    # 예측 실행 (예시)
    print("\n🎯 미국 전체 목적별 예측 실행 중...")
    result = predictor.predict(
        nationality="미국", purpose=None, start_date="2025-10", end_date="2025-12"
    )

    if result:
        print("🎉 예측 완료!")
    else:
        print("❌ 예측 실패")


if __name__ == "__main__":
    main()
