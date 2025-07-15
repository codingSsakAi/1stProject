# -*- coding: utf-8 -*-
"""
LSTM 모델 학습 및 저장 (Model Trainer)
Author: Jin
Created: 2025-01-15

기능:
- 국적별 목적별 모델 학습
- 모델과 스케일러 저장
- 학습 진행률 표시
- 배치 학습 지원
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle
from datetime import datetime
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


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


class ModelTrainer:
    """모델 학습 및 저장 담당 클래스"""

    def __init__(self):
        self.models = {}  # 목적별 모델 저장
        self.scalers = {}  # 목적별 스케일러 저장
        self.data = None

        # 저장 폴더
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

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
            print(
                f"📊 데이터 기간: {self.data['날짜'].min().strftime('%Y-%m')} ~ {self.data['날짜'].max().strftime('%Y-%m')}"
            )
            print(f"🌍 국적 수: {self.data['국적'].nunique()}개")
            print(f"🎯 목적 수: {self.data['목적'].nunique()}개")

            return True

        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False

    def get_available_combinations(self, nationality=None):
        """사용 가능한 국적-목적 조합 확인"""
        if self.data is None:
            self.load_data()

        if nationality:
            nationality_data = self.data[self.data["국적"] == nationality]
            purposes = nationality_data["목적"].unique().tolist()
            print(f"{self.get_country_flag(nationality)} {nationality} 목적별 데이터:")
            for purpose in purposes:
                count = len(nationality_data[nationality_data["목적"] == purpose])
                print(f"  📊 {purpose}: {count}개 데이터")
            return purposes
        else:
            combinations = self.data.groupby(["국적", "목적"]).size().reset_index(name="count")
            return combinations

    def prepare_features(self, data):
        """특성 준비"""
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

        # 음수 클리핑 (입국자수는 음수 불가)
        features_data["입국자수"] = np.clip(features_data["입국자수"], 0, None)

        return features_data

    def create_sequences(self, data, sequence_length=12):
        """시퀀스 데이터 생성"""
        # 스케일링
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        target_idx = data.columns.get_loc("입국자수")

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length : i])
            y.append(scaled_data[i, target_idx])

        return np.array(X), np.array(y), scaler

    def build_model(self, input_shape):
        """LSTM 모델 구축"""
        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        return model

    def train_purpose_model(self, nationality, purpose, verbose=True):
        """특정 국적-목적 조합의 모델 학습"""
        if verbose:
            print(f"🚀 {nationality}-{purpose} 모델 학습 시작...")

        # 해당 조합 데이터 필터링
        combo_data = self.data[
            (self.data["국적"] == nationality) & (self.data["목적"] == purpose)
        ].copy()

        if len(combo_data) < 24:  # 최소 24개월 데이터 필요
            if verbose:
                print(f"⚠️ {nationality}-{purpose} 데이터 부족 ({len(combo_data)}개월)")
            return False

        # 날짜 순 정렬
        combo_data = combo_data.sort_values("날짜").reset_index(drop=True)

        # 특성 준비
        features = self.prepare_features(combo_data)

        # 시퀀스 생성
        X, y, scaler = self.create_sequences(features)

        if len(X) < 10:  # 최소 시퀀스 개수
            if verbose:
                print(f"⚠️ {nationality}-{purpose} 시퀀스 부족 ({len(X)}개)")
            return False

        # 훈련/검증 분할
        split_idx = max(1, int(len(X) * 0.8))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 모델 구축 및 학습
        model = self.build_model(X.shape[1:])

        early_stopping = EarlyStopping(
            monitor="val_loss" if len(X_val) > 0 else "loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )

        # 검증 데이터가 있을 때만 validation_data 사용
        validation_data = (X_val, y_val) if len(X_val) > 0 else None

        history = model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=50,
            batch_size=min(32, len(X_train)),
            callbacks=[early_stopping],
            verbose=0 if not verbose else 1,
        )

        # 모델과 스케일러 저장
        key = f"{nationality}_{purpose}"
        self.models[key] = model
        self.scalers[key] = scaler

        if verbose:
            print(f"✅ {nationality}-{purpose} 모델 학습 완료")
        return True

    def train_nationality_models(self, nationality):
        """특정 국적의 모든 목적별 모델 학습"""
        print(f"\n🏋️ {self.get_country_flag(nationality)} {nationality} 모델 학습 시작...")

        # 해당 국적의 모든 목적 가져오기
        purposes = self.get_available_combinations(nationality)

        success_count = 0
        total_count = len(purposes)

        # 진행률 표시
        for purpose in tqdm(purposes, desc=f"{nationality} 모델 학습"):
            if self.train_purpose_model(nationality, purpose, verbose=False):
                success_count += 1

        print(f"🎯 {nationality} 학습 완료: {success_count}/{total_count} 모델 성공")
        return success_count

    def save_models(self, nationality=None):
        """학습된 모델들을 파일로 저장"""
        if not self.models:
            print("⚠️ 저장할 모델이 없습니다.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if nationality:
            # 특정 국적의 모델만 저장
            filtered_models = {k: v for k, v in self.models.items() if k.startswith(nationality)}
            filtered_scalers = {k: v for k, v in self.scalers.items() if k.startswith(nationality)}

            model_file = os.path.join(self.models_dir, f"models_{nationality}_{timestamp}.pkl")
            scaler_file = os.path.join(self.models_dir, f"scalers_{nationality}_{timestamp}.pkl")

            # 모델 저장 (TensorFlow 모델 별도 저장)
            models_dict = {}
            for key, model in filtered_models.items():
                model_path = os.path.join(self.models_dir, f"model_{key}_{timestamp}.h5")
                model.save(model_path)
                models_dict[key] = model_path

            # 모델 경로 딕셔너리 저장
            with open(model_file, "wb") as f:
                pickle.dump(models_dict, f)

            # 스케일러 저장
            with open(scaler_file, "wb") as f:
                pickle.dump(filtered_scalers, f)

            print(f"💾 {nationality} 모델 저장 완료:")
            print(f"   📁 모델: {model_file}")
            print(f"   📁 스케일러: {scaler_file}")

        else:
            # 전체 모델 저장
            model_file = os.path.join(self.models_dir, f"models_all_{timestamp}.pkl")
            scaler_file = os.path.join(self.models_dir, f"scalers_all_{timestamp}.pkl")

            # 모델 저장 (TensorFlow 모델 별도 저장)
            models_dict = {}
            for key, model in self.models.items():
                model_path = os.path.join(self.models_dir, f"model_{key}_{timestamp}.h5")
                model.save(model_path)
                models_dict[key] = model_path

            # 모델 경로 딕셔너리 저장
            with open(model_file, "wb") as f:
                pickle.dump(models_dict, f)

            # 스케일러 저장
            with open(scaler_file, "wb") as f:
                pickle.dump(self.scalers, f)

            print(f"💾 전체 모델 저장 완료:")
            print(f"   📁 모델: {model_file}")
            print(f"   📁 스케일러: {scaler_file}")

    def get_training_summary(self):
        """학습 요약 정보 출력"""
        if not self.models:
            print("🤷 학습된 모델이 없습니다.")
            return

        print(f"\n📊 학습 요약:")
        print(f"   🎯 총 모델 수: {len(self.models)}개")

        # 국적별 요약
        nationality_counts = {}
        for key in self.models.keys():
            nationality = key.split("_")[0]
            nationality_counts[nationality] = nationality_counts.get(nationality, 0) + 1

        for nationality, count in nationality_counts.items():
            flag = self.get_country_flag(nationality)
            print(f"   {flag} {nationality}: {count}개 모델")


def main():
    """메인 실행 함수"""
    print("🏋️ LSTM 모델 학습기 시작")
    print("=" * 50)

    # GPU 설정
    setup_gpu()

    # 트레이너 초기화
    trainer = ModelTrainer()

    if not trainer.load_data():
        return

    # 학습할 국적 선택 (예시)
    nationalities = ["일본", "미국", "중국"]  # 원하는 국적들 추가

    for nationality in nationalities:
        trainer.train_nationality_models(nationality)
        trainer.save_models(nationality)

    # 학습 요약
    trainer.get_training_summary()

    print("\n🎉 모든 모델 학습 완료!")


if __name__ == "__main__":
    main()
