# -*- coding: utf-8 -*-
"""
유연한 국적별 목적별 입국자 예측 모델
Author: Jin
Created: 2025-01-15

기능:
- 국적 선택 가능
- 목적 선택 가능 (특정 목적 또는 전체 목적별)
- 예측 기간 자유 설정
- 코로나 기간 마스크 표시
- 전문적인 색상 구성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


class FlexiblePredictor:
    """유연한 국적별 목적별 입국자 예측 모델"""

    def __init__(self):
        self.models = {}  # 목적별 모델 저장
        self.scalers = {}  # 목적별 스케일러 저장
        self.data = None

        # 결과 저장 폴더
        self.results_dir = "results"
        self.models_dir = "models"
        os.makedirs(self.results_dir, exist_ok=True)
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
        return self.country_flags.get(nationality, "🌍")  # 기본값은 지구 이모지

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

    def train_purpose_model(self, nationality, purpose):
        """특정 국적-목적 조합의 모델 학습"""
        print(f"🚀 {nationality}-{purpose} 모델 학습 시작...")

        # 해당 조합 데이터 필터링
        combo_data = self.data[
            (self.data["국적"] == nationality) & (self.data["목적"] == purpose)
        ].copy()

        if len(combo_data) < 24:  # 최소 24개월 데이터 필요
            print(f"⚠️ {nationality}-{purpose} 데이터 부족 ({len(combo_data)}개월)")
            return False

        # 날짜 순 정렬
        combo_data = combo_data.sort_values("날짜").reset_index(drop=True)

        # 특성 준비
        features = self.prepare_features(combo_data)

        # 시퀀스 생성
        X, y, scaler = self.create_sequences(features)

        if len(X) < 10:  # 최소 시퀀스 개수
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
            verbose=2,
        )

        # 모델과 스케일러 저장
        key = f"{nationality}_{purpose}"
        self.models[key] = model
        self.scalers[key] = scaler

        print(f"✅ {nationality}-{purpose} 모델 학습 완료")
        return True

    def predict_future_months(self, nationality, purpose, target_months):
        """미래 월들 예측"""
        key = f"{nationality}_{purpose}"

        if key not in self.models:
            success = self.train_purpose_model(nationality, purpose)
            if not success:
                return None

        model = self.models[key]
        scaler = self.scalers[key]

        # 해당 조합의 최근 데이터
        combo_data = (
            self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
            .copy()
            .sort_values("날짜")
        )

        # 특성 준비
        features = self.prepare_features(combo_data)

        # 최근 12개월 시퀀스
        recent_data = features.tail(12).copy()
        current_sequence = scaler.transform(recent_data)

        predictions = []
        sequence = current_sequence.copy()

        # 마지막 실제 날짜
        last_date = combo_data["날짜"].iloc[-1]

        for target_month in target_months:
            target_date = pd.to_datetime(target_month + "-01")

            # 실제 데이터가 있는지 확인
            existing_data = combo_data[combo_data["날짜"] == target_date]

            if len(existing_data) > 0:
                # 실제 데이터 사용
                actual_value = existing_data["입국자수"].iloc[0]
                predictions.append({"month": target_month, "value": actual_value, "type": "actual"})
            else:
                # 예측값 계산
                # 현재 시퀀스로 다음 달 예측
                pred_scaled = model.predict(sequence.reshape(1, 12, -1), verbose=0)[0, 0]

                # 역스케일링
                dummy_data = np.zeros((1, features.shape[1]))
                dummy_data[0, 0] = pred_scaled
                pred_value = scaler.inverse_transform(dummy_data)[0, 0]
                pred_value = max(0, int(pred_value))  # 음수 방지, 정수화

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

        # 연결을 위한 이전 달 추가 (2025-06)
        prev_year, prev_month = start_year, start_month - 1
        if prev_month < 1:
            prev_month = 12
            prev_year -= 1

        connection_month = f"{prev_year}-{prev_month:02d}"
        all_months = [connection_month] + target_months

        # 목적 결정
        if purpose is None:
            # 전체 목적별 예측
            purposes = self.get_available_combinations(nationality)
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

    def plot_single_purpose(self, nationality, purpose, predictions, start_date, end_date):
        """단일 목적 시각화 (이중 그래프)"""
        print("📊 이중 그래프 시각화 중...")

        # 전체 데이터에서 해당 조합 가져오기
        combo_data = (
            self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
            .copy()
            .sort_values("날짜")
        )

        # 코로나 기간을 포함한 전체 데이터
        display_data = combo_data.tail(60)  # 5년치

        # 최근 데이터 (예측 구간용)
        recent_data = combo_data.tail(18)  # 최근 18개월

        print(
            f"📅 전체 기간: {display_data['날짜'].min().strftime('%Y-%m')} ~ {display_data['날짜'].max().strftime('%Y-%m')}"
        )
        print(
            f"📅 예측 확대 기간: {recent_data['날짜'].min().strftime('%Y-%m')} ~ {recent_data['날짜'].max().strftime('%Y-%m')}"
        )

        # 🔧 이중 subplot 생성
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2, 3]}
        )

        # ================================
        # 상단: 전체 추이 그래프 (코로나 포함)
        # ================================

        # 코로나 기간 마스크
        covid_data = combo_data[combo_data["코로나기간"] == 1]
        if len(covid_data) > 0:
            covid_start_date = covid_data["날짜"].min()
            covid_end_date = covid_data["날짜"].max()

            ax1.axvspan(
                covid_start_date,
                covid_end_date,
                alpha=0.3,
                color="red",
                label="🦠 코로나 기간",
                zorder=1,
            )

        # 전체 실제 데이터
        ax1.plot(
            display_data["날짜"],
            display_data["입국자수"],
            "b-",
            linewidth=2,
            label="📊 실제값",
            alpha=0.8,
            zorder=3,
        )

        # 예측 기간 하이라이트
        pred_start = pd.to_datetime(start_date + "-01")
        pred_end = pd.to_datetime(end_date + "-01")
        ax1.axvspan(
            pred_start,
            pred_end,
            alpha=0.2,
            color="orange",
            label="🔍 예측 구간 (하단 확대)",
            zorder=1,
        )

        # 제목 설정 (동적 국기)
        ax1.set_title(
            f"{self.get_country_flag(nationality)} {nationality} - {purpose} 전체 추이 (코로나 기간 포함)",
            fontsize=16,
            fontweight="bold",
        )
        ax1.set_ylabel("👥 입국자수 (명)", fontsize=12)
        ax1.legend(fontsize=10, loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45, labelsize=10)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        # ================================
        # 하단: 예측 구간 확대 그래프
        # ================================

        # 최근 실제 데이터
        ax2.plot(
            recent_data["날짜"],
            recent_data["입국자수"],
            "b-",
            linewidth=3,
            label="📊 실제값",
            alpha=0.9,
            marker="o",
            markersize=4,
            zorder=3,
        )

        # 예측 데이터 분리
        pred_dates = []
        pred_values = []
        pred_labels = []
        actual_dates = []
        actual_values = []
        connection_dates = []
        connection_values = []

        for pred in predictions:
            pred_date = pd.to_datetime(pred["month"] + "-01")

            if pred["type"] == "actual":
                actual_dates.append(pred_date)
                actual_values.append(pred["value"])
            elif pred["month"] < start_date:  # 연결용 (2025-06)
                connection_dates.append(pred_date)
                connection_values.append(pred["value"])
            else:  # 예측 기간
                pred_dates.append(pred_date)
                pred_values.append(pred["value"])
                pred_labels.append(pred["month"])

        # 🔧 연결용 예측값 (회색 점선)
        if connection_dates:
            ax2.plot(
                connection_dates,
                connection_values,
                color="#404040",
                linestyle="--",
                linewidth=3,
                label="⚫ 예측값(참고용)",
                alpha=0.8,
                marker="o",
                markersize=8,
                zorder=4,
            )

        # 🔧 예측 기간 예측값 (빨간색 실선)
        if pred_dates:
            ax2.plot(
                pred_dates,
                pred_values,
                color="red",
                linestyle="-",
                linewidth=3,
                label="🔴 예측값(목표기간)",
                alpha=0.9,
                marker="s",
                markersize=10,
                zorder=4,
            )

            # 🔧 예측값 숫자 표시 (겹치지 않게 조정)
            for i, (date, value, month_label) in enumerate(
                zip(pred_dates, pred_values, pred_labels)
            ):
                # 위치를 번갈아가며 조정
                if i % 2 == 0:
                    xytext = (0, 30)  # 위쪽
                    va = "bottom"
                else:
                    xytext = (0, -40)  # 아래쪽
                    va = "top"

                ax2.annotate(
                    f"{month_label}\n{value:,}명",
                    xy=(date, value),
                    xytext=xytext,
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=11,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="yellow",
                        alpha=0.9,
                        edgecolor="red",
                        linewidth=2,
                    ),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0", color="red", alpha=0.7
                    ),
                    zorder=6,
                )

        # 실제값 (예측 기간 내)
        if actual_dates:
            ax2.plot(
                actual_dates,
                actual_values,
                "go",
                markersize=12,
                label="✅ 실제값(예측기간)",
                alpha=0.9,
                zorder=4,
            )

        # 🔧 실제-예측 연결선
        if len(recent_data) > 0 and (connection_dates or pred_dates):
            last_actual_date = recent_data["날짜"].iloc[-1]
            last_actual_value = recent_data["입국자수"].iloc[-1]

            if connection_dates:
                first_pred_date = connection_dates[0]
                first_pred_value = connection_values[0]
            elif pred_dates:
                first_pred_date = pred_dates[0]
                first_pred_value = pred_values[0]
            else:
                first_pred_date = None

            if first_pred_date:
                ax2.plot(
                    [last_actual_date, first_pred_date],
                    [last_actual_value, first_pred_value],
                    "k:",
                    alpha=0.6,
                    linewidth=2,
                    label="🔗 실제-예측 연결",
                    zorder=2,
                )

        # 예측 기간 하이라이트 (하단)
        ax2.axvspan(
            pred_start, pred_end, alpha=0.15, color="blue", label="📅 예측 목표 기간", zorder=1
        )

        # 🔧 y축 범위 조정 (예측값이 잘 보이도록)
        all_values = list(recent_data["입국자수"])
        if connection_values:
            all_values.extend(connection_values)
        if pred_values:
            all_values.extend(pred_values)

        y_min = min(all_values) * 0.9
        y_max = max(all_values) * 1.15  # 여유 공간
        ax2.set_ylim(y_min, y_max)

        ax2.set_title(
            f"📊 {nationality} - {purpose} 예측 구간 상세 ({start_date} ~ {end_date})",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("📅 날짜", fontsize=12)
        ax2.set_ylabel("👥 입국자수 (명)", fontsize=12)
        ax2.legend(fontsize=11, loc="upper left")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45, labelsize=11)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        # 전체 레이아웃 조정
        plt.tight_layout()

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.results_dir}/prediction_{nationality}_{purpose}_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"📊 이중 그래프 저장: {plot_path}")

    def plot_multiple_purposes(self, nationality, results, start_date, end_date):
        """다중 목적 시각화 (첫 번째 그래프 높이 확장)"""
        print("📊 다중 목적 이중 Y축 그래프 시각화 중...")

        num_purposes = len(results)

        # 🔧 전체 높이 더욱 확장 + 첫 번째 그래프 비율 증가
        fig = plt.figure(figsize=(18, 12 + 6 * num_purposes))  # 전체 높이 확장

        # 그리드 설정: 첫 번째 그래프 비율 대폭 증가
        gs = fig.add_gridspec(
            num_purposes + 1,
            1,
            height_ratios=[4] + [4] * num_purposes,  # 첫 번째: 2→4로 증가
            hspace=0.4,
        )

        # ================================
        # 상단: 전체 추이 통합 그래프 (높이 확장)
        # ================================
        ax_overview = fig.add_subplot(gs[0, 0])

        # 목적별 데이터 규모 분석
        purpose_scales = {}
        for purpose, predictions in results.items():
            combo_data = (
                self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
                .copy()
                .sort_values("날짜")
            )

            display_data = combo_data.tail(60)
            avg_value = display_data["입국자수"].mean()
            purpose_scales[purpose] = avg_value
            print(f"📊 {purpose}: 평균 {avg_value:,.0f}명")

        # 가장 큰 값을 가진 목적 찾기 (보통 관광)
        max_purpose = max(purpose_scales, key=purpose_scales.get)
        max_value = purpose_scales[max_purpose]

        # 이중 Y축 기준: 최대값의 1/10 이하면 우측 축 사용
        threshold = max_value / 10
        left_purposes = []  # 좌측 Y축 (큰 값들)
        right_purposes = []  # 우측 Y축 (작은 값들)

        for purpose, avg_val in purpose_scales.items():
            if avg_val >= threshold:
                left_purposes.append(purpose)
            else:
                right_purposes.append(purpose)

        print(f"🔧 좌측 Y축 (큰 값): {left_purposes}")
        print(f"🔧 우측 Y축 (작은 값): {right_purposes}")

        # 우측 Y축 생성
        ax_right = ax_overview.twinx() if right_purposes else None

        colors = ["blue", "green", "orange", "purple", "brown", "pink"]
        color_idx = 0

        # 🔧 좌측 Y축 목적들 그리기 (주요 목적 강조)
        for purpose in left_purposes:
            color = colors[color_idx % len(colors)]
            color_idx += 1

            combo_data = (
                self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
                .copy()
                .sort_values("날짜")
            )

            display_data = combo_data.tail(60)

            # 🌟 가장 큰 목적 vs 기타 목적 구분
            if purpose == max_purpose:
                # 주요 목적: 진한 색상, 굵은 선, 완전 불투명
                ax_overview.plot(
                    display_data["날짜"],
                    display_data["입국자수"],
                    color=color,
                    linewidth=6,  # 더 굵은 선
                    label=f"⭐ {purpose} (주요)",
                    alpha=1.0,  # 완전 불투명
                    zorder=5,  # 최상위 레이어
                    marker="o",
                    markersize=6,
                )
            else:
                # 기타 목적: 옅은 색상, 얇은 선, 반투명
                ax_overview.plot(
                    display_data["날짜"],
                    display_data["입국자수"],
                    color=color,
                    linewidth=2,  # 얇은 선
                    label=f"{purpose} (좌축)",
                    alpha=0.4,  # 반투명
                    zorder=2,  # 배경 레이어
                    marker="o",
                    markersize=3,
                    linestyle="-",
                )

        # 🔧 우측 Y축 목적들 그리기 (기타 목적들은 더 옅게)
        if ax_right and right_purposes:
            for purpose in right_purposes:
                color = colors[color_idx % len(colors)]
                color_idx += 1

                combo_data = (
                    self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
                    .copy()
                    .sort_values("날짜")
                )

                display_data = combo_data.tail(60)

                # 우축은 일반적으로 작은 값들이므로 더 옅게 표시
                ax_right.plot(
                    display_data["날짜"],
                    display_data["입국자수"],
                    color=color,
                    linewidth=2,  # 얇은 선
                    label=f"{purpose} (우축)",
                    alpha=0.3,  # 더 반투명
                    zorder=1,  # 가장 배경 레이어
                    linestyle="--",  # 점선으로 구분
                    marker="s",
                    markersize=3,
                )

        # 코로나 기간 마스크
        if num_purposes > 0:
            first_purpose = list(results.keys())[0]
            combo_data = self.data[
                (self.data["국적"] == nationality) & (self.data["목적"] == first_purpose)
            ].copy()

            covid_data = combo_data[combo_data["코로나기간"] == 1]
            if len(covid_data) > 0:
                covid_start_date = covid_data["날짜"].min()
                covid_end_date = covid_data["날짜"].max()

                ax_overview.axvspan(
                    covid_start_date,
                    covid_end_date,
                    alpha=0.3,
                    color="red",
                    label="🦠 코로나 기간",
                    zorder=1,
                )

                print(
                    f"🎭 코로나 마스크 적용: {covid_start_date.strftime('%Y-%m')} ~ {covid_end_date.strftime('%Y-%m')}"
                )

        # 예측 기간 하이라이트
        pred_start = pd.to_datetime(start_date + "-01")
        pred_end = pd.to_datetime(end_date + "-01")
        ax_overview.axvspan(
            pred_start,
            pred_end,
            alpha=0.2,
            color="orange",
            label="🔍 예측 구간 (하단 상세)",
            zorder=1,
        )

        # 🔧 축 설정 및 범례 (폰트 크기 더 증가)
        ax_overview.set_title(
            f"{self.get_country_flag(nationality)} {nationality} 전체 목적별 추이 (이중 Y축 - 확장)",
            fontsize=20,
            fontweight="bold",
        )  # 동적 국기로 변경
        ax_overview.set_ylabel(
            "👥 입국자수 - 좌축 (명)", fontsize=16, color="blue"
        )  # 라벨 크기 증가
        ax_overview.tick_params(axis="y", labelcolor="blue", labelsize=14)  # 틱 라벨 크기 증가

        if ax_right:
            ax_right.set_ylabel(
                "👥 입국자수 - 우축 (명)", fontsize=16, color="red"
            )  # 라벨 크기 증가
            ax_right.tick_params(axis="y", labelcolor="red", labelsize=14)  # 틱 라벨 크기 증가

        # 🔧 통합 범례 (크기 증가, 위치 조정)
        lines1, labels1 = ax_overview.get_legend_handles_labels()
        if ax_right:
            lines2, labels2 = ax_right.get_legend_handles_labels()
            ax_overview.legend(
                lines1 + lines2,
                labels1 + labels2,
                fontsize=14,
                loc="upper left",  # 범례 크기 증가
                bbox_to_anchor=(0.02, 0.98),
            )  # 위치 미세 조정
        else:
            ax_overview.legend(fontsize=14, loc="upper left", bbox_to_anchor=(0.02, 0.98))

        ax_overview.grid(True, alpha=0.3)
        ax_overview.tick_params(axis="x", rotation=45, labelsize=14)  # x축 라벨 크기 증가

        # Y축 포맷
        ax_overview.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
        if ax_right:
            ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        # ================================
        # 하단: 목적별 예측 상세 그래프들 (기존과 동일)
        # ================================

        for idx, (purpose, predictions) in enumerate(results.items()):
            ax = fig.add_subplot(gs[idx + 1, 0])

            # 해당 목적의 최근 데이터
            combo_data = (
                self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
                .copy()
                .sort_values("날짜")
            )

            recent_data = combo_data.tail(18)  # 최근 18개월

            # 실제 데이터 플롯
            ax.plot(
                recent_data["날짜"],
                recent_data["입국자수"],
                "b-",
                linewidth=4,
                label="📊 실제값",
                alpha=0.9,
                marker="o",
                markersize=6,
                zorder=3,
            )

            # 예측 데이터 분리
            pred_dates = []
            pred_values = []
            pred_labels = []
            actual_dates = []
            actual_values = []
            connection_dates = []
            connection_values = []

            for pred in predictions:
                pred_date = pd.to_datetime(pred["month"] + "-01")

                if pred["type"] == "actual":
                    actual_dates.append(pred_date)
                    actual_values.append(pred["value"])
                elif pred["month"] < start_date:  # 연결용 구간
                    connection_dates.append(pred_date)
                    connection_values.append(pred["value"])
                else:  # 예측 기간
                    pred_dates.append(pred_date)
                    pred_values.append(pred["value"])
                    pred_labels.append(pred["month"])

            # 연결용 예측값 (회색 점선)
            if connection_dates:
                ax.plot(
                    connection_dates,
                    connection_values,
                    color="#404040",
                    linestyle="--",
                    linewidth=4,
                    label="⚫ 예측값(참고용)" if idx == 0 else "",
                    alpha=0.8,
                    marker="o",
                    markersize=8,
                    zorder=4,
                )

            # 예측 기간 예측값 (빨간색 실선)
            if pred_dates:
                ax.plot(
                    pred_dates,
                    pred_values,
                    color="red",
                    linestyle="-",
                    linewidth=4,
                    label="🔴 예측값(목표기간)" if idx == 0 else "",
                    alpha=0.9,
                    marker="s",
                    markersize=10,
                    zorder=4,
                )

                # 예측값 숫자 표시
                for i, (date, value, month_label) in enumerate(
                    zip(pred_dates, pred_values, pred_labels)
                ):
                    if i % 2 == 0:
                        xytext = (0, 35)
                        va = "bottom"
                    else:
                        xytext = (0, -45)
                        va = "top"

                    ax.annotate(
                        f"{month_label}\n{value:,}명",
                        xy=(date, value),
                        xytext=xytext,
                        textcoords="offset points",
                        ha="center",
                        va=va,
                        fontsize=12,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.6",
                            facecolor="yellow",
                            alpha=0.9,
                            edgecolor="red",
                            linewidth=2,
                        ),
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="arc3,rad=0",
                            color="red",
                            alpha=0.7,
                            lw=2,
                        ),
                        zorder=6,
                    )

            # 실제값 (예측 기간 내)
            if actual_dates:
                ax.plot(
                    actual_dates,
                    actual_values,
                    "go",
                    markersize=14,
                    label="✅ 실제값(예측기간)" if idx == 0 else "",
                    alpha=0.9,
                    zorder=4,
                )

            # 실제-예측 연결선
            if len(recent_data) > 0 and (connection_dates or pred_dates):
                last_actual_date = recent_data["날짜"].iloc[-1]
                last_actual_value = recent_data["입국자수"].iloc[-1]

                if connection_dates:
                    first_pred_date = connection_dates[0]
                    first_pred_value = connection_values[0]
                elif pred_dates:
                    first_pred_date = pred_dates[0]
                    first_pred_value = pred_values[0]
                else:
                    first_pred_date = None

                if first_pred_date:
                    ax.plot(
                        [last_actual_date, first_pred_date],
                        [last_actual_value, first_pred_value],
                        "k:",
                        alpha=0.6,
                        linewidth=3,
                        label="🔗 실제-예측 연결" if idx == 0 else "",
                        zorder=2,
                    )

            # 예측 기간 하이라이트
            ax.axvspan(
                pred_start,
                pred_end,
                alpha=0.15,
                color="blue",
                label="📅 예측 목표 기간" if idx == 0 else "",
                zorder=1,
            )

            # y축 범위 조정
            all_values = list(recent_data["입국자수"])
            if connection_values:
                all_values.extend(connection_values)
            if pred_values:
                all_values.extend(pred_values)

            if all_values:
                y_min = min(all_values) * 0.85
                y_max = max(all_values) * 1.2
                ax.set_ylim(y_min, y_max)

            # 목적별 규모 표시
            avg_val = purpose_scales.get(purpose, 0)
            axis_info = "좌축" if purpose in left_purposes else "우축"
            ax.set_title(
                f"📊 {purpose} (평균: {avg_val:,.0f}명, {axis_info})",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_ylabel("👥 입국자수 (명)", fontsize=14)

            # 범례 (첫 번째에만)
            if idx == 0:
                ax.legend(fontsize=12, loc="upper left")

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45, labelsize=12)
            ax.tick_params(axis="y", labelsize=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        # 전체 제목
        fig.suptitle(
            f"{self.get_country_flag(nationality)} {nationality} 목적별 입국자 예측 (전체 확장 버전) ({start_date} ~ {end_date})",
            fontsize=22,
            fontweight="bold",
            y=0.98,
        )  # 제목 크기 더 증가

        plt.tight_layout()

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.results_dir}/prediction_{nationality}_all_purposes_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"📊 전체 확장 그래프 저장: {plot_path}")

    def save_results(self, nationality, results, start_date, end_date):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 결과 데이터프레임 생성
        result_rows = []

        for purpose, predictions in results.items():
            for pred in predictions:
                result_rows.append(
                    {
                        "국적": nationality,
                        "목적": purpose,
                        "예측월": pred["month"],
                        "입국자수": pred["value"],
                        "데이터타입": pred["type"],
                        "예측기간": f"{start_date}~{end_date}",
                        "생성시간": timestamp,
                    }
                )

        results_df = pd.DataFrame(result_rows)

        # CSV 저장
        results_path = f"{self.results_dir}/prediction_results_{nationality}_{timestamp}.csv"
        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

        print(f"💾 결과 저장 완료: {results_path}")

        # 결과 요약 출력
        print(f"\n📊 예측 결과 요약:")
        for purpose, predictions in results.items():
            pred_values = [
                p for p in predictions if p["type"] == "predicted" and p["month"] >= start_date
            ]
            if pred_values:
                total = sum(p["value"] for p in pred_values)
                print(f"   🎯 {purpose}: {total:,}명 (기간 합계)")

        return results_df


def main():
    """메인 실행 함수"""
    # GPU 설정
    gpu_available = setup_gpu()
    print(f"🍎 GPU 사용 가능: {'Yes' if gpu_available else 'No'}")

    print("🚀 유연한 입국자 예측 시스템 시작")
    print("=" * 60)

    # 예측기 초기화
    predictor = FlexiblePredictor()

    # 사용 예시
    print("📝 사용 예시:")
    print("1. 일본 관광만 예측:")
    print('   predictor.predict("일본", "관광", "2025-07", "2025-09")')
    print("\n2. 일본 전체 목적별 예측:")
    print('   predictor.predict("일본", None, "2025-07", "2025-09")')

    # 실제 예측 실행 (예시)
    try:
        print("\n🎯 일본 관광 예측 실행 중...")
        result = predictor.predict(
            nationality="미국", purpose=None, start_date="2025-10", end_date="2025-12"
        )

        if result:
            print("✅ 예측 완료!")
        else:
            print("❌ 예측 실패")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
