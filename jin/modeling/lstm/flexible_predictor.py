# -*- coding: utf-8 -*-
"""
유연한 국적별 목적별 입국자 예측 모델 (최종 최적화 버전)
유연한 국적별 목적별 입국자 예측 모델 (최종 안정화 버전)
Author: Jin
Created: 2025-01-15

주요 기능:
- 데이터 부족 자동 해결 (증강 + 합성 생성)
- cuDNN 최적화된 LSTM 모델
- 현실적 성능 평가 기준
- 통합 리포트 생성 (CSV 1개 + 그래프 1개)
- 타임스탬프 기반 결과 저장 구조
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re  # 정규표현식 처리용 추가
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # MinMaxScaler 추가
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_curve,
    auc,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from datetime import datetime
import warnings
import platform

# 특정 워닝만 선택적으로 제거 (디버깅시 중요한 워닝은 보존)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# M1 Mac 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# TensorFlow 로깅 레벨 조정 (워닝 최소화)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO와 WARNING 숨김

# GPU 최적화 (M1 감지 후 적용)
try:
    if platform.processor() == "arm" or "Apple" in str(platform.processor()):
        print("[M1/M2 Mac] Mixed precision 비활성화")
    else:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[최적화] Mixed precision 활성화")
except Exception:
    print("[경고] Mixed precision 설정 실패 - 기본 설정 사용")

# XLA 완전 비활성화
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="

# 앙상블 모듈 제거됨 - FlexiblePredictor만 사용
TOURISM_ENSEMBLE_AVAILABLE = False


def setup_gpu():
    """M1 GPU 최적화 설정"""
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("M1 GPU 메모리 증가 설정 완료")
        return len(physical_devices) > 0
    except (RuntimeError, ValueError) as e:
        print(f"GPU 설정 실패: {e}")
        return False
    except ImportError as e:
        print(f"TensorFlow GPU 모듈 로드 실패: {e}")
        return False


class SmartCountryMapper:
    """지능형 국적 매핑 클래스"""

    def __init__(self, data_nationalities=None):
        self.data_nationalities = data_nationalities or []

        # 확장된 25개 국가 한영 매핑 테이블
        self.basic_mapping = {
            # 주요 아시아 국가 (12개)
            "중국": ["china", "cn", "prc", "중국"],
            "일본": ["japan", "jp", "nippon", "일본"],
            "대만": ["taiwan", "tw", "formosa", "대만"],
            "태국": ["thailand", "th", "thai", "태국"],
            "베트남": ["vietnam", "vn", "베트남"],
            "필리핀": ["philippines", "ph", "필리핀"],
            "말레이시아": ["malaysia", "my", "말레이시아"],
            "싱가포르": ["singapore", "sg", "싱가포르"],
            "인도네시아": ["indonesia", "id", "인도네시아"],
            "인도": ["india", "in", "인도"],
            "몽골": ["mongolia", "mn", "몽골"],
            "네팔": ["nepal", "np", "네팔"],
            
            # 서구 선진국 (8개)
            "미국": ["usa", "us", "america", "united states", "미국"],
            "영국": ["uk", "gb", "britain", "england", "영국"],
            "독일": ["germany", "de", "독일"],
            "프랑스": ["france", "fr", "프랑스"],
            "이탈리아": ["italy", "it", "이탈리아"],
            "스페인": ["spain", "es", "스페인"],
            "호주": ["australia", "au", "호주"],
            "캐나다": ["canada", "ca", "캐나다"],
            
            # 기타 주요국 (5개)
            "러시아": ["russia", "ru", "러시아"],
            "브라질": ["brazil", "br", "브라질"],
            "멕시코": ["mexico", "mx", "멕시코"],
            "터키": ["turkey", "tr", "터키"],
            "이집트": ["egypt", "eg", "이집트"],
        }

    def find_nationality(self, user_input):
        """사용자 입력으로부터 국적 찾기"""
        user_input = user_input.lower().strip()

        # 직접 매칭
        for nationality, aliases in self.basic_mapping.items():
            if user_input in aliases:
                return nationality

        # 부분 매칭
        for nationality in self.data_nationalities:
            if user_input in nationality.lower():
                return nationality

        return None


class FlexiblePredictor:
    """
    유연한 LSTM 기반 입국자 수 예측 시스템
    - 코로나 데이터 처리 옵션 지원
    - M1/M2 Mac 최적화
    - 향상된 예측 정확도
    - 타임스탬프 기반 결과 저장
    """

    def __init__(self, covid_strategy="weighted", performance_mode="auto"):
        """
        초기화

        Args:
            covid_strategy (str): 코로나 데이터 처리 방법
                - "exclude": 코로나 기간 데이터 완전 제외
                - "weighted": 코로나 기간 데이터에 낮은 가중치 적용 (기본값)
                - "include": 모든 데이터 포함 (기존 방식)
            performance_mode (str): 성능 최적화 모드
                - "auto": 자동 감지
                - "m1_optimized": M1/M2 Mac 최적화
                - "standard": 표준 설정
        """
        # 하드웨어 감지
        import platform
        import tensorflow as tf

        self.covid_strategy = covid_strategy
        self.performance_mode = performance_mode

        # M1/M2 Mac 감지 및 최적화
        if performance_mode == "auto":
            if platform.processor() == "arm" or "Apple" in platform.processor():
                self.performance_mode = "m1_optimized"
                print("[M1/M2 Mac] 최적화 모드 활성화")
            else:
                self.performance_mode = "standard"

        # TensorFlow 설정 최적화
        if self.performance_mode == "m1_optimized":
            # M1/M2 Mac용 최적화 설정 (XLA 비활성화)
            tf.config.optimizer.set_jit(False)  # XLA 비활성화 - M1/M2 호환성
            print("[최적화] M1/M2 Metal 가속 활성화 (XLA 비활성화)")

        print(f"[설정] 코로나 데이터 처리 전략: {covid_strategy}")
        print(f"[설정] 성능 모드: {self.performance_mode}")

        # 기존 초기화 코드
        # 절대경로로 수정하여 안정성 확보
        import os
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.data_path = os.path.join(base_path, "jin", "data_preprocessing", "data", "processed", "외국인입국자_전처리완료_딥러닝용.csv")
        
        # 타임스탬프 기반 결과 저장 구조 초기화
        self.base_results_dir = "results"
        self.results_dir = None  # 실제 결과 디렉토리는 create_timestamped_results_dir()에서 설정
        self.timestamp = None
        
        self.models = {}
        self.scalers = {}
        self.performance_results = []
        self.country_mapping = {}
        
        # 학습 로그 저장용 리스트 추가
        self.training_logs = []

        # 타임스탬프 결과 디렉토리 생성
        self.create_timestamped_results_dir()

        # GPU 설정
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("[성공] GPU 메모리 증가 설정 완료")

        # Mixed precision 설정 (M1/M2에서 경고 방지)
        if self.performance_mode != "m1_optimized":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[성공] Mixed precision 활성화")
        else:
            print("[경고] M1/M2 Mac에서 Mixed precision 비활성화 (안정성 향상)")

        # 데이터 로드
        self.load_data()

        # 코로나 기간 정의
        self.covid_start = pd.to_datetime("2020-03-01")
        self.covid_end = pd.to_datetime("2022-10-31")

        # 기본 성능 기준
        self.base_thresholds = {
            "mae": 1000,
            "rmse": 1500,
            "r2_score": 0.2,  # 코로나 영향 고려하여 낮춤
            "mape": 50.0,  # 코로나 영향 고려하여 높임
            "accuracy": 0.75,
            "precision": 0.7,
            "recall": 0.7,
            "f1_score": 0.45,  # 코로나 영향 고려하여 낮춤
            "fbeta_score": 0.7,
            "roc_auc": 0.75,
        }

    def create_timestamped_results_dir(self):
        """타임스탬프 기반 결과 디렉토리 생성"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(self.base_results_dir, self.timestamp)
        
        # 결과 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"[디렉토리] 결과 저장 디렉토리 생성: {self.results_dir}")
        print(f"[시간] 타임스탬프: {self.timestamp}")

    def load_data(self):
        """데이터 로드 및 전처리 (코로나 데이터 처리 포함)"""
        print("데이터 로드 중...")

        # 데이터 로드
        self.data = pd.read_csv(self.data_path)

        # 날짜 컬럼 생성 (연도, 월을 이용)
        self.data["날짜"] = pd.to_datetime(
            self.data["연도"].astype(str) + "-" + self.data["월"].astype(str).str.zfill(2) + "-01"
        )

        # 계절 데이터를 숫자로 변환
        season_map = {"봄": 1, "여름": 2, "가을": 3, "겨울": 4}
        self.data["계절"] = self.data["계절"].map(season_map)
        print("계절 데이터를 숫자로 변환 완료")

        original_size = len(self.data)

        # 코로나 데이터 처리 전략 적용
        if self.covid_strategy == "exclude":
            # 코로나 기간 데이터 완전 제외
            self.data = self.data[self.data["코로나기간"] == 0].copy()
            excluded_count = original_size - len(self.data)
            print(
                f"[제외] 코로나 기간 데이터 제외: {excluded_count:,}행 제거 ({excluded_count/original_size*100:.1f}%)"
            )

        elif self.covid_strategy == "weighted":
            # 코로나 기간 데이터에 가중치 적용용 플래그 추가
            self.data["sample_weight"] = 1.0
            covid_mask = self.data["코로나기간"] == 1
            self.data.loc[covid_mask, "sample_weight"] = 0.1  # 코로나 기간 데이터 가중치 10%
            covid_count = covid_mask.sum()
            print(f"[가중치] 코로나 기간 데이터 가중치 조정: {covid_count:,}행에 10% 가중치 적용")

        elif self.covid_strategy == "include":
            # 모든 데이터 포함 (기존 방식)
            self.data["sample_weight"] = 1.0
            print("[포함] 모든 데이터 포함 (기존 방식)")

        print(f"데이터 로드 완료: {len(self.data):,}행")
        print(f"데이터 기간: {self.data['날짜'].min()} ~ {self.data['날짜'].max()}")
        print(f"국적 수: {self.data['국적'].nunique()}개")
        print(f"목적 수: {self.data['목적'].nunique()}개")

        # 국가 매핑 초기화
        self.initialize_country_mapping()

    def initialize_country_mapping(self):
        """국가 매핑 초기화"""
        try:
            unique_countries = self.data["국적"].unique()
            for i, country in enumerate(unique_countries):
                self.country_mapping[country] = i
            print(f"국가 매핑 초기화 완료: {len(unique_countries)}개 국가")
        except KeyError as e:
            print(f"데이터에 '국적' 컬럼이 없습니다: {e}")
            self.country_mapping = {}
        except (AttributeError, TypeError) as e:
            print(f"데이터 형식 오류: {e}")
            self.country_mapping = {}

    def augment_time_series_data(self, data, target_months=120):
        """변동성 보존형 시계열 데이터 증강"""
        print(f"변동성 보존형 데이터 증강 시작: {len(data)}개월 -> 목표 {target_months}개월")

        if len(data) >= target_months:
            print("충분한 데이터로 증강 생략")
            return [data]

        # 원본 데이터의 통계 분석
        original_std = data["입국자수"].std()
        original_mean = data["입국자수"].mean()
        seasonal_pattern = self.extract_seasonal_pattern(data)

        print(f"원본 변동성: std={original_std:.0f}, cv={original_std/original_mean:.2f}")

        augmented_datasets = [data]  # 원본 포함

        # 1. 계절성 강화 노이즈 증강 (변동성 보존)
        for noise_level in [0.15, 0.25, 0.35]:  # 더 큰 노이즈로 변동성 보존
            noise_data = self.add_seasonal_noise_augmentation(data, noise_level, seasonal_pattern)
            augmented_datasets.append(noise_data)

        # 2. 트렌드 보존 증강
        for trend_factor in [0.02, 0.05, -0.02]:  # 월별 성장/감소율
            trend_data = self.add_trend_augmentation(data, trend_factor, seasonal_pattern)
            augmented_datasets.append(trend_data)

        # 3. 계절성 강화 증강
        for seasonal_boost in [1.3, 1.5, 0.7]:  # 계절성 강화/완화
            seasonal_data = self.enhance_seasonality_augmentation(
                data, seasonal_boost, seasonal_pattern
            )
            augmented_datasets.append(seasonal_data)

        # 4. 패턴 기반 합성 데이터 (부족한 경우만)
        current_total = sum(len(d) for d in augmented_datasets)
        if current_total < target_months:
            shortage = target_months - current_total
            synthetic_data = self.generate_realistic_pattern_data(data, shortage, seasonal_pattern)
            augmented_datasets.append(synthetic_data)

        final_total = sum(len(d) for d in augmented_datasets)
        print(f"변동성 보존 증강 완료: {len(data)}개월 -> {final_total}개월")

        # 변동성 유지 확인
        final_combined = pd.concat(augmented_datasets, ignore_index=True)
        final_std = final_combined["입국자수"].std()
        print(
            f"증강 후 변동성: std={final_std:.0f}, cv={final_std/final_combined['입국자수'].mean():.2f}"
        )

        return augmented_datasets

    def add_seasonal_noise_augmentation(self, data, noise_level, seasonal_pattern):
        """계절성 기반 노이즈 증강"""
        augmented = data.copy()
        original_values = augmented["입국자수"].values

        # 계절별로 다른 노이즈 레벨 적용
        seasonal_noise = np.zeros_like(original_values)
        for i, month in enumerate(augmented["월"]):
            base_noise = np.random.normal(0, noise_level, 1)[0]
            # 성수기(여름/겨울)에는 더 큰 변동성
            if month in [7, 8, 12, 1]:
                seasonal_noise[i] = base_noise * 1.5
        else:
            seasonal_noise[i] = base_noise

        noisy_values = original_values * (1 + seasonal_noise)
        augmented["입국자수"] = np.maximum(noisy_values, 0)
        return augmented

    def add_trend_augmentation(self, data, trend_factor, seasonal_pattern):
        """트렌드 보존 증강"""
        augmented = data.copy()
        original_values = augmented["입국자수"].values

        # 시간에 따른 트렌드 적용
        trend_multiplier = np.zeros_like(original_values)
        for i in range(len(original_values)):
            # 기본 트렌드 + 계절성 조정
            month = augmented.iloc[i]["월"]
            seasonal_boost = (
                seasonal_pattern.get(month, augmented["입국자수"].mean())
                / augmented["입국자수"].mean()
            )
            trend_multiplier[i] = 1 + (trend_factor * i / len(original_values)) * seasonal_boost

        trended_values = original_values * trend_multiplier
        augmented["입국자수"] = np.maximum(trended_values, 0)
        return augmented

    def enhance_seasonality_augmentation(self, data, seasonal_boost, seasonal_pattern):
        """계절성 강화 증강"""
        augmented = data.copy()
        enhanced_values = []

        for i, row in augmented.iterrows():
            month = row["월"]
            original_value = row["입국자수"]

            # 해당 월의 계절적 특성 강화
            seasonal_avg = seasonal_pattern.get(month, original_value)
            overall_avg = augmented["입국자수"].mean()

            if seasonal_avg > overall_avg:  # 성수기
                enhanced_value = original_value * seasonal_boost
            else:  # 비수기
                enhanced_value = original_value / seasonal_boost

            enhanced_values.append(max(enhanced_value, 0))

        augmented["입국자수"] = enhanced_values
        return augmented

    def generate_realistic_pattern_data(self, data, target_months, seasonal_pattern):
        """현실적인 패턴 기반 합성 데이터"""
        # 최근 2년 패턴 기반으로 생성
        recent_data = data.tail(24) if len(data) >= 24 else data

        # 계절별 변동 패턴 추출
        monthly_variations = {}
        for month in range(1, 13):
            month_data = recent_data[recent_data["월"] == month]
            if len(month_data) > 0:
                monthly_variations[month] = (
                    month_data["입국자수"].std() / month_data["입국자수"].mean()
                )
            else:
                monthly_variations[month] = 0.2  # 기본 변동성

        synthetic_rows = []
        last_date = data["날짜"].max()

        for i in range(target_months):
            new_date = last_date + pd.DateOffset(months=i + 1)
            month = new_date.month
            year = new_date.year

            # 기본값은 계절 패턴 기반
            base_value = seasonal_pattern.get(month, data["입국자수"].mean())

            # 월별 변동성 적용
            variation = np.random.normal(0, monthly_variations[month])
            final_value = base_value * (1 + variation)

            synthetic_row = {
                "날짜": new_date,
                "연도": year,
                "월": month,
                "분기": ((month - 1) // 3) + 1,
                "계절": self.get_season_number(month),
                "코로나기간": 0,  # 미래는 코로나 이후
                "입국자수": max(final_value, 0),
            }
            synthetic_rows.append(synthetic_row)

        return pd.DataFrame(synthetic_rows)

    def add_noise_augmentation(self, data, noise_level=0.15):
        """변동성 보존 노이즈 증강"""
        augmented = data.copy()
        original_values = augmented["입국자수"].values

        # 데이터 크기에 따른 적응적 노이즈
        adaptive_noise = noise_level * (1 + np.log(len(data)) / 10)
        noise = np.random.normal(0, adaptive_noise, len(original_values))

        noisy_values = original_values * (1 + noise)
        augmented["입국자수"] = np.maximum(noisy_values, 0)
        return augmented

    def time_shift_augmentation(self, data, shift_months=1):
        """시간 이동 증강"""
        augmented = data.copy()
        augmented["날짜"] = augmented["날짜"] + pd.DateOffset(months=shift_months)
        augmented["연도"] = augmented["날짜"].dt.year
        augmented["월"] = augmented["날짜"].dt.month
        augmented["분기"] = ((augmented["월"] - 1) // 3) + 1

        # 계절 재계산
        season_map = {12: 4, 1: 4, 2: 4, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        augmented["계절"] = augmented["월"].map(season_map)

        return augmented

    def scale_augmentation(self, data, scale_factor=1.1):
        """스케일 변형 증강"""
        augmented = data.copy()
        augmented["입국자수"] = (augmented["입국자수"] * scale_factor).round().astype(int)
        return augmented

    def _create_cyclical_features(self, processed_data):
        """주기적 특성 생성 (계절성)"""
        # 순환 인코딩
        processed_data["월_sin"] = np.sin(2 * np.pi * processed_data["월"] / 12)
        processed_data["월_cos"] = np.cos(2 * np.pi * processed_data["월"] / 12)
        
        # 원핫 인코딩
        for quarter in [1, 2, 3, 4]:
            processed_data[f"분기_{quarter}"] = (processed_data["분기"] == quarter).astype(int)
        
        for season in [1, 2, 3, 4]:
            processed_data[f"계절_{season}"] = (processed_data["계절"] == season).astype(int)
        
        return processed_data

    def _create_lag_features(self, processed_data, lag_periods):
        """지연 특성 생성"""
        target_col = "입국자수"
        for lag in lag_periods:
            processed_data[f"lag_{lag}"] = processed_data[target_col].shift(lag)
        return processed_data

    def _create_moving_average_features(self, processed_data, windows):
        """이동평균 특성 생성"""
        target_col = "입국자수"
        for window in windows:
            ma_col = f"ma_{window}"
            processed_data[ma_col] = processed_data[target_col].rolling(window, min_periods=1).mean()
            processed_data[f"ma_ratio_{window}"] = processed_data[target_col] / processed_data[ma_col]
        return processed_data

    def _create_volatility_features(self, processed_data, windows):
        """변동성 특성 생성"""
        target_col = "입국자수"
        for window in windows:
            volatility_col = f"volatility_{window}"
            cv_col = f"cv_{window}"
            ma_col = f"ma_{window}"
            
            processed_data[volatility_col] = processed_data[target_col].rolling(window, min_periods=1).std()
            if ma_col in processed_data.columns:
                processed_data[cv_col] = processed_data[volatility_col] / processed_data[ma_col]
        return processed_data

    def _create_momentum_features(self, processed_data, periods):
        """모멘텀 및 변화율 특성 생성"""
        target_col = "입국자수"
        for period in periods:
            processed_data[f"momentum_{period}"] = processed_data[target_col].pct_change(period)
            processed_data[f"diff_{period}"] = processed_data[target_col].diff(period)
        return processed_data

    def create_advanced_features(self, data):
        """변동성 보존형 고급 특성 엔지니어링 (리팩토링 버전)"""
        processed_data = data.copy()

        # 1. 계절성 특성 강화
        processed_data = self._create_cyclical_features(processed_data)

        # 2. 변동성 보존 지연 특성
        processed_data = self._create_lag_features(processed_data, [1, 3, 6, 12])

        # 3. 동적 이동평균 (변동성 민감)
        processed_data = self._create_moving_average_features(processed_data, [3, 6, 12])

        # 4. 변동성 지표 강화
        processed_data = self._create_volatility_features(processed_data, [3, 6])

        # 5. 모멘텀 및 변화율 지표
        processed_data = self._create_momentum_features(processed_data, [1, 3, 6])

        # 6. 계절성 상호작용 특성
        processed_data["월_x_입국자수"] = processed_data["월"] * processed_data["입국자수"]
        processed_data["계절_x_입국자수"] = processed_data["계절"] * processed_data["입국자수"]

        # 핵심 특성 선택 (변동성 보존 중심)
        core_features = [
            "입국자수",
            "연도",
            "월",
            "분기",
            "계절",
            "코로나기간",
            "월_sin",
            "월_cos",
            "분기_1",
            "분기_2",
            "분기_3",
            "분기_4",
            "계절_1",
            "계절_2",
            "계절_3",
            "계절_4",
            "lag_1",
            "lag_3",
            "lag_6",
            "ma_3",
            "ma_6",
            "ma_12",
            "ma_ratio_3",
            "ma_ratio_6",
            "volatility_3",
            "volatility_6",
            "cv_3",
            "cv_6",
            "momentum_1",
            "momentum_3",
            "diff_1",
            "diff_3",
            "월_x_입국자수",
            "계절_x_입국자수",
        ]

        # 실제 존재하는 컬럼만 사용
        available_features = [col for col in core_features if col in processed_data.columns]
        features_data = processed_data[available_features].copy()

        # 결측값 처리 (변동성 보존)
        features_data = features_data.ffill().fillna(0)

        # 무한대 처리
        for col in features_data.select_dtypes(include=[np.number]).columns:
            features_data[col] = features_data[col].replace([np.inf, -np.inf], 0)

        # 입국자수는 음수 방지
        features_data["입국자수"] = np.clip(features_data["입국자수"], 0, None)

        return features_data

    def create_sequences(self, data, sequence_length):
        """MinMaxScaler 사용으로 변동성 보존"""
        # MinMaxScaler로 변경 (극값 보존)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        target_idx = data.columns.get_loc("입국자수")

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length : i])
            y.append(scaled_data[i, target_idx])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler

    def build_adaptive_model(self, input_shape, data_size):
        """적응형 모델 구축 (cuDNN 최적화)"""

        if data_size < 100:
            # 초소규모: Dense 네트워크
            model = Sequential(
                [
                    Dense(32, activation="relu", input_shape=(input_shape[1],)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(16, activation="relu"),
                    Dense(1, activation="linear", dtype="float32"),
                ]
            )
            print(f"초소규모 모델 구축: Dense 네트워크 (데이터: {data_size}개)")

        elif data_size < 200:
            # 소규모: 단일 LSTM (cuDNN 최적화)
            model = Sequential(
                [
                    LSTM(
                        32,
                        input_shape=input_shape,
                        activation="tanh",
                        recurrent_activation="sigmoid",
                        recurrent_dropout=0.0,
                        dropout=0.2,
                    ),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(16, activation="relu"),
                    Dense(1, activation="linear", dtype="float32"),
                ]
            )
            print(f"소규모 모델 구축: 단일 LSTM (데이터: {data_size}개)")

        else:
            # 대규모: 다층 LSTM (cuDNN 최적화)
            model = Sequential(
                [
                    LSTM(
                        64,
                        return_sequences=True,
                        input_shape=input_shape,
                        activation="tanh",
                        recurrent_activation="sigmoid",
                        recurrent_dropout=0.0,
                        dropout=0.2,
                    ),
                    BatchNormalization(),
                    Dropout(0.3),
                    LSTM(
                        32,
                        activation="tanh",
                        recurrent_activation="sigmoid",
                        recurrent_dropout=0.0,
                        dropout=0.2,
                    ),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(24, activation="relu"),
                    Dense(1, activation="linear", dtype="float32"),
                ]
            )
            print(f"대규모 모델 구축: 다층 LSTM (데이터: {data_size}개)")

        # 모델 컴파일 (적응형 학습률)
        if data_size < 50:
            learning_rate = 0.01  # 소규모: 높은 학습률
        elif data_size < 200:
            learning_rate = 0.005  # 중간: 중간 학습률
        else:
            learning_rate = 0.001  # 대규모: 낮은 학습률

        # 🔥 M1/M2 Mac 최적화된 optimizer 사용
        if self.performance_mode == "m1_optimized":
            # M1/M2 Mac에서 빠른 legacy Adam optimizer 사용
            from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam

            optimizer = LegacyAdam(learning_rate=learning_rate)
            print("⚡ M1/M2 최적화: Legacy Adam optimizer 사용")
        else:
            # 표준 Adam optimizer 사용
            optimizer = Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def get_improved_thresholds(self, data_size):
        """개선된 현실적 기준"""

        if data_size < 100:
            # 초소규모: 매우 관대한 기준
            return {
                "mae": self.base_thresholds["mae"] * 3,
                "rmse": self.base_thresholds["rmse"] * 3,
                "r2_score": -0.5,  # 음수도 허용
                "mape": 150.0,
                "accuracy": 0.4,
                "precision": 0.3,
                "recall": 0.3,
                "f1_score": 0.25,
                "fbeta_score": 0.25,
                "roc_auc": 0.4,
            }
        elif data_size < 200:
            # 소규모: 관대한 기준
            return {
                "mae": self.base_thresholds["mae"] * 2,
                "rmse": self.base_thresholds["rmse"] * 2,
                "r2_score": 0.0,
                "mape": 80.0,
                "accuracy": 0.5,
                "precision": 0.4,
                "recall": 0.4,
                "f1_score": 0.35,
                "fbeta_score": 0.35,
                "roc_auc": 0.5,
            }
        else:
            # 대규모: 일반적인 기준
            return {
                "mae": self.base_thresholds["mae"],
                "rmse": self.base_thresholds["rmse"],
                "r2_score": 0.2,
                "mape": 50.0,
                "accuracy": 0.6,
                "precision": 0.5,
                "recall": 0.5,
                "f1_score": 0.45,
                "fbeta_score": 0.45,
                "roc_auc": 0.6,
            }

    def safe_inverse_transform(self, y_true_scaled, y_pred_scaled, scaler):
        """안전한 역스케일링 (MinMaxScaler 호환)"""
        try:
            # MinMaxScaler와 StandardScaler 모두 호환
            if hasattr(scaler, "scale_"):
                n_features = len(scaler.scale_)
            elif hasattr(scaler, "data_max_"):
                n_features = len(scaler.data_max_)
            else:
                n_features = 1

            # 실제값 역스케일링
            dummy_true = np.zeros((len(y_true_scaled), n_features))
            dummy_true[:, 0] = y_true_scaled
            y_true_rescaled = scaler.inverse_transform(dummy_true)[:, 0]

            # 예측값 역스케일링
            dummy_pred = np.zeros((len(y_pred_scaled), n_features))
            dummy_pred[:, 0] = y_pred_scaled
            y_pred_rescaled = scaler.inverse_transform(dummy_pred)[:, 0]

            # 음수 방지
            y_true_rescaled = np.maximum(y_true_rescaled, 0)
            y_pred_rescaled = np.maximum(y_pred_rescaled, 0)

            return y_true_rescaled, y_pred_rescaled

        except ValueError as e:
            print(f"스케일러 데이터 형식 오류: {e}")
            return np.abs(y_true_scaled), np.abs(y_pred_scaled)
        except AttributeError as e:
            print(f"스케일러 속성 오류: {e}")
            return np.abs(y_true_scaled), np.abs(y_pred_scaled)
        except (IndexError, TypeError) as e:
            print(f"배열 처리 오류: {e}")
            return np.abs(y_true_scaled), np.abs(y_pred_scaled)

    def calculate_comprehensive_metrics(self, y_true, y_pred, purpose_name, thresholds):
        """포괄적인 성능 메트릭 계산"""

        # 회귀 메트릭
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

        # 분류 메트릭을 위한 임계값 설정
        threshold = np.mean(y_true)
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)

        # 분류 메트릭
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        fbeta = fbeta_score(y_true_binary, y_pred_binary, beta=1.5, zero_division=0)

        # ROC AUC 계산
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.0

        # 메트릭 딕셔너리 생성
        metrics = {
            "purpose": purpose_name,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "mape": mape,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "fbeta_score": fbeta,
            "roc_auc": roc_auc,
            "threshold": threshold,
            "total_samples": len(y_true),
            "avg_actual": np.mean(y_true),
            "avg_predicted": np.mean(y_pred),
        }

        # 기준값 및 등급 정보 추가
        for metric in thresholds.keys():
            if metric in metrics:
                metrics[f"{metric}_기준값"] = thresholds[metric]
                metrics[f"{metric}_등급"] = self.evaluate_metric_performance(
                    metric, metrics[metric], thresholds
                )

        return metrics

    def evaluate_metric_performance(self, metric_name, value, thresholds):
        """성능 지표 평가 및 등급 부여"""
        threshold = thresholds[metric_name]

        # 낮을수록 좋은 지표들
        if metric_name in ["mae", "rmse", "mape"]:
            if value <= threshold * 0.7:
                return "우수"
            elif value <= threshold:
                return "양호"
            elif value <= threshold * 1.3:
                return "개선필요"
            else:
                return "심각"
        # 높을수록 좋은 지표들
        else:
            if value >= threshold * 1.3:
                return "우수"
            elif value >= threshold:
                return "양호"
            elif value >= threshold * 0.7:
                return "개선필요"
            else:
                return "심각"

    def train_purpose_model(self, nationality, purpose):
        """모델 학습 - Fit 로그 캡처 강화"""
        print(f"모델 학습 시작: {nationality}-{purpose}")

        # 해당 조합 데이터 필터링
        combo_data = self.data[
            (self.data["국적"] == nationality) & (self.data["목적"] == purpose)
        ].copy()

        print(f"원본 데이터: {len(combo_data)}개월")

        # 데이터 부족 시 증강
        if len(combo_data) < 24:
            print("데이터 부족으로 증강 실행...")
            augmented_datasets = self.augment_time_series_data(combo_data, target_months=120)

            # 모든 증강 데이터 결합
            all_data = []
            for dataset in augmented_datasets:
                all_data.append(dataset)
            combo_data = pd.concat(all_data, ignore_index=True)
            print(f"증강 후 데이터: {len(combo_data)}개월")

        # 날짜 순 정렬
        combo_data = combo_data.sort_values("날짜").reset_index(drop=True)
        data_size = len(combo_data)

        # 특성 엔지니어링
        features = self.create_advanced_features(combo_data)

        # 시퀀스 생성
        sequence_length = 6 if data_size < 100 else 12
        X, y, scaler = self.create_sequences(features, sequence_length)

        if len(X) == 0:
            print(f"시퀀스 생성 실패: {nationality}-{purpose}")
            return False

        # 훈련/검증 분할
        split_idx = max(1, int(len(X) * 0.85))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"학습: {len(X_train)}개, 검증: {len(X_val)}개 시퀀스")

        # 모델 구축
        model = self.build_adaptive_model(X.shape[1:], data_size)

        # 컴파일
        learning_rate = 0.01 if data_size < 100 else 0.001

        # 🔥 M1/M2 Mac 최적화된 optimizer 사용
        if self.performance_mode == "m1_optimized":
            # M1/M2 Mac에서 빠른 legacy Adam optimizer 사용
            from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam

            optimizer = LegacyAdam(learning_rate=learning_rate)
            print("⚡ M1/M2 최적화: Legacy Adam optimizer 사용")
        else:
            # 표준 Adam optimizer 사용
            optimizer = Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        # 콜백
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if len(X_val) > 0 else "loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if len(X_val) > 0 else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        # 학습
        epochs = 50 if data_size < 100 else 100
        validation_data = (X_val, y_val) if len(X_val) > 0 else None

        print(f"학습 시작 (에포크: {epochs})")
        history = model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            verbose=2,
        )

        print("학습 완료!")

        # 🔥 Fit 로그 캡처 및 저장
        training_log = self.capture_training_logs(history, nationality, purpose, data_size)
        self.training_logs.append(training_log)

        # 성능 평가
        if len(X_val) > 0:
            print("성능 평가 중...")
            y_pred_val = model.predict(X_val, verbose=1).flatten()

            # 역스케일링
            y_true_rescaled, y_pred_rescaled = self.safe_inverse_transform(
                y_val, y_pred_val, scaler
            )

            print(f"예측값 범위: {y_pred_rescaled.min():,.0f} ~ {y_pred_rescaled.max():,.0f}명")

            # 현실적 기준 적용
            realistic_thresholds = self.get_improved_thresholds(data_size)

            # 메트릭 계산
            metrics = self.calculate_comprehensive_metrics(
                y_true_rescaled, y_pred_rescaled, f"{nationality}_{purpose}", realistic_thresholds
            )

            # 🔥 추가 정보 (Fit 로그 포함)
            metrics.update(
                {
                    "nationality": nationality,
                    "training_samples": len(X_train),
                    "validation_samples": len(X_val),
                    "epochs_trained": len(history.history["loss"]),
                    "final_train_loss": history.history["loss"][-1],
                    "final_val_loss": history.history.get("val_loss", [None])[-1],
                    "final_train_mae": history.history["mae"][-1],
                    "final_val_mae": history.history.get("val_mae", [None])[-1],
                    "best_train_loss": min(history.history["loss"]),
                    "best_val_loss": min(history.history.get("val_loss", [float('inf')])),
                    "best_train_mae": min(history.history["mae"]),
                    "best_val_mae": min(history.history.get("val_mae", [float('inf')])),
                    "early_stopped": len(history.history["loss"]) < epochs,
                    "learning_rate_used": learning_rate,
                    "data_size": data_size,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                }
            )

            self.performance_results.append(metrics)

            # 성능 출력
            print(f"성능 결과: MAE {metrics['mae']:,.0f}, R2 {metrics['r2_score']:.3f}")

        # 모델 저장
        key = f"{nationality}_{purpose}"
        self.models[key] = model
        self.scalers[key] = scaler

        print(f"모델 학습 완료: {nationality}-{purpose}")
        return True

    def capture_training_logs(self, history, nationality, purpose, data_size):
        """🔥 학습 과정 상세 로그 캡처"""
        training_log = {
            "nationality": nationality,
            "purpose": purpose,
            "data_size": data_size,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            
            # 학습 결과 요약
            "epochs_trained": len(history.history["loss"]),
            "final_train_loss": history.history["loss"][-1],
            "final_train_mae": history.history["mae"][-1],
            "best_train_loss": min(history.history["loss"]),
            "best_train_mae": min(history.history["mae"]),
            
            # 검증 결과 (있는 경우)
            "has_validation": "val_loss" in history.history,
            "final_val_loss": history.history.get("val_loss", [None])[-1],
            "final_val_mae": history.history.get("val_mae", [None])[-1],
            "best_val_loss": min(history.history.get("val_loss", [float('inf')])),
            "best_val_mae": min(history.history.get("val_mae", [float('inf')])),
            
            # 학습 곡선 데이터
            "loss_curve": history.history["loss"],
            "mae_curve": history.history["mae"],
            "val_loss_curve": history.history.get("val_loss", []),
            "val_mae_curve": history.history.get("val_mae", []),
            
            # 학습 품질 지표
            "loss_improvement": (history.history["loss"][0] - history.history["loss"][-1]) / history.history["loss"][0] * 100,
            "mae_improvement": (history.history["mae"][0] - history.history["mae"][-1]) / history.history["mae"][0] * 100,
            "convergence_speed": len(history.history["loss"]) / 100,  # 에포크 대비 수렴 속도
        }
        
        print(f"📊 학습 로그 캡처 완료: {nationality}-{purpose}")
        print(f"   손실 개선: {training_log['loss_improvement']:.1f}%")
        print(f"   MAE 개선: {training_log['mae_improvement']:.1f}%")
        print(f"   수렴 속도: {training_log['convergence_speed']:.2f}")
        
        return training_log

    def save_training_logs_report(self):
        """🔥 학습 로그 전용 리포트 생성"""
        if not self.training_logs:
            print("저장할 학습 로그가 없습니다.")
            return
        
        # 학습 로그 데이터프레임 생성
        logs_df = pd.DataFrame(self.training_logs)
        
        # 학습 로그 전용 리포트 생성
        logs_report_data = []
        
        for _, log in logs_df.iterrows():
            report_row = {
                "국적": log["nationality"],
                "목적": log["purpose"],
                "데이터크기": log["data_size"],
                "학습에포크": log["epochs_trained"],
                "최종학습손실": f"{log['final_train_loss']:.6f}",
                "최종학습MAE": f"{log['final_train_mae']:.6f}",
                "최고학습손실": f"{log['best_train_loss']:.6f}",
                "최고학습MAE": f"{log['best_train_mae']:.6f}",
                "검증데이터유무": "있음" if log["has_validation"] else "없음",
                "최종검증손실": f"{log['final_val_loss']:.6f}" if log["final_val_loss"] is not None else "N/A",
                "최종검증MAE": f"{log['final_val_mae']:.6f}" if log["final_val_mae"] is not None else "N/A",
                "최고검증손실": f"{log['best_val_loss']:.6f}" if log["best_val_loss"] != float('inf') else "N/A",
                "최고검증MAE": f"{log['best_val_mae']:.6f}" if log["best_val_mae"] != float('inf') else "N/A",
                "손실개선률": f"{log['loss_improvement']:.1f}%",
                "MAE개선률": f"{log['mae_improvement']:.1f}%",
                "수렴속도": f"{log['convergence_speed']:.2f}",
                "생성시간": log["timestamp"],
            }
            logs_report_data.append(report_row)
        
        # 학습 로그 리포트 저장
        logs_report_df = pd.DataFrame(logs_report_data)
        logs_report_path = f"{self.results_dir}/학습로그_리포트.csv"
        logs_report_df.to_csv(logs_report_path, index=False, encoding="utf-8-sig")
        print(f"학습 로그 리포트 저장: {logs_report_path}")
        
        return logs_report_path

    def predict_future_months(self, nationality, purpose, target_months):
        """실제 패턴 반영 예측 (연속성 보정 포함) - 관광 목적 특별 처리"""

        # 🏖️ 관광 목적 특별 처리 (성능 향상)
        if purpose == "관광":
            print(f"🏖️ {nationality} 관광 목적 - 특별 최적화 모델 적용")
            return self._predict_tourism_optimized(nationality, purpose, target_months)

        # 기존 LSTM 모델 로직
        key = f"{nationality}_{purpose}"

        if key not in self.models:
            success = self.train_purpose_model(nationality, purpose)
            if not success:
                return None

        model = self.models[key]
        scaler = self.scalers[key]

        # 해당 조합의 데이터
        combo_data = (
            self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
            .copy()
            .sort_values("날짜")
        )

        # 계절성 패턴 추출 - 개선된 방식
        seasonal_pattern = self.extract_improved_seasonal_pattern(combo_data)

        # 최근 트렌드 계산 (최근 12개월 평균 변화율)
        recent_trend = self.calculate_recent_trend(combo_data)

        # 변동성 패턴 분석 (실제 데이터의 월별 변동성)
        volatility_pattern = self.analyze_volatility_pattern(combo_data)

        # 특성 준비
        features = self.create_advanced_features(combo_data)
        sequence_length = 6 if len(combo_data) < 100 else 12
        recent_data = features.tail(sequence_length).copy()
        current_sequence = scaler.transform(recent_data)

        # 🔥 연속성 보정을 위한 실제값 마지막 포인트 추출
        last_actual_value = combo_data["입국자수"].iloc[-1]
        last_actual_date = combo_data["날짜"].iloc[-1]

        # 최근 3개월 평균값 계산 (안정적인 기준값)
        recent_3months_avg = combo_data["입국자수"].tail(3).mean()

        print(f"연속성 보정 기준: {last_actual_date.strftime('%Y-%m')} = {last_actual_value:,}명")
        print(f"최근 3개월 평균: {recent_3months_avg:,}명")

        predictions = []
        sequence = current_sequence.copy()

        # 🔥 첫 번째 예측값을 위한 연속성 계수 계산
        first_pred_month = target_months[0]
        first_pred_date = pd.to_datetime(first_pred_month + "-01")
        months_gap = (first_pred_date.year - last_actual_date.year) * 12 + (
            first_pred_date.month - last_actual_date.month
        )

        # 점진적 변화를 위한 연속성 강도 (간격이 클수록 연속성 약화)
        continuity_strength = max(0.4, 1.0 - (months_gap * 0.08))  # 더 강한 연속성
        print(f"연속성 강도: {continuity_strength:.2f} (간격: {months_gap}개월)")

        for idx, target_month in enumerate(target_months):
            target_date = pd.to_datetime(target_month + "-01")

            # 실제 데이터가 있는지 확인
            existing_data = combo_data[combo_data["날짜"] == target_date]

            if len(existing_data) > 0:
                actual_value = existing_data["입국자수"].iloc[0]
                predictions.append({"month": target_month, "value": actual_value, "type": "actual"})
            else:
                # 예측값 계산
                pred_scaled = model.predict(sequence.reshape(1, sequence_length, -1), verbose=1)[
                    0, 0
                ]

                # 역스케일링
                dummy_data = np.zeros((1, features.shape[1]))
                dummy_data[0, 0] = pred_scaled
                pred_value = scaler.inverse_transform(dummy_data)[0, 0]

                # 🔥 강화된 연속성 보정 적용
                if idx == 0:
                    # 첫 번째 예측값은 실제값과 매우 연속적으로 연결
                    continuity_factor = continuity_strength * 0.9  # 90%까지 연속성 적용
                    pred_value = (pred_value * (1 - continuity_factor)) + (
                        last_actual_value * continuity_factor
                    )
                elif idx <= 2:
                    # 처음 3개월은 강한 연속성 적용
                    continuity_factor = continuity_strength * (0.6 - idx * 0.1)  # 60%, 50%, 40%
                    prev_value = predictions[-1]["value"]
                    pred_value = (pred_value * (1 - continuity_factor)) + (
                        prev_value * continuity_factor
                    )
                else:
                    # 이후에는 점진적으로 연속성 약화
                    continuity_factor = continuity_strength * max(0.1, 0.4 - (idx - 2) * 0.05)
                    prev_value = predictions[-1]["value"]
                    pred_value = (pred_value * (1 - continuity_factor)) + (
                        prev_value * continuity_factor
                    )

                # 🔥 개선된 계절성 보정 적용
                month = target_date.month
                if seasonal_pattern and month in seasonal_pattern:
                    seasonal_factor = seasonal_pattern[month]
                    # 계절성 보정을 더 강하게 적용 (±25% 범위)
                    seasonal_adjustment = 1.0 + (seasonal_factor - 1.0) * 0.25
                    pred_value *= seasonal_adjustment

                # 🔥 트렌드 반영 (장기적 증가/감소 패턴)
                if recent_trend != 0:
                    trend_factor = 1.0 + (recent_trend * idx * 0.1)  # 시간이 지날수록 트렌드 강화
                    pred_value *= trend_factor

                # 🔥 개선된 자연스러운 변동 추가
                month_volatility = volatility_pattern.get(month, 0.08)  # 해당 월의 변동성 사용

                # 점진적 변동성 증가 (시간이 지날수록 불확실성 증가)
                base_volatility = month_volatility
                if idx == 0:
                    variation_range = base_volatility * 0.3  # 첫 번째는 매우 안정적
                elif idx <= 2:
                    variation_range = base_volatility * 0.6  # 초기 3개월은 안정적
                else:
                    variation_range = base_volatility * min(
                        1.5, 0.8 + (idx - 2) * 0.1
                    )  # 점진적 증가

                # 정규분포 기반 변동 (더 자연스러운 변동)
                natural_variation = np.random.normal(1.0, variation_range / 3)
                natural_variation = max(0.7, min(1.3, natural_variation))  # 극단적 변동 제한
                pred_value *= natural_variation

                # 🔥 최소값 보장 (0이 되지 않도록)
                pred_value = max(1, int(pred_value))

                predictions.append(
                    {"month": target_month, "value": pred_value, "type": "predicted"}
                )

                # 🔥 개선된 시퀀스 업데이트
                new_features = np.zeros(features.shape[1])
                new_features[0] = pred_scaled  # 예측된 입국자수

                # 계절성 특성 업데이트
                month_sin_idx = (
                    features.columns.get_loc("월_sin") if "월_sin" in features.columns else -1
                )
                month_cos_idx = (
                    features.columns.get_loc("월_cos") if "월_cos" in features.columns else -1
                )

                if month_sin_idx >= 0:
                    new_features[month_sin_idx] = np.sin(2 * np.pi * month / 12)
                if month_cos_idx >= 0:
                    new_features[month_cos_idx] = np.cos(2 * np.pi * month / 12)

                # 계절 특성 추가
                season = self.get_season_number(month)
                for s in range(1, 5):
                    season_col = f"계절_{s}"
                    if season_col in features.columns:
                        season_idx = features.columns.get_loc(season_col)
                        new_features[season_idx] = 1 if s == season else 0

                # 🔥 트렌드 특성 추가 (시간에 따른 변화 반영)
                if "트렌드" in features.columns:
                    trend_idx = features.columns.get_loc("트렌드")
                    new_features[trend_idx] = idx + 1  # 예측 시점

                # 시퀀스 업데이트 (슬라이딩 윈도우)
                sequence = np.roll(sequence, -1, axis=0)
                sequence[-1] = new_features

        return predictions

    def _predict_tourism_optimized(self, nationality, purpose, target_months):
        """관광 목적 특별 최적화 예측"""
        print("🏖️ 관광 전용 최적화 처리 시작...")
        
        key = f"{nationality}_{purpose}"
        
        # 1. 관광 전용 모델 학습 (강화된 설정)
        if key not in self.models:
            success = self._train_tourism_model(nationality, purpose)
            if not success:
                print("⚠️ 관광 최적화 실패, 기본 모델로 전환")
                # 기본 모델 학습 시도
                if not self.train_purpose_model(nationality, purpose):
                    return None
        
        # 2. 관광 데이터 특별 처리
        combo_data = (
            self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
            .copy()
            .sort_values("날짜")
        )
        
        # 3. 관광 특화 계절성 패턴 추출
        seasonal_pattern = self._extract_tourism_seasonal_pattern(combo_data)
        
        # 4. 관광 변동성 스무딩
        smoothed_data = self._apply_tourism_smoothing(combo_data)
        
        # 5. 관광 최적화 예측 실행
        model = self.models[key]
        scaler = self.scalers[key]
        
        # 특성 준비 (관광 최적화)
        features = self._create_tourism_features(smoothed_data)
        sequence_length = min(9, len(features) // 3)  # 관광은 더 긴 시퀀스
        recent_data = features.tail(sequence_length).copy()
        current_sequence = scaler.transform(recent_data)
        
        # 연속성 보정
        last_actual_value = combo_data["입국자수"].iloc[-1]
        last_actual_date = combo_data["날짜"].iloc[-1]
        recent_3months_avg = combo_data["입국자수"].tail(3).mean()
        
        print(f"관광 연속성 보정 기준: {last_actual_date.strftime('%Y-%m')} = {last_actual_value:,}명")
        print(f"관광 최근 3개월 평균: {recent_3months_avg:,}명")
        
        predictions = []
        sequence = current_sequence.copy()
        
        first_pred_month = target_months[0]
        first_pred_date = pd.to_datetime(first_pred_month + "-01")
        months_gap = (first_pred_date.year - last_actual_date.year) * 12 + (
            first_pred_date.month - last_actual_date.month
        )
        
        # 관광 전용 연속성 강도 (더 부드럽게)
        continuity_strength = max(0.6, 1.0 - (months_gap * 0.05))
        print(f"관광 연속성 강도: {continuity_strength:.2f} (간격: {months_gap}개월)")
        
        for idx, target_month in enumerate(target_months):
            target_date = pd.to_datetime(target_month + "-01")
            
            # 기존 데이터 확인
            existing_data = combo_data[combo_data["날짜"] == target_date]
            
            if len(existing_data) > 0:
                actual_value = existing_data["입국자수"].iloc[0]
                predictions.append({"month": target_month, "value": actual_value, "type": "actual"})
            else:
                # 예측값 계산
                pred_scaled = model.predict(sequence.reshape(1, sequence_length, -1), verbose=1)[0, 0]
                
                # 역스케일링
                dummy_data = np.zeros((1, features.shape[1]))
                dummy_data[0, 0] = pred_scaled
                pred_value = scaler.inverse_transform(dummy_data)[0, 0]
                
                # 🏖️ 관광 특화 후처리
                # 1. 계절성 적용
                month_num = target_date.month
                seasonal_factor = seasonal_pattern.get(month_num, 1.0)
                pred_value *= seasonal_factor
                
                # 🔄 2. 정밀화된 관광 연속성 보정 시스템 (90% 강도)
                if idx == 0:
                    # 첫 번째 예측: 실제값과 강한 연속성 (최대 90%)
                    adaptive_strength = min(0.9, continuity_strength + 0.2)  # 기본 강도 + 20%
                    continuity_factor = adaptive_strength * 0.85  # 85%까지
                    pred_value = (pred_value * (1 - continuity_factor)) + (last_actual_value * continuity_factor)
                    print(f"  🔗 첫 예측 연속성: {continuity_factor:.2f} 강도로 {last_actual_value:,.0f}명과 연결")
                    
                elif idx <= 2:
                    # 초기 2-3개월: 높은 연속성 유지
                    base_factor = 0.7 - (idx - 1) * 0.15  # 0.7 → 0.55 감소
                    continuity_factor = continuity_strength * base_factor
                    prev_value = predictions[-1]["value"]
                    pred_value = (pred_value * (1 - continuity_factor)) + (prev_value * continuity_factor)
                    print(f"  🔗 초기 연속성 ({idx+1}개월): {continuity_factor:.2f} 강도")
                    
                elif idx <= 5:
                    # 중기 3-6개월: 중간 연속성
                    base_factor = 0.4 - (idx - 3) * 0.08  # 0.4 → 0.24 감소
                    continuity_factor = continuity_strength * base_factor
                    prev_value = predictions[-1]["value"]
                    
                    # 급격한 변화 감지 시 연속성 강화
                    change_rate = abs(pred_value - prev_value) / prev_value if prev_value > 0 else 0
                    if change_rate > 0.15:  # 15% 이상 변화시
                        continuity_factor *= 1.5  # 연속성 50% 강화
                        print(f"  ⚡ 급변 감지 ({change_rate:.1%}): 연속성 {continuity_factor:.2f}로 강화")
                    
                    pred_value = (pred_value * (1 - continuity_factor)) + (prev_value * continuity_factor)
                    
                elif idx <= 8:
                    # 후기 7-9개월: 약한 연속성
                    base_factor = max(0.1, 0.2 - (idx - 6) * 0.03)  # 0.2 → 0.14 감소
                    continuity_factor = continuity_strength * base_factor
                    prev_value = predictions[-1]["value"]
                    
                    # 장기 트렌드 고려
                    if idx >= 3:
                        recent_trend = (predictions[-1]["value"] - predictions[-3]["value"]) / predictions[-3]["value"]
                        if abs(recent_trend) < 0.05:  # 안정적 트렌드인 경우
                            continuity_factor *= 1.2  # 연속성 20% 강화
                    
                    pred_value = (pred_value * (1 - continuity_factor)) + (prev_value * continuity_factor)
                    
                else:
                    # 장기 10개월+: 최소 연속성
                    base_factor = max(0.05, 0.15 - (idx - 9) * 0.01)  # 최소 5% 유지
                    continuity_factor = continuity_strength * base_factor
                    prev_value = predictions[-1]["value"]
                    pred_value = (pred_value * (1 - continuity_factor)) + (prev_value * continuity_factor)
                
                # 🔥 3. 강화된 관광 변동성 제어 시스템 (25% 제한)
                if idx > 0:
                    prev_value = predictions[-1]["value"]
                    
                    # 적응형 변동성 제한 (시간 경과에 따라 완화)
                    if idx <= 3:
                        max_change_rate = 0.20  # 초기 3개월: 20% 제한 (더 엄격)
                    elif idx <= 6:
                        max_change_rate = 0.25  # 중간 3개월: 25% 제한 (기본)
                    elif idx <= 9:
                        max_change_rate = 0.30  # 후반 3개월: 30% 제한 (완화)
                    else:
                        max_change_rate = 0.35  # 장기 예측: 35% 제한 (불확실성 반영)
                    
                    max_change = prev_value * max_change_rate
                    change = pred_value - prev_value
                    
                    # 변화량 제한 적용
                    if abs(change) > max_change:
                        limited_change = max_change if change > 0 else -max_change
                        pred_value = prev_value + limited_change
                        
                        # 제한 적용 로깅
                        if idx < 5:  # 초기 몇 개월만 로깅
                            print(f"  📊 관광 변동성 제어: {target_month} - 변화량 {change/prev_value*100:.1f}% → {max_change_rate*100:.0f}% 제한")
                    
                    # 추가: 급격한 감소 방지 (관광은 급감하지 않음)
                    min_value = prev_value * 0.80  # 전월 대비 80% 이상 유지
                    if pred_value < min_value:
                        pred_value = min_value
                        print(f"  🛡️ 관광 최소값 보장: {target_month} - {min_value:,.0f}명 이상 유지")
                
                # 4. 최소값 보장 (관광은 0이 될 수 없음)
                pred_value = max(pred_value, last_actual_value * 0.15)
                
                predictions.append({"month": target_month, "value": int(pred_value), "type": "predicted"})
                
                # 시퀀스 업데이트 (관광 최적화)
                if idx < len(target_months) - 1:
                    # 간단한 시퀀스 업데이트
                    new_features = np.zeros(features.shape[1])
                    new_features[0] = pred_value  # 입국자수 위치
                    new_features[1] = target_date.year  # 연도
                    new_features[2] = target_date.month  # 월
                    
                    new_sequence = scaler.transform(new_features.reshape(1, -1))[0]
                    sequence = np.roll(sequence, -1, axis=0)
                    sequence[-1] = new_sequence
        
        print(f"✅ 관광 최적화 예측 완료 - 변동성 제어 적용")
        return predictions
    
    def _train_tourism_model(self, nationality, purpose):
        """관광 전용 모델 학습"""
        print("🏖️ 관광 전용 모델 학습 시작...")
        
        combo_data = (
            self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
            .copy()
            .sort_values("날짜")
        )
        
        if len(combo_data) < 36:
            print("❌ 관광 모델 학습을 위한 최소 데이터 부족")
            return False
        
        # 관광 데이터 스무딩
        smoothed_data = self._apply_tourism_smoothing(combo_data)
        
        # 관광 특화 특성 생성
        features = self._create_tourism_features(smoothed_data)
        
        # 시퀀스 생성 (관광은 더 긴 시퀀스)
        sequence_length = min(9, len(features) // 3)
        X, y, scaler = self.create_sequences(features, sequence_length)
        
        if len(X) == 0:
            print("❌ 관광 시퀀스 생성 실패")
            return False
        
        # 관광 전용 모델 구조
        model = self._build_tourism_model(X.shape[1:], len(combo_data))
        
        # 학습 설정 (관광 최적화)
        split_idx = int(len(X) * 0.85)
        train_X, train_y = X[:split_idx], y[:split_idx]
        val_X, val_y = X[split_idx:], y[split_idx:]
        
        if len(val_X) == 0:
            # 검증 데이터가 없으면 단순 학습
            print("🏖️ 관광 모델 단순 학습 (검증 데이터 부족)")
            model.fit(train_X, train_y, epochs=50, batch_size=min(8, len(train_X)), verbose=1)
        else:
            # 관광 전용 콜백
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
            ]
            
            # 학습 실행
            print("🏖️ 관광 최적화 모델 학습 중...")
            model.fit(
                train_X, train_y,
                validation_data=(val_X, val_y),
                epochs=100,  # 관광은 더 긴 학습
                batch_size=min(16, max(4, len(train_X) // 15)),
                callbacks=callbacks,
                verbose=1
            )
        
        # 모델 저장
        key = f"{nationality}_{purpose}"
        self.models[key] = model
        self.scalers[key] = scaler
        
        print("✅ 관광 최적화 모델 학습 완료")
        return True
    
    def _apply_tourism_smoothing(self, data):
        """관광 데이터 스무딩 (변동성 감소)"""
        smoothed_data = data.copy()
        
        # 이동평균 스무딩 (더 부드럽게)
        smoothed_data["입국자수"] = smoothed_data["입국자수"].rolling(
            window=3, center=True, min_periods=1
        ).mean()
        
        return smoothed_data
    
    def _create_tourism_features(self, data):
        """🌍 관광 특화 강화된 계절성 특성 생성"""
        # 기본 특성 생성
        features = self.create_advanced_features(data)
        
        # 🏖️ 관광 전용 강화된 계절성 특성
        # 1. 다중 주기 계절성 (월별, 분기별, 반기별)
        features["강화계절_sin"] = np.sin(4 * np.pi * features["월"] / 12)  # 2배 주기
        features["강화계절_cos"] = np.cos(4 * np.pi * features["월"] / 12)
        features["분기계절_sin"] = np.sin(2 * np.pi * features["분기"] / 4)  # 분기별 계절성
        features["분기계절_cos"] = np.cos(2 * np.pi * features["분기"] / 4)
        features["반기계절_sin"] = np.sin(2 * np.pi * features["월"] / 6)   # 반기별 계절성
        features["반기계절_cos"] = np.cos(2 * np.pi * features["월"] / 6)
        
        # 2. 세분화된 휴가철/성수기 지표
        # 여름 성수기 (7-8월)
        features["여름성수기"] = features["월"].isin([7, 8]).astype(int)
        # 겨울 휴가철 (12-2월)
        features["겨울휴가철"] = features["월"].isin([12, 1, 2]).astype(int)
        # 봄 관광철 (4-5월)
        features["봄관광철"] = features["월"].isin([4, 5]).astype(int)
        # 가을 관광철 (9-11월)
        features["가을관광철"] = features["월"].isin([9, 10, 11]).astype(int)
        # 어깨철 (비성수기)
        features["어깨철"] = features["월"].isin([3, 6]).astype(int)
        
        # 3. 주요 관광 이벤트 기반 특성
        # 한국 벚꽃철 (4월)
        features["벚꽃철"] = (features["월"] == 4).astype(int)
        # 단풍철 (10-11월)
        features["단풍철"] = features["월"].isin([10, 11]).astype(int)
        # 스키철 (12-2월)
        features["스키철"] = features["월"].isin([12, 1, 2]).astype(int)
        # 해수욕철 (7-8월)
        features["해수욕철"] = features["월"].isin([7, 8]).astype(int)
        
        # 4. 날씨 기반 관광 특성
        # 더위지수 (여름철 관광 영향)
        features["더위지수"] = 0
        for month in [6, 7, 8]:
            month_mask = features["월"] == month
            if month == 6:
                features.loc[month_mask, "더위지수"] = 2
            elif month == 7:
                features.loc[month_mask, "더위지수"] = 3
            elif month == 8:
                features.loc[month_mask, "더위지수"] = 3
        
        # 추위지수 (겨울철 관광 영향)
        features["추위지수"] = 0
        for month in [12, 1, 2]:
            month_mask = features["월"] == month
            if month == 12:
                features.loc[month_mask, "추위지수"] = 2
            elif month == 1:
                features.loc[month_mask, "추위지수"] = 3
            elif month == 2:
                features.loc[month_mask, "추위지수"] = 2
        
        # 5. 관광 선호도 지수 (월별 가중치)
        tourism_preference = {1: 0.7, 2: 0.6, 3: 0.8, 4: 0.95, 5: 0.9, 6: 0.85,
                            7: 1.0, 8: 1.0, 9: 0.9, 10: 0.95, 11: 0.9, 12: 0.8}
        features["관광선호도"] = features["월"].map(tourism_preference).fillna(0.7)
        
        # 6. 강화된 관광 패턴 지표
        # 이동평균 기반 트렌드 (3개월, 6개월, 12개월)
        features["관광_트렌드_3m"] = features["입국자수"].rolling(3, min_periods=1).mean()
        features["관광_트렌드_6m"] = features["입국자수"].rolling(6, min_periods=1).mean()
        features["관광_트렌드_12m"] = features["입국자수"].rolling(12, min_periods=1).mean()
        
        # 계절별 변동성
        features["관광_변동성_3m"] = features["입국자수"].rolling(3, min_periods=1).std().fillna(0)
        features["관광_변동성_6m"] = features["입국자수"].rolling(6, min_periods=1).std().fillna(0)
        
        # 전년 동월 비교 (가능한 경우)
        if len(features) >= 12:
            features["전년동월_비율"] = features["입국자수"] / features["입국자수"].shift(12)
            features["전년동월_비율"] = features["전년동월_비율"].fillna(1.0)
        else:
            features["전년동월_비율"] = 1.0
        
        # 7. 계절성 상호작용 특성 (강화)
        features["월_x_관광선호도"] = features["월"] * features["관광선호도"]
        features["계절_x_관광선호도"] = features["계절"] * features["관광선호도"]
        features["여름성수기_x_입국자수"] = features["여름성수기"] * features["입국자수"]
        features["겨울휴가철_x_입국자수"] = features["겨울휴가철"] * features["입국자수"]
        
        # 8. 장기 패턴 추출
        # 계절성 강도 (해당 월의 평균 대비 비율)
        if len(features) >= 24:  # 2년 이상 데이터
            monthly_avg = features.groupby(features.index % 12)["입국자수"].transform('mean')
            overall_avg = features["입국자수"].mean()
            features["계절성_강도"] = monthly_avg / overall_avg if overall_avg > 0 else 1.0
        else:
            features["계절성_강도"] = 1.0
        
        print(f"🏖️ 관광 특화 강화 특성 생성 완료: {len([col for col in features.columns if any(keyword in col for keyword in ['계절', '성수기', '휴가', '관광', '벚꽃', '단풍', '스키', '해수욕'])])}개 계절성 특성")
        
        return features
    
    def _extract_tourism_seasonal_pattern(self, data):
        """관광 특화 계절성 패턴 추출"""
        monthly_avg = data.groupby(data["날짜"].dt.month)["입국자수"].mean()
        overall_avg = data["입국자수"].mean()
        
        # 계절성 비율 계산
        seasonal_pattern = {}
        for month in range(1, 13):
            if month in monthly_avg.index and overall_avg > 0:
                seasonal_pattern[month] = monthly_avg[month] / overall_avg
            else:
                seasonal_pattern[month] = 1.0
        
        return seasonal_pattern
    
    def _build_tourism_model(self, input_shape, data_size):
        """🏖️ 관광 전용 최적화 2층 LSTM 아키텍처"""
        # 데이터 크기에 따른 적응형 구조
        if data_size < 80:
            # 초소규모: 단일 LSTM + 강화된 정규화
            model = Sequential([
                LSTM(
                    48,  # 뉴런 수 증가 (32→48)
                    input_shape=input_shape, 
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    dropout=0.25,  # 드롭아웃 조정
                    recurrent_dropout=0.15,
                    return_sequences=False
                ),
                BatchNormalization(momentum=0.9),  # 배치정규화 강화
                Dropout(0.35),  # 드롭아웃 증가
                Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1, activation="linear", dtype="float32"),
            ])
            print(f"🏖️ 관광 소규모 강화 모델 구축 (데이터: {data_size}개, 뉴런: 48)")
            
        elif data_size < 150:
            # 중규모: 개선된 2층 LSTM
            model = Sequential([
                # 첫 번째 LSTM층 (장기 패턴 감지)
                LSTM(
                    80,  # 뉴런 수 증가 (64→80)
                    return_sequences=True,
                    input_shape=input_shape,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    dropout=0.3,
                    recurrent_dropout=0.2,
                ),
                BatchNormalization(momentum=0.95),  # 배치정규화 강화
                Dropout(0.35),
                
                # 두 번째 LSTM층 (단기 패턴 정제)
                LSTM(
                    40,  # 뉴런 수 증가 (32→40)
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    dropout=0.25,
                    recurrent_dropout=0.15,
                    return_sequences=False
                ),
                BatchNormalization(momentum=0.9),
                Dropout(0.4),  # 드롭아웃 강화
                
                # 강화된 완전연결층
                Dense(48, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(24, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
                Dropout(0.2),
                Dense(1, activation="linear", dtype="float32"),
            ])
            print(f"🏖️ 관광 최적화 2층 LSTM 모델 구축 (데이터: {data_size}개, 뉴런: 80→40)")
        else:
            # 대규모: 고성능 3층 LSTM
            model = Sequential([
                # 첫 번째 LSTM층 (장기 트렌드 감지)
                LSTM(
                    96,  # 뉴런 수 증가
                    return_sequences=True,
                    input_shape=input_shape,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    dropout=0.25,
                    recurrent_dropout=0.15,
                ),
                BatchNormalization(momentum=0.95),
                Dropout(0.3),
                
                # 두 번째 LSTM층 (중기 패턴 감지)
                LSTM(
                    64,
                    return_sequences=True,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    dropout=0.3,
                    recurrent_dropout=0.2,
                ),
                BatchNormalization(momentum=0.9),
                Dropout(0.35),
                
                # 세 번째 LSTM층 (단기 정밀 예측)
                LSTM(
                    32,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    dropout=0.25,
                    recurrent_dropout=0.15,
                    return_sequences=False
                ),
                BatchNormalization(momentum=0.9),
                Dropout(0.4),
                
                # 고도화된 완전연결층
                Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1, activation="linear", dtype="float32"),
            ])
            print(f"🏖️ 관광 고성능 3층 LSTM 모델 구축 (데이터: {data_size}개, 뉴런: 96→64→32)")
        
        # 관광 전용 최적화 컴파일 설정
        # 적응형 학습률
        if data_size < 80:
            learning_rate = 0.002  # 소규모는 높은 학습률
        elif data_size < 150:
            learning_rate = 0.0015  # 중규모는 중간 학습률
        else:
            learning_rate = 0.001  # 대규모는 안정적 학습률
        
        if self.performance_mode == "m1_optimized":
            from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
            optimizer = LegacyAdam(
                learning_rate=learning_rate,
                beta_1=0.9,      # 모멘텀 최적화
                beta_2=0.999,    # RMSprop 최적화
                epsilon=1e-7,    # 수치 안정성
                clipnorm=1.0     # 그래디언트 클리핑
            )
            print(f"🏖️ M1 최적화: 관광 전용 Legacy Adam optimizer (lr={learning_rate})")
        else:
            optimizer = Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0
            )
            print(f"🏖️ 관광 전용 최적화 Adam optimizer (lr={learning_rate})")
        
        # 손실함수 개선
        model.compile(
            optimizer=optimizer,
            loss="huber",  # Huber 손실 (MSE + MAE의 장점 결합)
            metrics=["mae", "mse"]
        )
        
        return model

    def extract_seasonal_pattern(self, data):
        """계절성 패턴 추출"""
        seasonal_pattern = {}
        for month in data["월"].unique():
            monthly_data = data[data["월"] == month]
            seasonal_pattern[month] = monthly_data["입국자수"].mean()
        return seasonal_pattern

    def extract_improved_seasonal_pattern(self, data):
        """개선된 계절성 패턴 추출 - 정규화된 계절성 팩터"""
        if len(data) < 12:
            return {}

        seasonal_pattern = {}
        overall_avg = data["입국자수"].mean()

        for month in range(1, 13):
            monthly_data = data[data["월"] == month]
            if len(monthly_data) > 0:
                month_avg = monthly_data["입국자수"].mean()
                # 전체 평균 대비 비율로 계산 (1.0 = 평균, 1.2 = 20% 높음)
                seasonal_factor = month_avg / overall_avg if overall_avg > 0 else 1.0
                seasonal_pattern[month] = seasonal_factor
            else:
                seasonal_pattern[month] = 1.0

        return seasonal_pattern

    def calculate_recent_trend(self, data):
        """최근 트렌드 계산 - 최근 12개월 평균 변화율"""
        if len(data) < 12:
            return 0.0

        recent_12months = data.tail(12)
        if len(recent_12months) < 6:
            return 0.0

        # 선형 회귀를 통한 트렌드 계산
        x = np.arange(len(recent_12months))
        y = recent_12months["입국자수"].values

        # 최소제곱법으로 기울기 계산
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            avg_value = np.mean(y)
            # 월 평균 변화율로 정규화
            trend_rate = slope / avg_value if avg_value > 0 else 0.0
            return max(-0.1, min(0.1, trend_rate))  # ±10% 범위로 제한

        return 0.0

    def analyze_volatility_pattern(self, data):
        """변동성 패턴 분석 - 월별 변동성 계산"""
        volatility_pattern = {}

        for month in range(1, 13):
            monthly_data = data[data["월"] == month]
            if len(monthly_data) > 1:
                # 월별 데이터의 표준편차를 평균으로 나눈 변동계수
                std_dev = monthly_data["입국자수"].std()
                mean_val = monthly_data["입국자수"].mean()
                volatility = std_dev / mean_val if mean_val > 0 else 0.08
                volatility_pattern[month] = max(0.02, min(0.2, volatility))  # 2%~20% 범위
            else:
                volatility_pattern[month] = 0.08  # 기본값 8%

        return volatility_pattern

    def get_season_number(self, month):
        """월을 계절로 변환"""
        if month in [12, 1, 2]:
            return 1
        elif month in [3, 4, 5]:
            return 2
        elif month in [6, 7, 8]:
            return 3
        else:
            return 4

    def save_comprehensive_report(self):
        """통합 리포트 생성 (CSV 1개 + 그래프 1개) - 사용자 요청만 처리"""
        if not self.performance_results:
            print("저장할 성능 평가 결과가 없습니다.")
            return

        # 성능 결과 데이터프레임 생성
        performance_df = pd.DataFrame(self.performance_results)

        # 🔥 리포트 데이터 생성 시 누락 체크 강화
        report_data = []

        for _, row in performance_df.iterrows():
            nationality = row["nationality"]
            purpose = row["purpose"]

            # 각 지표별 데이터 정리
            report_row = {
                "국적": nationality,
                "목적": purpose,
                "학습샘플수": row["training_samples"],
                "검증샘플수": row["validation_samples"],
                "학습에포크": row["epochs_trained"],
                # MAE (낮을수록 좋음)
                "MAE_실제값": f"{row['mae']:,.0f}",
                "MAE_기준값": f"{row['mae_기준값']:,.0f}",
                "MAE_달성여부": "↓" if row["mae"] <= row["mae_기준값"] else "↑",
                "MAE_등급": row["mae_등급"],
                # RMSE (낮을수록 좋음)
                "RMSE_실제값": f"{row['rmse']:,.0f}",
                "RMSE_기준값": f"{row['rmse_기준값']:,.0f}",
                "RMSE_달성여부": "↓" if row["rmse"] <= row["rmse_기준값"] else "↑",
                "RMSE_등급": row["rmse_등급"],
                # R² (높을수록 좋음)
                "R2_실제값": f"{row['r2_score']:.4f}",
                "R2_기준값": f"{row['r2_score_기준값']:.2f}",
                "R2_달성여부": "↑" if row["r2_score"] >= row["r2_score_기준값"] else "↓",
                "R2_등급": row["r2_score_등급"],
                # MAPE (낮을수록 좋음)
                "MAPE_실제값": f"{row['mape']:.1f}%",
                "MAPE_기준값": f"{row['mape_기준값']:.1f}%",
                "MAPE_달성여부": "↓" if row["mape"] <= row["mape_기준값"] else "↑",
                "MAPE_등급": row["mape_등급"],
                # F1 Score (높을수록 좋음)
                "F1_실제값": f"{row['f1_score']:.3f}",
                "F1_기준값": f"{row['f1_score_기준값']:.2f}",
                "F1_달성여부": "↑" if row["f1_score"] >= row["f1_score_기준값"] else "↓",
                "F1_등급": row["f1_score_등급"],
                # 🔥 학습 로그 정보 추가
                "최종학습손실": f"{row['final_train_loss']:.6f}",
                "최종검증손실": (
                    f"{row['final_val_loss']:.6f}" if row["final_val_loss"] is not None else "N/A"
                ),
                "최종학습MAE": f"{row['final_train_mae']:.6f}",
                "최종검증MAE": (
                    f"{row['final_val_mae']:.6f}" if row["final_val_mae"] is not None else "N/A"
                ),
                "최고학습손실": f"{row['best_train_loss']:.6f}",
                "최고검증손실": (
                    f"{row['best_val_loss']:.6f}" if row["best_val_loss"] != float('inf') else "N/A"
                ),
                "조기종료여부": "예" if row["early_stopped"] else "아니오",
                "학습률": f"{row['learning_rate_used']:.6f}",
                "생성시간": row["timestamp"],
            }

            report_data.append(report_row)

        # 리포트 데이터프레임 생성
        report_df = pd.DataFrame(report_data)

        # 사용자 요청한 결과만 리포트 생성
        print(f"[리포트] {len(report_df)}개 모델의 성능 결과를 리포트에 포함")

        # CSV 저장 (타임스탬프 디렉토리에)
        report_path = f"{self.results_dir}/리포트.csv"
        report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"통합 리포트 저장: {report_path}")

        # 종합 성능 그래프 생성
        self.create_comprehensive_performance_chart(performance_df)

        # 콘솔 요약 출력
        self.print_summary_statistics(performance_df)

        return report_path

    def create_comprehensive_performance_chart(self, performance_df):
        """종합 성능 차트 생성 - 범례 최적화"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 모델명 생성
        model_names = [
            f"{row['nationality']}-{row['purpose']}" for _, row in performance_df.iterrows()
        ]

        # 1. MAE vs 기준값 비교
        mae_actual = performance_df["mae"].values
        mae_threshold = performance_df["mae_기준값"].values

        x_pos = np.arange(len(model_names))
        width = 0.35

        ax1.bar(x_pos - width / 2, mae_actual, width, label="실제값", color="lightcoral", alpha=0.8)
        ax1.bar(
            x_pos + width / 2, mae_threshold, width, label="기준값", color="lightblue", alpha=0.8
        )

        ax1.set_title("MAE 성능 비교 (낮을수록 좋음)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("MAE")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha="right")
        
        # 🔥 범례 최적화
        self._create_optimized_legend(ax1, position="upper right")

        ax1.grid(True, alpha=0.3)

        # 2. R² Score vs 기준값 비교
        r2_actual = performance_df["r2_score"].values
        r2_threshold = performance_df["r2_score_기준값"].values

        ax2.bar(x_pos - width / 2, r2_actual, width, label="실제값", color="lightgreen", alpha=0.8)
        ax2.bar(x_pos + width / 2, r2_threshold, width, label="기준값", color="gold", alpha=0.8)

        ax2.set_title("R² Score 성능 비교 (높을수록 좋음)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("R² Score")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha="right")
        
        # 🔥 범례 최적화
        self._create_optimized_legend(ax2, position="upper left")

        ax2.grid(True, alpha=0.3)

        # 3. 종합 달성률 차트
        metrics = ["MAE", "RMSE", "R²", "MAPE", "F1"]

        # 각 모델별 달성률 계산
        achievement_data = []
        for _, row in performance_df.iterrows():
            achievements = []
            achievements.append(100 if row["mae"] <= row["mae_기준값"] else 0)
            achievements.append(100 if row["rmse"] <= row["rmse_기준값"] else 0)
            achievements.append(100 if row["r2_score"] >= row["r2_score_기준값"] else 0)
            achievements.append(100 if row["mape"] <= row["mape_기준값"] else 0)
            achievements.append(100 if row["f1_score"] >= row["f1_score_기준값"] else 0)
            achievement_data.append(achievements)

        # 평균 달성률 계산
        avg_achievements = np.mean(achievement_data, axis=0)

        colors = ["red", "orange", "green", "blue", "purple"]
        bars = ax3.bar(metrics, avg_achievements, color=colors, alpha=0.7)
        ax3.set_title("평균 달성률 (%)", fontsize=14, fontweight="bold")
        ax3.set_ylabel("달성률 (%)")
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        # 수치 표시
        for bar, value in zip(bars, avg_achievements):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 2,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. 학습 정보 요약
        epochs = performance_df["epochs_trained"].values
        train_samples = performance_df["training_samples"].values

        scatter = ax4.scatter(
            train_samples, epochs, s=200, alpha=0.7, c=range(len(model_names)), cmap="viridis"
        )

        # 🔥 모델명 라벨 최적화 (겹침 방지)
        self._add_optimized_labels(ax4, train_samples, epochs, model_names)

        ax4.set_title("학습 정보 (샘플수 vs 에포크)", fontsize=14, fontweight="bold")
        ax4.set_xlabel("학습 샘플 수")
        ax4.set_ylabel("학습 에포크")
        ax4.grid(True, alpha=0.3)

        # 전체 제목
        fig.suptitle("모델 성능 종합 리포트", fontsize=18, fontweight="bold")

        plt.tight_layout()

        # 🔥 그래프 저장 (타임스탬프 디렉토리에)
        chart_path = f"{self.results_dir}/성능_리포트.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        print(f"종합 성능 차트 저장: {chart_path}")

        plt.show()

    def _create_optimized_legend(self, ax, position="auto"):
        """🔥 최적화된 범례 생성"""
        if position == "auto":
            # 그래프 내용에 따라 자동 위치 결정
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # 데이터 분포에 따라 위치 결정
            if ylim[1] > ylim[0] * 2:  # 세로로 긴 경우
                position = "upper right"
            else:
                position = "upper left"
        
        # 범례 스타일 최적화
        legend = ax.legend(
            fontsize=11,
            loc=position,
            frameon=True,
            fancybox=True,
            shadow=True,
            borderpad=1.0,
            columnspacing=1.0,
            ncol=1,  # 세로 배치로 겹침 방지
            bbox_to_anchor=None
        )
        
        # 범례 프레임 스타일 개선
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.9)
        frame.set_edgecolor('gray')
        frame.set_linewidth(1.0)

    def _add_optimized_labels(self, ax, x_values, y_values, labels):
        """🔥 최적화된 라벨 추가 (겹침 방지)"""
        from matplotlib.patches import Rectangle
        
        # 라벨 간격 계산
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)
        
        # 겹침 방지를 위한 최소 간격
        min_x_gap = x_range * 0.05
        min_y_gap = y_range * 0.05
        
        placed_labels = []
        
        for i, (x, y, label) in enumerate(zip(x_values, y_values, labels)):
            # 기존 라벨과의 거리 확인
            too_close = False
            for placed_x, placed_y in placed_labels:
                if abs(x - placed_x) < min_x_gap and abs(y - placed_y) < min_y_gap:
                    too_close = True
                    break
            
            if not too_close:
                # 라벨 위치 결정
                if i % 2 == 0:
                    xytext = (5, 5)
                    va = "bottom"
                    ha = "left"
                else:
                    xytext = (-5, -15)
                    va = "top"
                    ha = "right"
                
                # 라벨 추가
                ax.annotate(
                    label,
                    (x, y),
                    xytext=xytext,
                    textcoords="offset points",
                    fontsize=9,
                    ha=ha,
                    va=va,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="gray",
                        linewidth=0.5
                    ),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.1",
                        color="gray",
                        alpha=0.7,
                        lw=1
                    )
                )
                
                placed_labels.append((x, y))
            else:
                # 겹치는 경우 간단한 점만 표시
                ax.annotate(
                    f"•",
                    (x, y),
                    xytext=(0, 0),
                    textcoords="offset points",
                    fontsize=12,
                    ha="center",
                    va="center",
                    color="red"
                )

    def print_summary_statistics(self, performance_df):
        """요약 통계 출력"""
        print(f"\n" + "=" * 80)
        print(f"모델 성능 종합 요약")
        print(f"=" * 80)

        total_models = len(performance_df)
        print(f"총 평가 모델 수: {total_models}개")

        # 주요 지표별 달성률
        metrics_info = [
            ("MAE", "mae", "mae_기준값", True),
            ("RMSE", "rmse", "rmse_기준값", True),
            ("R²", "r2_score", "r2_score_기준값", False),
            ("MAPE", "mape", "mape_기준값", True),
            ("F1", "f1_score", "f1_score_기준값", False),
        ]

        print(f"\n지표별 달성 현황:")
        print(f"-" * 80)

        overall_achievements = []

        for name, actual_col, threshold_col, lower_better in metrics_info:
            if lower_better:
                achieved = (performance_df[actual_col] <= performance_df[threshold_col]).sum()
            else:
                achieved = (performance_df[actual_col] >= performance_df[threshold_col]).sum()

            achievement_rate = (achieved / total_models) * 100
            overall_achievements.append(achievement_rate)

            avg_actual = performance_df[actual_col].mean()
            avg_threshold = performance_df[threshold_col].mean()

            print(
                f"{name:6}: {achieved:2}/{total_models} 달성 ({achievement_rate:5.1f}%) | "
                f"평균 {avg_actual:8.3f} (기준: {avg_threshold:6.3f})"
            )

        # 전체 달성률
        overall_rate = np.mean(overall_achievements)
        print(f"\n전체 평균 달성률: {overall_rate:.1f}%")

        if overall_rate >= 80:
            print("상태: 우수 - 대부분 지표에서 기준 달성")
        elif overall_rate >= 60:
            print("상태: 양호 - 많은 지표에서 기준 달성")
        elif overall_rate >= 40:
            print("상태: 보통 - 일부 지표에서 개선 필요")
        else:
            print("상태: 개선 필요 - 다수 지표에서 기준 미달성")

        print(f"=" * 80)

    def find_nationality_simple(self, input_text, nationalities):
        """강화된 국가 매핑 (한글/영어 지원)"""
        input_text = input_text.lower().strip()

        # 직접 매칭 (대소문자 무시)
        for nat in nationalities:
            if input_text == nat.lower():
                return nat

        # 부분 매칭
        for nat in nationalities:
            if input_text in nat.lower() or nat.lower() in input_text:
                return nat

        # 확장된 한영 매핑
        mapping = {
            # 기존 매핑
            "대만": "대만", "taiwan": "대만", "tw": "대만",
            "중국": "중국", "china": "중국", "cn": "중국", "중": "중국",
            "일본": "일본", "japan": "일본", "jp": "일본", "일": "일본",
            "미국": "미국", "usa": "미국", "america": "미국", "us": "미국", "미": "미국",
            "태국": "태국", "thailand": "태국", "th": "태국", "태": "태국",
            "베트남": "베트남", "vietnam": "베트남", "vn": "베트남", "베": "베트남",
            "싱가포르": "싱가포르", "singapore": "싱가포르", "sg": "싱가포르", "싱": "싱가포르",
            # 추가 매핑
            "홍콩": "홍콩", "hongkong": "홍콩", "hk": "홍콩", "홍": "홍콩",
            "필리핀": "필리핀", "philippines": "필리핀", "ph": "필리핀", "필": "필리핀",
            "인도네시아": "인도네시아", "indonesia": "인도네시아", "id": "인도네시아", "인": "인도네시아",
            "말레이시아": "말레이시아", "malaysia": "말레이시아", "my": "말레이시아", "말": "말레이시아",
            "인도": "인도", "india": "인도", "in": "인도",
            "영국": "영국", "uk": "영국", "britain": "영국", "영": "영국",
            "프랑스": "프랑스", "france": "프랑스", "fr": "프랑스", "프": "프랑스",
            "독일": "독일", "germany": "독일", "de": "독일", "독": "독일",
            "이탈리아": "이탈리아", "italy": "이탈리아", "it": "이탈리아", "이": "이탈리아",
            "스페인": "스페인", "spain": "스페인", "es": "스페인", "스": "스페인",
            "러시아": "러시아(연방)", "russia": "러시아(연방)", "ru": "러시아(연방)", "러": "러시아(연방)",
            "캐나다": "캐나다", "canada": "캐나다", "ca": "캐나다", "캐": "캐나다",
            "호주": "오스트레일리아", "australia": "오스트레일리아", "au": "오스트레일리아", "호": "오스트레일리아",
            "브라질": "브라질", "brazil": "브라질", "br": "브라질", "브": "브라질",
            "몽골": "몽골", "mongolia": "몽골", "mn": "몽골", "몽": "몽골",
        }

        if input_text in mapping:
            target = mapping[input_text]
            for nat in nationalities:
                if target in nat:
                    return nat

        return None

    def safe_input_nationality(self, nationalities):
        """안전한 국적 입력 처리"""
        while True:
            try:
                try:
                    nationality_input = input(
                        "국적을 입력하세요 (한글/영어 가능, 'list'로 전체 목록 보기): "
                    ).strip()
                except UnicodeDecodeError:
                    nationality_input = input(
                        "국적을 영어로 입력하세요 (또는 'list'로 전체 목록 보기): "
                    ).strip()

                if not nationality_input:
                    print("빈 값은 입력할 수 없습니다. 다시 입력해주세요.")
                    continue

                if nationality_input.lower() == "list":
                    print(f"\n전체 국가 목록:")
                    for i, nat in enumerate(nationalities, 1):
                        print(f"  {i:2}. {nat}")
                    continue

                # 국가 매핑을 통한 찾기 (간단한 문자열 매칭)
                found_nationality = self.find_nationality_simple(nationality_input, nationalities)

                if found_nationality:
                    print(f"선택된 국적: {found_nationality}")
                    return found_nationality
                else:
                    print(f"'{nationality_input}'에 해당하는 국적을 찾을 수 없습니다.")
                    print("사용 가능한 영어 표현 예시: 대만, 중국, 일본, 미국 등")
                    continue

            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                return None
            except Exception as e:
                print(f"입력 처리 중 오류가 발생했습니다: {e}")
                continue

    def safe_input_purpose(self, nationality, available_purposes):
        """안전한 목적 입력 처리"""
        while True:
            try:
                print(f"\n{nationality}의 사용 가능한 목적:")
                for i, purpose in enumerate(available_purposes, 1):
                    data_count = len(
                        self.data[
                            (self.data["국적"] == nationality) & (self.data["목적"] == purpose)
                        ]
                    )
                    print(f"  {i}. {purpose} ({data_count}개월 데이터)")

                try:
                    purpose_input = input("목적을 입력하세요 (번호 또는 이름, 전체는 'all'): ").strip()
                except UnicodeDecodeError:
                    purpose_input = input("목적을 영어로 입력하세요 (번호 또는 이름, 전체는 'all'): ").strip()

                if not purpose_input:
                    print("빈 값은 입력할 수 없습니다. 다시 입력해주세요.")
                    continue

                if purpose_input.lower() in ["all", "none", "전체"]:
                    print("전체 목적별 예측을 선택했습니다.")
                    return None

                # 번호로 입력한 경우
                if purpose_input.isdigit():
                    idx = int(purpose_input) - 1
                    if 0 <= idx < len(available_purposes):
                        selected_purpose = available_purposes[idx]
                        print(f"선택된 목적: {selected_purpose}")
                        return selected_purpose
                    else:
                        print(
                            f"잘못된 번호입니다. 1-{len(available_purposes)} 사이의 번호를 입력하세요."
                        )
                        continue

                # 이름으로 입력한 경우
                for purpose in available_purposes:
                    if purpose_input.lower() in purpose.lower():
                        print(f"선택된 목적: {purpose}")
                        return purpose

                print(f"'{purpose_input}'에 해당하는 목적을 찾을 수 없습니다.")
                continue

            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                return None
            except Exception as e:
                print(f"입력 처리 중 오류가 발생했습니다: {e}")
                continue

    def safe_input_date(self, date_type="시작"):
        """안전한 날짜 입력 처리"""
        while True:
            try:
                try:
                    date_input = input(f"{date_type} 날짜를 입력하세요 (예: 2025-07): ").strip()
                except UnicodeDecodeError:
                    date_input = input(f"{date_type} date (YYYY-MM): ").strip()

                if not date_input:
                    print("빈 값은 입력할 수 없습니다. 다시 입력해주세요.")
                    continue

                # 날짜 형식 검증
                if re.match(r"^\d{4}-\d{2}$", date_input):
                    year, month = map(int, date_input.split("-"))
                    if 1 <= month <= 12:
                        print(f"{date_type} 날짜: {date_input}")
                        return date_input
                    else:
                        print("월은 01-12 사이여야 합니다.")
                        continue
                else:
                    print("올바른 형식: YYYY-MM (예: 2025-07)")
                    continue

            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                return None
            except Exception as e:
                print(f"입력 처리 중 오류가 발생했습니다: {e}")
                continue

    def create_prediction_visualization(self, nationality, results, start_date, end_date):
        """예측 결과 시각화 생성 (고급 이중 그래프 버전)"""
        print(f"\n{nationality} 고급 예측 결과 시각화 생성 중...")

        if not results:
            print("시각화할 예측 결과가 없습니다.")
            return

        num_purposes = len(results)
        purposes = list(results.keys())

        # 목적별 데이터 규모 분석
        purpose_scales = {}
        all_combo_data = {}

        for purpose in purposes:
            combo_data = (
                self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)]
                .copy()
                .sort_values("날짜")
            )
            all_combo_data[purpose] = combo_data

            if len(combo_data) > 0:
                display_data = combo_data.tail(60)  # 5년치
                avg_value = display_data["입국자수"].mean()
                purpose_scales[purpose] = avg_value
                print(f"{purpose}: 평균 {avg_value:,.0f}명")
            else:
                purpose_scales[purpose] = 0

        # 가장 큰 값을 가진 목적 찾기 (주요 목적)
        max_purpose = max(purpose_scales, key=purpose_scales.get) if purpose_scales else purposes[0]
        max_value = purpose_scales[max_purpose]

        # 이중 Y축 기준: 최대값의 1/10 이하면 우측 축 사용
        threshold = max_value / 10 if max_value > 0 else 0
        left_purposes = []  # 좌측 Y축 (큰 값들)
        right_purposes = []  # 우측 Y축 (작은 값들)

        for purpose, avg_val in purpose_scales.items():
            if avg_val >= threshold:
                left_purposes.append(purpose)
            else:
                right_purposes.append(purpose)

        print(f"좌측 Y축 (주요): {left_purposes}")
        print(f"우측 Y축 (보조): {right_purposes}")

        # 전체 높이 확장
        fig = plt.figure(figsize=(20, 14 + 6 * num_purposes))

        # 그리드 설정: 첫 번째 그래프 비율 대폭 증가
        gs = fig.add_gridspec(
            num_purposes + 1,
            1,
            height_ratios=[5] + [4] * num_purposes,  # 첫 번째 그래프 더 크게
            hspace=0.5,
        )

        # ================================
        # 상단: 전체 추이 통합 그래프 (이중 Y축)
        # ================================
        ax_overview = fig.add_subplot(gs[0, 0])

        # 우측 Y축 생성
        ax_right = ax_overview.twinx() if right_purposes else None

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#F4A460"]
        color_idx = 0

        # 좌측 Y축 목적들 그리기 (주요 목적 강조)
        for purpose in left_purposes:
            color = colors[color_idx % len(colors)]
            color_idx += 1

            combo_data = all_combo_data[purpose]
            display_data = combo_data.tail(60) if len(combo_data) > 0 else pd.DataFrame()

            if len(display_data) > 0:
                # 가장 큰 목적 vs 기타 목적 구분
                if purpose == max_purpose:
                    # 주요 목적: 진한 색상, 굵은 선, 완전 불투명
                    ax_overview.plot(
                        display_data["날짜"],
                        display_data["입국자수"],
                        color=color,
                        linewidth=6,  # 더 굵은 선
                        label=f"{purpose} (주요 수요)",
                        alpha=1.0,  # 완전 불투명
                        zorder=5,  # 최상위 레이어
                        marker="o",
                        markersize=8,
                        markerfacecolor="white",
                        markeredgewidth=3,
                        markeredgecolor=color,
                    )
                else:
                    # 기타 목적: 옅은 색상, 얇은 선, 반투명
                    ax_overview.plot(
                        display_data["날짜"],
                        display_data["입국자수"],
                        color=color,
                        linewidth=3,  # 얇은 선
                        label=f"{purpose}",
                        alpha=0.5,  # 반투명
                        zorder=3,  # 중간 레이어
                        marker="o",
                        markersize=4,
                        linestyle="-",
                    )

        # 우측 Y축 목적들 그리기 (보조 목적들은 더 옅게)
        if ax_right and right_purposes:
            for purpose in right_purposes:
                color = colors[color_idx % len(colors)]
                color_idx += 1

                combo_data = all_combo_data[purpose]
                display_data = combo_data.tail(60) if len(combo_data) > 0 else pd.DataFrame()

                if len(display_data) > 0:
                    # 우축은 일반적으로 작은 값들이므로 더 옅게 표시
                    ax_right.plot(
                        display_data["날짜"],
                        display_data["입국자수"],
                        color=color,
                        linewidth=2,  # 얇은 선
                        label=f"{purpose} (보조축)",
                        alpha=0.4,  # 더 반투명
                        zorder=2,  # 배경 레이어
                        linestyle="--",  # 점선으로 구분
                        marker="s",
                        markersize=3,
                    )

        # 코로나 기간 마스크
        if num_purposes > 0 and len(all_combo_data[purposes[0]]) > 0:
            first_combo_data = all_combo_data[purposes[0]]
            covid_data = first_combo_data[first_combo_data["코로나기간"] == 1]

            if len(covid_data) > 0:
                covid_start_date = covid_data["날짜"].min()
                covid_end_date = covid_data["날짜"].max()

                ax_overview.axvspan(
                    covid_start_date,
                    covid_end_date,
                    alpha=0.3,
                    color="red",
                    label="코로나 기간",
                    zorder=1,
                )

        # 예측 기간 하이라이트
        pred_start = pd.to_datetime(start_date + "-01")
        pred_end = pd.to_datetime(end_date + "-01")
        ax_overview.axvspan(
            pred_start,
            pred_end,
            alpha=0.2,
            color="orange",
            label="예측 구간 (하단 상세)",
            zorder=1,
        )

        # 축 설정 및 범례
        ax_overview.set_title(
            f"{nationality} 전체 목적별 입국자 추이 (이중 Y축 - 주요 목적 강조)",
            fontsize=20,
            fontweight="bold",
            pad=20,
        )
        ax_overview.set_ylabel(
            "입국자수 - 주요 목적 (명)", fontsize=16, color="blue", fontweight="bold"
        )
        ax_overview.tick_params(axis="y", labelcolor="blue", labelsize=14)

        if ax_right:
            ax_right.set_ylabel(
                "입국자수 - 보조 목적 (명)", fontsize=16, color="red", fontweight="bold"
            )
            ax_right.tick_params(axis="y", labelcolor="red", labelsize=14)

        # 통합 범례 (크기 증가, 위치 조정)
        lines1, labels1 = ax_overview.get_legend_handles_labels()
        if ax_right:
            lines2, labels2 = ax_right.get_legend_handles_labels()
            ax_overview.legend(
                lines1 + lines2,
                labels1 + labels2,
                fontsize=14,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                frameon=True,
                fancybox=True,
                shadow=True,
            )
        else:
            ax_overview.legend(
                fontsize=14,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                frameon=True,
                fancybox=True,
                shadow=True,
            )

        ax_overview.grid(True, alpha=0.3, linestyle="--")
        ax_overview.tick_params(axis="x", rotation=45, labelsize=14)

        # Y축 포맷
        ax_overview.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
        if ax_right:
            ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        # ================================
        # 하단: 목적별 예측 상세 그래프들
        # ================================

        for idx, (purpose, predictions) in enumerate(results.items()):
            ax = fig.add_subplot(gs[idx + 1, 0])

            # 해당 목적의 최근 데이터
            combo_data = all_combo_data[purpose]
            recent_data = combo_data.tail(18) if len(combo_data) > 0 else pd.DataFrame()

            # 실제 데이터 플롯
            if len(recent_data) > 0:
                ax.plot(
                    recent_data["날짜"],
                    recent_data["입국자수"],
                    "b-",
                    linewidth=4,
                    label="실제값",
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

            # 🔥 실제-예측 매끄러운 연결선 (개선된 버전)
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
                # 🔥 매끄러운 연결을 위한 중간 포인트 생성
                time_diff = (first_pred_date - last_actual_date).days
                value_diff = first_pred_value - last_actual_value

                # 월 간격 계산 (자연스러운 전환을 위해)
                months_gap = (first_pred_date.year - last_actual_date.year) * 12 + (
                    first_pred_date.month - last_actual_date.month
                )

                if months_gap <= 2:
                    # 간격이 짧으면 중간 포인트 1개 추가
                    mid_date = last_actual_date + pd.DateOffset(months=months_gap // 2 + 1)
                    mid_value = last_actual_value + (value_diff * 0.6)  # 60% 지점에서 부드럽게 전환

                    # 3점 곡선 연결 (실제값 → 중간점 → 예측값)
                    ax.plot(
                        [last_actual_date, mid_date, first_pred_date],
                        [last_actual_value, mid_value, first_pred_value],
                        color="#666666",
                        linestyle="-",
                        linewidth=4,
                        alpha=0.8,
                        label="실제-예측 연결(매끄러운)" if idx == 0 else "",
                        zorder=3,
                    )

                    # 중간 포인트 표시
                    ax.plot(
                        mid_date,
                        mid_value,
                        "o",
                        color="#666666",
                        markersize=8,
                        alpha=0.7,
                        zorder=3,
                    )

                elif months_gap <= 6:
                    # 간격이 중간이면 중간 포인트 2개 추가
                    mid1_date = last_actual_date + pd.DateOffset(months=months_gap // 3 + 1)
                    mid2_date = last_actual_date + pd.DateOffset(months=(months_gap * 2) // 3 + 1)

                    mid1_value = last_actual_value + (value_diff * 0.4)  # 40% 지점
                    mid2_value = last_actual_value + (value_diff * 0.75)  # 75% 지점

                    # 4점 곡선 연결
                    ax.plot(
                        [last_actual_date, mid1_date, mid2_date, first_pred_date],
                        [last_actual_value, mid1_value, mid2_value, first_pred_value],
                        color="#666666",
                        linestyle="-",
                        linewidth=4,
                        alpha=0.8,
                        label="실제-예측 연결(매끄러운)" if idx == 0 else "",
                        zorder=3,
                    )

                    # 중간 포인트들 표시
                    for mid_date, mid_value in [
                        (mid1_date, mid1_value),
                        (mid2_date, mid2_value),
                    ]:
                        ax.plot(
                            mid_date,
                            mid_value,
                            "o",
                            color="#666666",
                            markersize=6,
                            alpha=0.6,
                            zorder=3,
                        )
                else:
                    # 간격이 길면 점진적 연결 (여러 중간점)
                    num_points = min(months_gap, 8)  # 최대 8개 중간점

                    connection_dates_list = []
                    connection_values_list = []

                    for i in range(1, num_points):
                        ratio = i / num_points
                        # 부드러운 전환을 위한 sigmoid 함수 적용
                        smooth_ratio = 1 / (1 + np.exp(-6 * (ratio - 0.5)))

                        conn_date = last_actual_date + pd.DateOffset(months=int(months_gap * ratio))
                        conn_value = last_actual_value + (value_diff * smooth_ratio)

                        connection_dates_list.append(conn_date)
                        connection_values_list.append(conn_value)

                    # 전체 연결선 그리기
                    full_dates = [last_actual_date] + connection_dates_list + [first_pred_date]
                    full_values = [last_actual_value] + connection_values_list + [first_pred_value]

                    ax.plot(
                        full_dates,
                        full_values,
                        color="#666666",
                        linestyle="-",
                        linewidth=4,
                        alpha=0.8,
                        label="실제-예측 연결(매끄러운)" if idx == 0 else "",
                        zorder=3,
                    )

                    # 중간 포인트들 표시 (일부만)
                    for i, (conn_date, conn_value) in enumerate(
                        zip(connection_dates_list, connection_values_list)
                    ):
                        if i % 2 == 0:  # 격개로 표시
                            ax.plot(
                                conn_date,
                                conn_value,
                                "o",
                                color="#666666",
                                markersize=4,
                                alpha=0.5,
                                zorder=3,
                            )

                # 🔥 연결 구간 하이라이트 (옅은 배경)
                ax.axvspan(
                    last_actual_date,
                    first_pred_date,
                    alpha=0.08,
                    color="orange",
                    label="실제-예측 전환구간" if idx == 0 else "",
                    zorder=1,
                )

                # 🔥 연결점 정보 표시
                if idx == 0:  # 첫 번째 그래프에만 표시
                    # 실제값 끝점 강조
                    ax.plot(
                        last_actual_date,
                        last_actual_value,
                        "o",
                        color="blue",
                        markersize=12,
                        markerfacecolor="lightblue",
                        markeredgewidth=3,
                        markeredgecolor="blue",
                        label="실제값 마지막",
                        zorder=4,
                    )

                    # 예측값 시작점 강조
                    ax.plot(
                        first_pred_date,
                        first_pred_value,
                        "s",
                        color="red",
                        markersize=12,
                        markerfacecolor="lightcoral",
                        markeredgewidth=3,
                        markeredgecolor="red",
                        label="예측값 시작",
                        zorder=4,
                    )

                    # 연결 정보 텍스트
                    mid_date_for_text = last_actual_date + pd.DateOffset(months=months_gap // 2)
                    mid_value_for_text = (last_actual_value + first_pred_value) / 2

                    change_pct = ((first_pred_value - last_actual_value) / last_actual_value) * 100
                    change_text = f"변화: {change_pct:+.1f}%"

                    ax.annotate(
                        change_text,
                        xy=(mid_date_for_text, mid_value_for_text),
                        xytext=(0, 20),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            alpha=0.9,
                            edgecolor="gray",
                            linewidth=1,
                        ),
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="arc3,rad=0.1",
                            color="gray",
                            alpha=0.7,
                            lw=1,
                        ),
                        zorder=5,
                    )

            # 예측 기간 예측값 (빨간색 실선)
            if pred_dates:
                ax.plot(
                    pred_dates,
                    pred_values,
                    color="red",
                    linestyle="-",
                    linewidth=4,
                    label="예측값(목표기간)" if idx == 0 else "",
                    alpha=0.9,
                    marker="s",
                    markersize=10,
                    zorder=4,
                )

                # 예측값 숫자 표시 (아름다운 라벨)
                for i, (date, value, month_label) in enumerate(
                    zip(pred_dates, pred_values, pred_labels)
                ):
                    if i % 2 == 0:
                        xytext = (0, 35)
                        va = "bottom"
                        facecolor = "yellow"
                    else:
                        xytext = (0, -45)
                        va = "top"
                        facecolor = "lightgreen"

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
                            facecolor=facecolor,
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
                    label="실제값(예측기간)" if idx == 0 else "",
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
                        label="실제-예측 연결" if idx == 0 else "",
                        zorder=2,
                    )

            # 예측 기간 하이라이트
            ax.axvspan(
                pred_start,
                pred_end,
                alpha=0.15,
                color="blue",
                label="예측 목표 기간" if idx == 0 else "",
                zorder=1,
            )

            # y축 범위 조정
            all_values = []
            if len(recent_data) > 0:
                all_values.extend(recent_data["입국자수"].tolist())
            if connection_values:
                all_values.extend(connection_values)
            if pred_values:
                all_values.extend(pred_values)

            if all_values:
                y_min = min(all_values) * 0.85
                y_max = max(all_values) * 1.2
                ax.set_ylim(y_min, y_max)

            # 목적별 규모 및 강조 표시
            avg_val = purpose_scales.get(purpose, 0)
            axis_info = "주요축" if purpose in left_purposes else "보조축"
            emphasis = " ★" if purpose == max_purpose else ""

            ax.set_title(
                f"{purpose}{emphasis} (평균: {avg_val:,.0f}명, {axis_info})",
                fontsize=16,
                fontweight="bold",
                color="darkblue" if purpose == max_purpose else "darkgray",
            )
            ax.set_ylabel("입국자수 (명)", fontsize=14)

            # 범례 (첫 번째에만)
            if idx == 0:
                ax.legend(fontsize=12, loc="upper left", frameon=True, fancybox=True, shadow=True)

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(axis="x", rotation=45, labelsize=12)
            ax.tick_params(axis="y", labelsize=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

            # 목적별 통계 정보 박스
            if pred_values:
                total_pred = sum(pred_values)
                avg_pred = total_pred / len(pred_values)

                # 주요 목적과 보조 목적 구분해서 박스 색상
                box_color = "lightblue" if purpose == max_purpose else "lightgray"

                ax.text(
                    0.02,
                    0.98,
                    f"예측 총합: {total_pred:,}명\n월평균: {avg_pred:,.0f}명",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, alpha=0.8),
                    fontsize=11,
                    fontweight="bold",
                )

            # �� 연결용 예측값 (회색 점선) - 참고용
            if connection_dates:
                ax.plot(
                    connection_dates,
                    connection_values,
                    color="#404040",
                    linestyle="--",
                    linewidth=4,
                    label="예측값(참고용)" if idx == 0 else "",
                    alpha=0.8,
                    marker="o",
                    markersize=8,
                    zorder=4,
                )

        # 전체 제목
        total_prediction = sum(
            sum(p["value"] for p in preds if p["month"] >= start_date) for preds in results.values()
        )

        fig.suptitle(
            f"{nationality} 목적별 입국자 예측 리포트\n"
            f"기간: {start_date} ~ {end_date} | 총 예측: {total_prediction:,}명 | 주요: {max_purpose}",
            fontsize=22,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.results_dir}/{nationality}_완전예측리포트_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"완전 예측 리포트 저장: {plot_path}")

        # CSV로도 저장 (상세 버전)
        csv_data = []
        months = sorted(set(pred["month"] for preds in results.values() for pred in preds))

        for month in months:
            row = {"월": month, "총합": 0}
            for purpose, predictions in results.items():
                value = next((p["value"] for p in predictions if p["month"] == month), 0)
                row[purpose] = value
                row["총합"] += value

            # 주요 목적 비율 계산
            if row["총합"] > 0 and max_purpose in row:
                row[f"{max_purpose}_비율"] = f"{(row[max_purpose] / row['총합'] * 100):.1f}%"

            csv_data.append(row)

        csv_df = pd.DataFrame(csv_data)
        csv_path = f"{self.results_dir}/{nationality}_완전예측리포트_{timestamp}.csv"
        csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"완전 예측 리포트 CSV 저장: {csv_path}")

        plt.show()

        # 콘솔 요약 (더 상세하게)
        print(f"\n" + "=" * 80)
        print(f"{nationality} 완전 예측 리포트 요약")
        print(f"=" * 80)
        print(f"주요 수요 목적: {max_purpose} (평균 {purpose_scales[max_purpose]:,.0f}명/월)")
        print(
            f"예측 기간: {start_date} ~ {end_date} ({len([m for m in months if m >= start_date])}개월)"
        )
        print(f"예측 목적 수: {len(purposes)}개")
        print(f"전체 예측 총합: {total_prediction:,}명")

        if total_prediction > 0:
            main_prediction = sum(
                p["value"] for p in results.get(max_purpose, []) if p["month"] >= start_date
            )
            main_ratio = (main_prediction / total_prediction * 100) if total_prediction > 0 else 0
            print(f"{max_purpose} 예측: {main_prediction:,}명 ({main_ratio:.1f}%)")

        print(
            f"월평균 예측: {total_prediction/max(1, len([m for m in months if m >= start_date])):,.0f}명"
        )
        print(f"저장 파일: {plot_path}")
        print(f"CSV 파일: {csv_path}")
        print(f"=" * 80)

    def predict(self, nationality, purpose=None, start_date="2025-07", end_date="2025-09"):
        """메인 예측 함수"""
        print(f"예측 시작: {nationality}")
        print(f"기간: {start_date} ~ {end_date}")

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

        # 목적 결정
        if purpose is None:
            # 전체 목적별 예측
            available_purposes = self.data[self.data["국적"] == nationality]["목적"].unique()
            results = {}

            for p in available_purposes:
                predictions = self.predict_future_months(nationality, p, target_months)
                if predictions:
                    results[p] = predictions

            if results:
                # 예측 결과 시각화 생성
                self.create_prediction_visualization(nationality, results, start_date, end_date)

                # 통합 리포트 생성
                self.save_comprehensive_report()
                
                # 🔥 학습 로그 리포트 생성
                self.save_training_logs_report()

            return results
        else:
            # 특정 목적 예측
            predictions = self.predict_future_months(nationality, purpose, target_months)

            if predictions:
                results = {purpose: predictions}

                # 예측 결과 시각화 생성
                self.create_prediction_visualization(nationality, results, start_date, end_date)

                # 통합 리포트 생성
                self.save_comprehensive_report()
                
                # 🔥 학습 로그 리포트 생성
                self.save_training_logs_report()
                
                return results

            return None


def main():
    """메인 실행 함수"""
    # GPU 설정
    gpu_available = setup_gpu()
    print(f"GPU 사용 가능: {'Yes' if gpu_available else 'No'}")

    print("유연한 입국자 예측 시스템 시작")
    print("=" * 60)

    # 🔥 코로나 데이터 처리 전략 선택
    print("\n📊 코로나 데이터 처리 전략을 선택하세요:")
    print("=" * 50)
    print("  1. exclude  - 🚫 코로나 데이터 완전 제외 (최고 성능, MAE 57% 개선)")
    print("  2. weighted - ⚖️  코로나 데이터 10% 가중치 적용 (기본값, 중간 성능)")
    print("  3. include  - 📊 모든 데이터 포함 (기존 방식)")
    print("=" * 50)

    while True:
        try:
            choice = input("선택하세요 (1-3, 기본값 2): ").strip()

            if not choice:  # 엔터만 누른 경우
                covid_strategy = "weighted"
                print("⚖️ 기본값 선택: weighted (10% 가중치)")
                break
            elif choice == "1":
                covid_strategy = "exclude"
                print("🚫 선택됨: exclude (코로나 데이터 완전 제외)")
                break
            elif choice == "2":
                covid_strategy = "weighted"
                print("⚖️ 선택됨: weighted (10% 가중치)")
                break
            elif choice == "3":
                covid_strategy = "include"
                print("📊 선택됨: include (모든 데이터 포함)")
                break
            else:
                print("❌ 잘못된 입력입니다. 1-3 중에서 선택하세요.")
                continue
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            return
        except Exception as e:
            print(f"입력 오류: {e}")
            continue

    print("=" * 60)

    # 예측기 초기화 (선택된 전략으로)
    predictor = FlexiblePredictor(covid_strategy=covid_strategy)

    # 데이터 로드 확인
    if predictor.data is None or len(predictor.data) == 0:
        print("❌ 데이터 로드 실패")
        return

    # 실제 데이터에서 국적 목록 가져오기
    nationalities = sorted(predictor.data["국적"].unique())

    print(f"\n총 {len(nationalities)}개 국적 사용 가능")

    # 메인 입력 루프
    while True:
        try:
            print("\n" + "=" * 60)
            print("새로운 예측 시작")
            print("=" * 60)

            # 1. 국적 입력
            nationality = predictor.safe_input_nationality(nationalities)
            if nationality is None:
                break

            # 2. 해당 국적의 목적 확인
            available_purposes = predictor.data[predictor.data["국적"] == nationality][
                "목적"
            ].unique()

            # 3. 목적 입력
            purpose = predictor.safe_input_purpose(nationality, available_purposes)
            if purpose is None and nationality is None:
                break

            # 4. 날짜 범위 입력
            start_date = predictor.safe_input_date("시작")
            if start_date is None:
                break

            end_date = predictor.safe_input_date("종료")
            if end_date is None:
                break

            # 5. 예측 실행
            print(f"\n{nationality} 예측 실행 중...")
            print(f"목적: {purpose if purpose else '전체 목적별'}")
            print(f"기간: {start_date} ~ {end_date}")

            result = predictor.predict(
                nationality=nationality, purpose=purpose, start_date=start_date, end_date=end_date
            )

            if result:
                print("예측 완료!")

                # 예측 결과 요약
                total_predicted = 0
                for purpose_name, predictions in result.items():
                    pred_values = [
                        p
                        for p in predictions
                        if p["type"] == "predicted" and p["month"] >= start_date
                    ]
                    if pred_values:
                        purpose_total = sum(p["value"] for p in pred_values)
                        total_predicted += purpose_total
                        print(f"   {purpose_name}: {purpose_total:,}명")

                print(f"   전체 예측 합계: {total_predicted:,}명")

                # 다시 예측할지 물어보기
                while True:
                    try:
                        continue_predict = (
                            input("\n다른 예측을 진행하시겠습니까? (y/n): ").lower().strip()
                        )
                        if continue_predict in ["y", "yes", "네", "ㅇ"]:
                            break
                        elif continue_predict in ["n", "no", "아니오", "ㄴ"]:
                            print("예측 시스템을 종료합니다.")
                            return
                        else:
                            print("'y' 또는 'n'으로 입력해주세요.")
                            continue
                    except KeyboardInterrupt:
                        print("\n프로그램을 종료합니다.")
                        return
            else:
                print("예측 실패")
                print("다시 시도해보세요.")
                continue

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"예상치 못한 오류가 발생했습니다: {e}")
            print("다시 시도해보세요.")
            continue


if __name__ == "__main__":
    main()