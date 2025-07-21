# config.py
# 이 파일은 예측 모델의 다양한 설정 값들을 한곳에 모아 관리합니다.
# 코드를 직접 수정하지 않고도 여기서 모델의 동작을 변경할 수 있습니다.

import os
from datetime import datetime

# --- 파일 경로 설정 ---
# 데이터 파일이 위치한 절대 경로를 지정합니다.
# 프로젝트의 'jin/data_preprocessing/data/processed' 폴더에 있는 CSV 파일을 사용합니다.
DATA_PATH = "/Volumes/DATA/mbc_project/1stProject/jin/data_preprocessing/data/processed/외국인입국자_전처리완료_딥러닝용.csv"

# 예측 결과가 저장될 기본 디렉토리입니다.
# 'results' 폴더 아래에 타임스탬프별로 폴더가 생성됩니다.
BASE_RESULTS_DIR = "/Volumes/DATA/mbc_project/1stProject/jin/modeling/results"

# --- 코로나 데이터 처리 전략 설정 ---
# 모델이 코로나19 팬데믹 기간의 데이터를 어떻게 처리할지 결정합니다.
# - "exclude": 코로나 기간 데이터를 완전히 제외합니다. (가장 높은 성능을 기대할 수 있습니다.)
# - "weighted": 코로나 기간 데이터에 낮은 가중치(10%)를 적용합니다. (기본값, 균형 잡힌 성능)
# - "include": 모든 데이터를 포함합니다. (기존 방식, 코로나 영향이 그대로 반영됩니다.)
DEFAULT_COVID_STRATEGY = "weighted"

# --- 목적별 최적 코로나 전략 설정 ---
# 목적별 코로나 영향도에 따른 최적화된 전략 (성능 향상 목적)
PURPOSE_OPTIMAL_COVID_STRATEGY = {
    "관광": "exclude",     # 관광: 코로나 영향 극심 → 제외가 최적
    "공용": "weighted",    # 공용: 정부 정책 → 가중치 적용
    "상용": "include",     # 상용: 비즈니스 연속성 → 포함
    "유학연수": "weighted" # 유학연수: 교육 정책 → 가중치 적용
}

# --- 성능 최적화 모드 설정 ---
# 모델 학습 시 하드웨어에 맞춰 성능을 최적화하는 방법을 지정합니다.
# - "auto": 시스템을 자동으로 감지하여 최적의 모드를 선택합니다. (M1/M2 Mac 자동 감지)
# - "m1_optimized": M1/M2 Mac에 특화된 최적화 설정을 적용합니다.
# - "standard": 일반적인 시스템에 적용되는 표준 설정을 사용합니다.
DEFAULT_PERFORMANCE_MODE = "auto"

# --- 모델 학습 하이퍼파라미터 설정 ---
# LSTM 모델의 학습 과정에 영향을 미치는 중요한 값들입니다.
# 데이터 크기에 따라 동적으로 조정될 수 있습니다.
LSTM_SEQUENCE_LENGTH_SMALL_DATA = 6   # 데이터가 적을 때 사용할 시퀀스 길이
LSTM_SEQUENCE_LENGTH_LARGE_DATA = 12  # 데이터가 많을 때 사용할 시퀀스 길이
LSTM_EPOCHS_SMALL_DATA = 50           # 데이터가 적을 때 학습할 에포크 수
LSTM_EPOCHS_LARGE_DATA = 100          # 데이터가 많을 때 학습할 에포크 수
LSTM_BATCH_SIZE = 16                  # 한 번에 처리할 데이터 샘플 수

# Early Stopping (조기 종료) 설정: 모델 성능이 더 이상 개선되지 않을 때 학습을 멈춥니다.
EARLY_STOPPING_PATIENCE = 10          # 성능 개선이 없을 때 몇 에포크를 더 기다릴지
EARLY_STOPPING_MONITOR = "val_loss"   # 어떤 지표를 모니터링할지 (검증 손실)

# ReduceLROnPlateau (학습률 감소) 설정: 성능 개선이 없을 때 학습률을 자동으로 줄입니다.
REDUCE_LR_FACTOR = 0.7
REDUCE_LR_PATIENCE = 10                # 성능 개선이 없을 때 몇 에포크를 더 기다릴지
REDUCE_LR_MIN_LR = 1e-6               # 학습률의 최소값

# --- 성능 평가 기준 설정 ---
# 모델의 예측 성능을 평가하는 기준값들입니다.
# 데이터의 특성(코로나 영향 등)에 따라 유연하게 조정됩니다.
BASE_PERFORMANCE_THRESHOLDS = {
    "mae_기준값": 1000,       # 평균 절대 오차 (낮을수록 좋음)
    "rmse_기준값": 1500,      # 제곱근 평균 제곱 오차 (낮을수록 좋음)
    "r2_score_기준값": 0.35,  # 결정 계수 (소규모 데이터 고려하여 상향 조정)
    "mape_기준값": 50.0,      # 평균 절대 백분율 오차 (낮을수록 좋음)
    "accuracy": 0.7,   # 정확도 (소규모 데이터 고려하여 하향 조정)
    "precision": 0.6,  # 정밀도 (소규모 데이터 고려하여 하향 조정)
    "recall": 0.6,     # 재현율 (소규모 데이터 고려하여 하향 조정)
    "f1_score_기준값": 0.3,   # F1 점수 (실제 달성 가능한 수준으로 조정)
    "fbeta_score": 0.7, # F-beta 점수 (높을수록 좋음)
    "roc_auc": 0.75,   # ROC AUC (높을수록 좋음)
}

# --- 목적별 차별화된 성능 기준 설정 ---
# 목적별 데이터 특성과 예측 난이도에 맞춘 차별화된 성능 기준값
PURPOSE_SPECIFIC_THRESHOLDS = {
    "관광": {
        "r2_score_기준값": 0.5,    # 관광: 패턴이 명확하여 높은 기준
        "f1_score_기준값": 0.4,    # 관광: 계절성으로 예측 정확도 높음
        "mape_기준값": 40.0        # 관광: 변동성이 있지만 예측 가능
    },
    "공용": {
        "r2_score_기준값": 0.25,   # 공용: 정책 변화로 예측 어려움
        "f1_score_기준값": 0.2,    # 공용: 낮은 기준 적용
        "mape_기준값": 60.0        # 공용: 높은 불확실성 허용
    },
    "상용": {
        "r2_score_기준값": 0.4,    # 상용: 비즈니스 패턴으로 중간 기준
        "f1_score_기준값": 0.35,   # 상용: 경기 사이클 반영
        "mape_기준값": 45.0        # 상용: 적당한 변동성 허용
    },
    "유학연수": {
        "r2_score_기준값": 0.3,    # 유학연수: 교육 정책 변화로 중하 기준
        "f1_score_기준값": 0.25,   # 유학연수: 학기별 패턴 고려
        "mape_기준값": 55.0        # 유학연수: 정책 변화 허용
    }
}

# --- F1-score 계산을 위한 허용 오차율 설정 ---
# 예측값과 실제값의 상대 오차(백분율)가 이 값 이내일 경우 '정답'으로 간주합니다.
F1_SCORE_TOLERANCE_PERCENTAGE = 10.0 # 예: 10.0은 10% 오차율 허용

# --- 데이터 증강 설정 ---
# 데이터가 부족할 때 인공적으로 데이터를 늘리는 방법입니다.
AUGMENTATION_TARGET_MONTHS = 200 # 데이터 증강을 통해 목표로 하는 월별 데이터 수 (16.7년, 현실적으로 조정)
AUGMENTATION_NOISE_LEVELS = [0.15, 0.25, 0.35] # 노이즈 증강 시 적용할 노이즈 수준
AUGMENTATION_TREND_FACTORS = [0.02, 0.05, -0.02] # 트렌드 증강 시 적용할 트렌드 요인
AUGMENTATION_SEASONAL_BOOSTS = [1.3, 1.5, 0.7] # 계절성 강화 증강 시 적용할 부스트 요인

# --- 목적별 차별화된 데이터 증강 설정 ---
# 목적별로 다른 증강 전략을 적용하여 더 나은 성능을 달성합니다.
PURPOSE_SPECIFIC_AUGMENTATION = {
    "관광": 180,      # 관광: 충분한 데이터, 적당한 증강
    "공용": 250,      # 공용: 부족한 데이터, 많은 증강
    "상용": 220,      # 상용: 중간 정도 증강
    "유학연수": 240   # 유학연수: 많은 증강 필요
}

# --- 코로나 기간 정의 ---
# 데이터에서 코로나 기간을 식별하는 데 사용됩니다.
COVID_START_DATE = "2020-03-01"
COVID_END_DATE = "2022-10-31"

# --- 시각화 설정 ---
# 그래프 생성 시 사용되는 설정입니다.
PLOT_DPI = 300 # 이미지 해상도
PLOT_FACECOLOR = "white" # 그래프 배경색

# --- 로깅 설정 ---
# TensorFlow의 메시지 출력 레벨을 조정합니다.
# '0'은 모든 메시지 출력, '1'은 INFO 숨김, '2'는 INFO와 WARNING 숨김, '3'은 모든 메시지 숨김
TF_CPP_MIN_LOG_LEVEL = '2'

# --- M1 Mac 폰트 설정 ---
# M1/M2 Mac에서 한글 폰트가 깨지지 않도록 설정합니다.
M1_FONT_FAMILY = "AppleGothic"

# --- 기타 설정 ---
# 앙상블 모델 사용 여부 (현재는 사용하지 않음)
TOURISM_ENSEMBLE_AVAILABLE = False

# --- 관광 목적 예측 특화 설정 ---
# 관광 목적 데이터는 특수한 패턴을 가지므로, 별도의 예측 설정을 사용합니다.
TOURISM_SEQUENCE_LENGTH = 9 # 관광 모델에 사용할 시퀀스 길이 (일반 모델보다 길 수 있음)

# --- 목적별 차별화된 시퀀스 길이 설정 ---
# 목적별 데이터 패턴에 맞춘 최적 시퀀스 길이로 성능을 향상시킵니다.
PURPOSE_SPECIFIC_SEQUENCE_LENGTH = {
    "관광": 18,        # 관광: 계절성 1.5년 주기 (강한 계절 패턴)
    "공용": 24,        # 공용: 정책 변화 2년 주기 (장기 정책 패턴)
    "상용": 15,        # 상용: 비즈니스 사이클 1.25년 (경기 순환)
    "유학연수": 21     # 유학연수: 학기 1.75년 주기 (학기별 패턴)
}

# 예측값의 급격한 변화를 제어하기 위한 최대 변화율 (전월 대비)
TOURISM_MAX_CHANGE_RATE_INITIAL = 0.20 # 초기 3개월 예측에 적용될 최대 변화율 (더 엄격)
TOURISM_MAX_CHANGE_RATE_MEDIUM = 0.25 # 중간 3개월 예측에 적용될 최대 변화율
TOURISM_MAX_CHANGE_RATE_LONG = 0.30 # 후반 3개월 예측에 적용될 최대 변화율
TOURISM_MAX_CHANGE_RATE_VERY_LONG = 0.35 # 장기 예측에 적용될 최대 변화율 (불확실성 반영)

# 예측값의 최소값 보장 (전월 대비 비율)
TOURISM_MIN_VALUE_RATIO = 0.15 # 예측값이 이 비율 이하로 떨어지지 않도록 보장 (예: 전월의 15%)
TOURISM_MIN_VALUE_PREV_MONTH_RATIO = 0.80 # 전월 대비 급격한 감소 방지 (예: 전월의 80% 이하로 떨어지지 않음)

# --- 관광 목적 모델 학습 하이퍼파라미터 설정 ---
# 관광 목적 모델은 일반 모델과 다른 학습 전략을 가질 수 있습니다.
TOURISM_LSTM_EPOCHS_SMALL_DATA = 75 # 관광 데이터가 적을 때 학습할 에포크 수 (일반보다 많음)
TOURISM_LSTM_EPOCHS_LARGE_DATA = 150 # 관광 데이터가 많을 때 학습할 에포크 수 (일반보다 많음)
TOURISM_EARLY_STOPPING_PATIENCE = 15 # 관광 모델의 조기 종료 대기 에포크 수 (일반보다 김)
TOURISM_REDUCE_LR_FACTOR = 0.5 # 관광 모델의 학습률 감소 비율
TOURISM_REDUCE_LR_PATIENCE = 8 # 관광 모델의 학습률 감소 대기 에포크 수
TOURISM_REDUCE_LR_MIN_LR = 1e-8 # 관광 모델의 최소 학습률
TOURISM_LSTM_BATCH_SIZE_SMALL_DATA = 8 # 관광 데이터가 적을 때 배치 크기
TOURISM_LSTM_BATCH_SIZE_LARGE_DATA = 32 # 관광 데이터가 많을 때 배치 크기

# --- 관광 목적 모델 학습률 설정 ---
TOURISM_LEARNING_RATE_SMALL_DATA = 0.002
TOURISM_LEARNING_RATE_MEDIUM_DATA = 0.0015
TOURISM_LEARNING_RATE_LARGE_DATA = 0.001

# --- 자동 실행 모드 기본값 설정 ---
DEFAULT_NATIONALITY = "중국"
DEFAULT_START_DATE = "2026-01"
DEFAULT_END_DATE = "2026-12"

# --- 동적 설정 ---
# AVAILABLE_NATIONALITIES는 프로그램 시작 시 DataHandler에 의해 동적으로 채워집니다.
AVAILABLE_NATIONALITIES = []

# --- 앙상블 시스템 설정 (1단계 개선) ---
# 앙상블 모드 활성화 여부 (단계별 활성화를 위한 플래그)
ENABLE_ENSEMBLE = True

# --- XGBoost 모델 설정 (2단계 개선) ---
# XGBoost 모델 활성화 여부
ENABLE_XGBOOST = True

# --- 동적 최적화 시스템 설정 (3단계 개선) ---
# 동적 모델 선택 및 하이퍼파라미터 튜닝 활성화
ENABLE_SMART_OPTIMIZATION = True

# 데이터 특성 분석 기준값들
DATA_ANALYSIS_THRESHOLDS = {
    "small_data": 100,      # 작은 데이터셋 기준
    "medium_data": 200,     # 중간 데이터셋 기준  
    "large_data": 300,      # 큰 데이터셋 기준
    "high_volatility": 0.3, # 높은 변동성 기준
    "strong_seasonality": 0.15, # 강한 계절성 기준
    "stable_trend": 0.05    # 안정적 트렌드 기준
}

# 데이터 특성별 최적 모델 매핑
OPTIMAL_MODEL_BY_CHARACTERISTICS = {
    # 데이터 크기 + 변동성 + 계절성 조합별 최적 모델
    "small_high_volatile": ["XGBOOST", "DENSE"],
    "small_seasonal": ["LSTM_ATTENTION", "GRU"], 
    "small_stable": ["DENSE", "GRU"],
    "medium_high_volatile": ["XGBOOST", "GRU", "LSTM"],
    "medium_seasonal": ["LSTM_ATTENTION", "LSTM", "XGBOOST"],
    "medium_stable": ["LSTM", "GRU", "XGBOOST"],
    "large_high_volatile": ["LSTM_ATTENTION", "XGBOOST", "LSTM"],
    "large_seasonal": ["LSTM_ATTENTION", "LSTM", "GRU"],
    "large_stable": ["LSTM", "GRU", "XGBOOST", "LSTM_ATTENTION"]
}

# 동적 하이퍼파라미터 튜닝 범위
HYPERPARAMETER_TUNING_RANGES = {
    "lstm_units": [32, 64, 128],
    "learning_rate": [0.001, 0.005, 0.01],
    "dropout": [0.2, 0.3, 0.4],
    "batch_size": [16, 32, 64],
    "xgb_n_estimators": [100, 150, 200],
    "xgb_max_depth": [3, 4, 5, 6],
    "xgb_learning_rate": [0.05, 0.1, 0.15]
}

# XGBoost 하이퍼파라미터 설정 (목적별 최적화)
XGBOOST_PARAMS = {
    "관광": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    },
    "공용": {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1
    },
    "상용": {
        "n_estimators": 180,
        "max_depth": 5,
        "learning_rate": 0.12,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1
    },
    "유학연수": {
        "n_estimators": 160,
        "max_depth": 4,
        "learning_rate": 0.15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1
    }
}

# 앙상블에 사용할 모델들의 조합 설정 (XGBoost 포함 - 2단계 개선)
ENSEMBLE_MODELS = {
    "관광": {
        "LSTM_ATTENTION": 0.3,  # 관광은 복잡한 패턴 → Attention 모델 비중 높임
        "XGBOOST": 0.25,        # XGBoost: 비선형 패턴 학습
        "LSTM": 0.25,
        "GRU": 0.15,
        "DENSE": 0.05
    },
    "공용": {
        "XGBOOST": 0.3,         # 공용: XGBoost로 정책 변화 패턴 학습
        "LSTM": 0.25,
        "GRU": 0.25,
        "LSTM_ATTENTION": 0.15,
        "DENSE": 0.05
    },
    "상용": {
        "XGBOOST": 0.35,        # 상용: XGBoost로 경제 지표 패턴 학습
        "GRU": 0.3,
        "LSTM": 0.2,
        "DENSE": 0.1,
        "LSTM_ATTENTION": 0.05
    },
    "유학연수": {
        "LSTM": 0.3,            # 유학연수: 장기 패턴 LSTM + XGBoost 조합
        "XGBOOST": 0.25,
        "GRU": 0.25,
        "LSTM_ATTENTION": 0.15,
        "DENSE": 0.05
    }
}

# 앙상블 예측 후처리 설정 (자연스러운 변동을 위한 가중치)
ENSEMBLE_SMOOTHING_WEIGHTS = {
    "관광": 0.7,     # 관광: 변동성 유지를 위해 스무딩 적게
    "공용": 0.8,     # 공용: 안정성 중시
    "상용": 0.75,    # 상용: 중간 수준
    "유학연수": 0.8  # 유학연수: 안정성 중시
}


