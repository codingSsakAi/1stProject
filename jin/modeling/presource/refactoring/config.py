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
BASE_RESULTS_DIR = "results"

# --- 코로나 데이터 처리 전략 설정 ---
# 모델이 코로나19 팬데믹 기간의 데이터를 어떻게 처리할지 결정합니다.
# - "exclude": 코로나 기간 데이터를 완전히 제외합니다. (가장 높은 성능을 기대할 수 있습니다.)
# - "weighted": 코로나 기간 데이터에 낮은 가중치(10%)를 적용합니다. (기본값, 균형 잡힌 성능)
# - "include": 모든 데이터를 포함합니다. (기존 방식, 코로나 영향이 그대로 반영됩니다.)
DEFAULT_COVID_STRATEGY = "weighted"

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
    "mae": 1000,       # 평균 절대 오차 (낮을수록 좋음)
    "rmse": 1500,      # 제곱근 평균 제곱 오차 (낮을수록 좋음)
    "r2_score": 0.2,   # 결정 계수 (높을수록 좋음, 0~1)
    "mape": 50.0,      # 평균 절대 백분율 오차 (낮을수록 좋음)
    "accuracy": 0.75,  # 정확도 (높을수록 좋음)
    "precision": 0.7,  # 정밀도 (높을수록 좋음)
    "recall": 0.7,     # 재현율 (높을수록 좋음)
    "f1_score": 0.45,  # F1 점수 (높을수록 좋음)
    "fbeta_score": 0.7, # F-beta 점수 (높을수록 좋음)
    "roc_auc": 0.75,   # ROC AUC (높을수록 좋음)
}

# --- 데이터 증강 설정 ---
# 데이터가 부족할 때 인공적으로 데이터를 늘리는 방법입니다.
AUGMENTATION_TARGET_MONTHS = 240 # 데이터 증강을 통해 목표로 하는 월별 데이터 수
AUGMENTATION_NOISE_LEVELS = [0.15, 0.25, 0.35] # 노이즈 증강 시 적용할 노이즈 수준
AUGMENTATION_TREND_FACTORS = [0.02, 0.05, -0.02] # 트렌드 증강 시 적용할 트렌드 요인
AUGMENTATION_SEASONAL_BOOSTS = [1.3, 1.5, 0.7] # 계절성 강화 증강 시 적용할 부스트 요인

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


