# -*- coding: utf-8 -*-
"""
모델링 설정 파일
Author: Jin
Created: 2025-01-15
"""

import os
from pathlib import Path

# 프로젝트 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data_preprocessing" / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# 데이터 파일 경로
DATA_FILE = DATA_DIR / "외국인입국자_전처리완료_딥러닝용.csv"

# 모델 저장 경로
LSTM_MODEL_DIR = MODEL_DIR / "lstm"
PROPHET_MODEL_DIR = MODEL_DIR / "prophet"
BEST_MODEL_DIR = MODEL_DIR / "best_model"

# 결과 저장 경로
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

# 디렉토리 생성
for directory in [
    LSTM_MODEL_DIR,
    PROPHET_MODEL_DIR,
    BEST_MODEL_DIR,
    PREDICTIONS_DIR,
    EVALUATION_DIR,
    PLOTS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# 데이터 설정
TARGET_COLUMN = "입국자수"
FEATURE_COLUMNS = [
    "연도",
    "월",
    "분기",
    "계절",
    "코로나기간",
    "시계열순서",
    "입국자수_1개월전",
    "입국자수_3개월전",
    "입국자수_12개월전",
    "입국자수_3개월평균",
    "입국자수_12개월평균",
    "전년동월대비증감률",
]

# 카테고리 컬럼
CATEGORY_COLUMNS = ["국적", "목적", "계절"]

# 시계열 설정
SEQUENCE_LENGTH = 6  # 6개월 시퀀스
FORECAST_HORIZON = 6  # 6개월 예측

# 데이터 분할 설정
TRAIN_END_YEAR = 2021
VAL_END_YEAR = 2022
TEST_START_YEAR = 2023

# LSTM 모델 하이퍼파라미터
LSTM_CONFIG = {
    "sequence_length": SEQUENCE_LENGTH,
    "n_features": len(FEATURE_COLUMNS),
    "lstm_units": [128, 64],
    "dense_units": [32, 16],
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "patience": 10,
    "validation_split": 0.2,
}

# Prophet 모델 설정
PROPHET_CONFIG = {
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "growth": "linear",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
}

# 평가 지표
EVALUATION_METRICS = ["mae", "mse", "rmse", "mape", "r2"]

# 로깅 설정
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 랜덤 시드 설정
RANDOM_SEED = 42

# 시각화 설정
PLOT_CONFIG = {"figsize": (12, 8), "dpi": 300, "style": "seaborn-v0_8", "color_palette": "husl"}

# 모델 선택 기준
MODEL_SELECTION_METRIC = "mape"  # 낮을수록 좋음

print("✅ 설정 파일 로드 완료")
print(f"📊 데이터 파일: {DATA_FILE}")
print(f"📁 모델 저장 경로: {MODEL_DIR}")
print(f"📈 결과 저장 경로: {RESULTS_DIR}")
