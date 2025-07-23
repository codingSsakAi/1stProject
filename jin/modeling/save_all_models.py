import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 기존 예측 함수 import
from foreign_visitor_forecast_tf import (
    select_best_model_and_predict,
    save_model_and_scaler,
    FEATURE_COLS,
    SUPPORTED_MODELS,
)

# --- 파라미터 설정 (필요시 수정) ---
CSV_PATH = "../data_preprocessing/data/processed/외국인입국자_전처리완료_딥러닝용.csv"
SAVE_DIR = "../flask/models"
WINDOW = 12
EPOCHS = 50
MODEL_TYPES = SUPPORTED_MODELS  # ['lstm', 'xgboost']
N_MONTHS = 12  # 예측 개월 수
MIN_MONTHS = 6  # 최소 허용 개월 수


def get_predict_period(df, n_months=12):
    """가장 최근 데이터 이후 n개월 예측 기간 반환"""
    last_date = df["날짜"].max()
    from dateutil.relativedelta import relativedelta

    predict_start = (last_date + relativedelta(months=1)).strftime("%Y-%m")
    predict_end = (last_date + relativedelta(months=n_months)).strftime("%Y-%m")
    return predict_start, predict_end


def get_unique_combinations(csv_path):
    """CSV에서 고유한 국적-목적 조합 추출"""
    df = pd.read_csv(csv_path)
    if "연도" in df.columns and "월" in df.columns:
        df["날짜"] = pd.to_datetime(
            df[["연도", "월"]]
            .rename(columns={"연도": "year", "월": "month"})
            .assign(day=1)
        )
    combinations = df[["국적", "목적"]].drop_duplicates()
    valid_combinations = []
    for _, row in combinations.iterrows():
        subset = df[(df["국적"] == row["국적"]) & (df["목적"] == row["목적"])]
        if len(subset) >= MIN_MONTHS:
            valid_combinations.append((row["국적"], row["목적"]))
    return valid_combinations


def train_and_save_all_models():
    """모든 국적-목적 조합에 대해 모델 학습 및 저장 (최적/블렌딩 자동)"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("📊 데이터 로딩 중...")
    df = pd.read_csv(CSV_PATH)
    if "연도" in df.columns and "월" in df.columns:
        df["날짜"] = pd.to_datetime(
            df[["연도", "월"]]
            .rename(columns={"연도": "year", "월": "month"})
            .assign(day=1)
        )
    # 예측 기간 자동 설정
    predict_start, predict_end = get_predict_period(df, N_MONTHS)
    print(f"⏳ 예측 기간: {predict_start} ~ {predict_end}")
    combinations = get_unique_combinations(CSV_PATH)
    print(f"🎯 총 {len(combinations)}개의 국적-목적 조합 발견")
    success_count = 0
    failed_combinations = []
    for i, (국적, 목적) in enumerate(combinations, 1):
        print(f"\n🔧 [{i}/{len(combinations)}] {국적} - {목적} 모델 학습 중...")
        try:
            subset = df[(df["국적"] == 국적) & (df["목적"] == 목적)].copy()
            n_months = len(subset)
            if n_months < MIN_MONTHS:
                print(f"❌ 데이터 부족: {n_months}개월 (최소 {MIN_MONTHS}개월 필요)")
                failed_combinations.append((국적, 목적, "데이터 부족"))
                continue
            # window, 모델 타입 자동 조정
            auto_window = max(3, min(WINDOW, n_months // 2))
            auto_model_types = MODEL_TYPES
            # linear 관련 분기 완전 삭제
            # 최적/블렌딩 모델 예측
            future_dates, pred_values, label, hist_df, metrics = (
                select_best_model_and_predict(
                    subset,
                    목적,
                    predict_start,
                    predict_end,
                    window=auto_window,
                    epochs=EPOCHS,
                    model_types=auto_model_types,
                    verbose=True,
                )
            )
            if future_dates is not None and pred_values is not None:
                success_count += 1
                print(f"✅ {국적} - {목적} 모델 학습/저장 완료")
            else:
                failed_combinations.append((국적, 목적, "모델 학습 실패"))
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            failed_combinations.append((국적, 목적, str(e)))
    print(f"\n📈 저장 완료 요약:")
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {len(failed_combinations)}개")
    if failed_combinations:
        print(f"\n❌ 실패한 조합들:")
        for 국적, 목적, 이유 in failed_combinations:
            print(f"  - {국적} - {목적}: {이유}")
    return success_count, failed_combinations


if __name__ == "__main__":
    print("🚀 모든 모델 일괄 저장 시작...")
    success_count, failed_combinations = train_and_save_all_models()
    print(f"\n🎉 작업 완료! 성공: {success_count}개 모델 저장됨")
