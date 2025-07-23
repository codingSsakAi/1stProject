#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 저장 기능 테스트 스크립트 (고도화)
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from foreign_visitor_forecast_tf import (
    select_best_model_and_predict,
    save_model_and_scaler,
    SUPPORTED_MODELS,
)

MIN_MONTHS = 6  # 최소 허용 개월 수
N_MONTHS = 12  # 예측 개월 수


def get_predict_period(df, n_months=12):
    last_date = df["날짜"].max()
    from dateutil.relativedelta import relativedelta

    predict_start = (last_date + relativedelta(months=1)).strftime("%Y-%m")
    predict_end = (last_date + relativedelta(months=n_months)).strftime("%Y-%m")
    return predict_start, predict_end


def test_single_model_save():
    """단일 모델 저장 테스트 (최적/블렌딩 자동, 데이터 부족 시 window/모델 자동 조정)"""
    print("🧪 단일 모델 저장 테스트 시작...")
    csv_path = (
        "../data_preprocessing/data/processed/외국인입국자_전처리완료_딥러닝용.csv"
    )
    window = 12
    epochs = 10  # 빠른 테스트를 위해 epochs 줄임
    model_types = SUPPORTED_MODELS
    df = pd.read_csv(csv_path)
    if "연도" in df.columns and "월" in df.columns:
        df["날짜"] = pd.to_datetime(
            df[["연도", "월"]]
            .rename(columns={"연도": "year", "월": "month"})
            .assign(day=1)
        )
    predict_start, predict_end = get_predict_period(df, N_MONTHS)
    unique_combinations = df[["국적", "목적"]].drop_duplicates()
    for _, row in unique_combinations.iterrows():
        국적, 목적 = row["국적"], row["목적"]
        subset = df[(df["국적"] == 국적) & (df["목적"] == 목적)]
        n_months = len(subset)
        if n_months < MIN_MONTHS:
            print(f"❌ 데이터 부족: {n_months}개월 (최소 {MIN_MONTHS}개월 필요)")
            continue
        # window, 모델 타입 자동 조정
        auto_window = max(3, min(window, n_months // 2))
        auto_model_types = model_types
        if n_months < 12:
            auto_model_types = ["linear"]
        print(f"🎯 테스트 대상: {국적} - {목적} ({n_months}개월 데이터)")
        try:
            future_dates, pred_values, label, hist_df, metrics = (
                select_best_model_and_predict(
                    subset,
                    목적,
                    predict_start,
                    predict_end,
                    window=auto_window,
                    epochs=epochs,
                    model_types=auto_model_types,
                    verbose=True,
                )
            )
            if future_dates is not None and pred_values is not None:
                print(f"✅ 테스트 성공: {국적} - {목적}")
                print(f"   - 예측 기간: {len(future_dates)}개월")
                print(f"   - 평가 지표: {metrics}")
                return True
            else:
                print(f"❌ 테스트 실패: {국적} - {목적} (모델 학습 실패)")
        except Exception as e:
            print(f"❌ 테스트 오류: {국적} - {목적} - {e}")
    print("❌ 유효한 테스트 데이터를 찾을 수 없습니다.")
    return False


def test_save_all_models():
    """전체 모델 저장 테스트 (최적/블렌딩 자동)"""
    print("\n🚀 전체 모델 저장 테스트 시작...")
    try:
        from save_all_models import train_and_save_all_models

        success_count, failed_combinations = train_and_save_all_models()
        print(f"\n📊 테스트 결과:")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {len(failed_combinations)}개")
        if failed_combinations:
            print(f"\n❌ 실패한 조합들:")
            for 국적, 목적, 이유 in failed_combinations:
                print(f"  - {국적} - {목적}: {이유}")
        return success_count > 0
    except Exception as e:
        print(f"❌ 전체 모델 저장 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    print("🧪 모델 저장 기능 테스트 시작...")
    single_test_result = test_single_model_save()
    if single_test_result:
        print("\n" + "=" * 50)
        print(
            "전체 모델 저장 테스트를 실행하시겠습니까? (시간이 오래 걸릴 수 있습니다)"
        )
        response = input("실행하려면 'y'를 입력하세요: ").strip().lower()
        if response == "y":
            all_test_result = test_save_all_models()
            if all_test_result:
                print("🎉 모든 테스트 성공!")
            else:
                print("⚠️ 전체 모델 저장 테스트에 문제가 있습니다.")
        else:
            print("전체 모델 저장 테스트를 건너뜁니다.")
    else:
        print("❌ 단일 모델 테스트 실패로 전체 테스트를 건너뜁니다.")
