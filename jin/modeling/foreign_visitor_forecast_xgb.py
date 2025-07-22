import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split
import platform

# Mac에서 한글 폰트 설정
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 결과 저장 경로 설정
RESULT_DIR = "./results_xgb"
os.makedirs(RESULT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder = os.path.join(RESULT_DIR, f"forecast_{timestamp}")
os.makedirs(run_folder, exist_ok=True)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, rmse, mape


def get_reliability(mape):
    if mape <= 20:
        return "매우 높음"
    elif mape <= 35:
        return "높음"
    elif mape <= 50:
        return "보통 (주의 필요)"
    else:
        return "낮음 (신뢰 어려움)"


def get_deployment_advice(mape):
    if mape <= 35:
        return "실무 적용 가능성 높음"
    elif mape <= 50:
        return "실무 적용에 보완 필요"
    else:
        return "실무 적용 어려움"


def load_and_prepare_data(file_path, nation, purpose):
    df = pd.read_csv(file_path)
    df = df[(df["국적"] == nation) & (df["목적"] == purpose)].copy()
    if df.empty:
        print(
            f"[오류] '{nation}' 국적과 '{purpose}' 목적에 해당하는 데이터가 없습니다."
        )
        exit(1)
    df["연월"] = pd.to_datetime(
        df["연도"].astype(str) + "-" + df["월"].astype(str).str.zfill(2)
    )
    df["월"] = df["연월"].dt.month
    df["년"] = df["연월"].dt.year
    df["월_index"] = (df["연월"] - df["연월"].min()).dt.days // 30
    return df


def forecast_xgb(df, predict_start, predict_end):
    X = df[["월_index", "월", "계절", "코로나기간"]]
    y = df["입국자수"]

    # 모델 훈련
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)

    # 미래 예측용 월 인덱스 생성
    last_date = df["연월"].max()
    future_months = pd.date_range(start=predict_start, end=predict_end, freq="MS")
    future_df = pd.DataFrame({"연월": future_months})
    future_df["월"] = future_df["연월"].dt.month
    future_df["월_index"] = (future_df["연월"] - df["연월"].min()).dt.days // 30
    future_df["계절"] = future_df["월"].apply(lambda m: (m % 12 + 3) // 3)
    future_df["코로나기간"] = 0  # 기본 0 처리 (추후 필요 시 조정)

    # 예측
    future_X = future_df[["월_index", "월", "계절", "코로나기간"]]
    preds = model.predict(future_X)
    future_df["예측입국자수"] = preds.astype(int)

    return model, future_df


def visualize(df, future_df, nation, purpose):
    plt.figure(figsize=(10, 5))
    plt.plot(df["연월"], df["입국자수"], label="실제입국자수", color="gray")
    plt.plot(
        future_df["연월"],
        future_df["예측입국자수"],
        label="예측입국자수",
        color="red",
        linestyle="--",
    )
    plt.axvspan(
        future_df["연월"].min(), future_df["연월"].max(), color="yellow", alpha=0.2
    )
    plt.title(f"[{nation}] {purpose} 입국자 수 예측 (XGBoost)")
    plt.xlabel("연월")
    plt.ylabel("입국자수")
    plt.legend()
    plt.tight_layout()
    file_name = os.path.join(run_folder, f"{nation}_{purpose}_예측결과.png")
    plt.savefig(file_name)
    plt.close()


def main():
    BASE_DIR = "../data_preprocessing/data/processed/"
    file_path = BASE_DIR + "/외국인입국자_전처리완료_딥러닝용.csv"
    nation = input("국적을 입력하세요 (예: 일본): ").strip()
    purpose_input = input("목적을 입력하세요 (전체일 경우 엔터): ").strip()
    predict_start = input("예측 시작월을 입력하세요 (예: 2026-01): ").strip()
    predict_end = input("예측 종료월을 입력하세요 (예: 2026-12): ").strip()

    df_all = pd.read_csv(file_path)
    purposes = (
        [purpose_input]
        if purpose_input
        else df_all[df_all["국적"] == nation]["목적"].unique()
    )

    for purpose in purposes:
        df = load_and_prepare_data(file_path, nation, purpose)
        model, future_df = forecast_xgb(df, predict_start, predict_end)
        mae, rmse, mape = evaluate(
            df["입국자수"], model.predict(df[["월_index", "월", "계절", "코로나기간"]])
        )

        print(
            f"[검증 결과] {purpose} — MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}% → 신뢰도: {get_reliability(mape)}"
        )
        print(
            f"[실무 적용 판단] '{purpose}' 목적 예측은 {get_deployment_advice(mape)} (MAPE={mape:.2f}%)"
        )

        future_df.to_csv(
            os.path.join(run_folder, f"{nation}_{purpose}_예측결과.csv"), index=False
        )
        visualize(df, future_df, nation, purpose)
    print(f"[완료] 예측 결과가 {run_folder} 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
