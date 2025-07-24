import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import platform
import os
from prophet import Prophet
import math

# Mac에서 한글 폰트 설정
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 사용할 feature 컬럼 리스트 (입국자수만 사용)
FEATURE_COLS = ["입국자수"]


# Prophet 예측 함수
def predict_with_prophet(df, start_month, end_month):
    """
    Prophet 기반 월별 예측 함수
    - df: '날짜', '입국자수' 컬럼 포함 DataFrame (특정 국적/목적만 필터링)
    - start_month, end_month: 예측 시작/종료 (YYYY-MM)
    """
    df_prophet = (
        df[["날짜", "입국자수"]].rename(columns={"날짜": "ds", "입국자수": "y"}).copy()
    )
    df_prophet = df_prophet.groupby("ds")["y"].sum().reset_index()
    model = Prophet(yearly_seasonality=True)
    model.fit(df_prophet)
    start_date = pd.to_datetime(start_month)
    end_date = pd.to_datetime(end_month)
    n_months = (
        (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    )
    last_date = df_prophet["ds"].max()
    periods = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)
    future = model.make_future_dataframe(periods=max(periods, n_months), freq="MS")
    forecast = model.predict(future)
    pred_mask = (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)
    pred_dates = forecast.loc[pred_mask, "ds"].values
    pred_values = forecast.loc[pred_mask, "yhat"].values
    return pred_dates, pred_values


# 메인 함수: 데이터 로딩 → 예측 → 시각화 → 결과 저장
def forecast_visitors_prophet_only(csv_path, 국적, 목적, 시작월, 종료월):
    # 결과 저장 폴더 생성
    RESULT_DIR = "./results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(RESULT_DIR, f"forecast_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    # CSV 파일에서 데이터 로딩 및 날짜 컬럼 생성
    df = pd.read_csv(csv_path)
    df["날짜"] = pd.to_datetime(
        df[["연도", "월"]].rename(columns={"연도": "year", "월": "month"}).assign(day=1)
    )

    목적순서 = ["관광", "유학연수", "공용", "상용"]
    colors = {
        "관광": "tab:blue",
        "유학연수": "tab:green",
        "공용": "tab:orange",
        "상용": "tab:red",
    }

    # 예측할 목적별 데이터 분리 (strip 적용)
    df["국적"] = df["국적"].astype(str).str.strip()
    df["목적"] = df["목적"].astype(str).str.strip()
    if 목적:
        targets = [
            (
                목적.strip(),
                df[(df["국적"] == 국적.strip()) & (df["목적"] == 목적.strip())],
            )
        ]
    else:
        targets = [
            (m, df[(df["국적"] == 국적.strip()) & (df["목적"] == m.strip())])
            for m in 목적순서
        ]

    # 2x2 subplot에 목적별 Prophet 예측 결과를 모두 그림
    n = len(targets)
    ncols = 2
    nrows = 2 if n > 2 else 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axs = axs.flatten() if n > 1 else [axs]
    plot_idx = 0
    for i, (m, d) in enumerate(targets):
        if len(d) == 0:
            continue
        prophet_dates, prophet_preds = predict_with_prophet(d, 시작월, 종료월)
        # 실제 데이터
        hist_data = d.copy().sort_values("날짜")
        start_date = pd.to_datetime(시작월)
        end_date = pd.to_datetime(종료월)
        hist_window_start = start_date - pd.DateOffset(months=13)
        plot_mask = (hist_data["날짜"] >= hist_window_start) & (
            hist_data["날짜"] <= end_date
        )
        plot_data = hist_data[plot_mask].copy()
        ax = axs[plot_idx]
        plot_idx += 1
        ax.set_facecolor("white")
        if not plot_data.empty:
            ax.plot(
                plot_data["날짜"],
                plot_data["입국자수"],
                color="tab:blue",
                label="실제값(과거)",
                linewidth=2,
                zorder=2,
            )
        ax.plot(
            prophet_dates,
            prophet_preds,
            color=colors.get(m, "tab:green"),
            linestyle="-",
            marker="o",
            markersize=5,
            linewidth=2,
            label="예측값(Prophet)",
            zorder=3,
        )
        ax.set_title(f"[{국적}] {m}", fontsize=14)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle(f"{국적} 목적별 입국자수 예측 결과 (Prophet)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(run_folder, f"{국적}_전체목적_prophet_plot.png")
    fig.savefig(fname)
    plt.show()


if __name__ == "__main__":
    import re

    def valid_yyyymm(date_str):
        # YYYY-MM 형식(공백 포함 시 제거)만 True 반환
        return bool(re.fullmatch(r"\d{4}-\d{2}", date_str.strip()))

    csv_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../data_preprocessing/data/processed/외국인입국자_전처리완료_딥러닝용.csv",
        )
    )

    df_tmp = pd.read_csv(csv_path)
    available_nations = sorted(df_tmp["국적"].dropna().unique())
    available_purposes = sorted(df_tmp["목적"].dropna().unique())

    default_nation = "일본"
    default_purpose = "관광"
    default_start_month = "2025-06"
    default_end_month = "2026-05"

    while True:
        nation_input = (
            input(f"국적을 입력하세요 (예: 일본) [엔터 : {default_nation}]: ").strip()
            or default_nation
        )
        if nation_input in available_nations:
            국적 = nation_input
            break
        print(
            f"[경고] '{nation_input}'은(는) 데이터에 존재하지 않는 국적입니다. 가능한 값: {', '.join(available_nations[:10])} ..."
        )

    while True:
        purpose_input = input(
            f"목적을 입력하세요 (전체일 경우 엔터) [엔터 : {default_purpose}]: "
        ).strip()
        if not purpose_input:
            목적 = None
            break
        if purpose_input in available_purposes:
            목적 = purpose_input
            break
        print(
            f"[경고] '{purpose_input}'은(는) 데이터에 존재하지 않는 목적입니다. 가능한 값: {', '.join(available_purposes)}"
        )

    while True:
        시작월 = (
            input(
                f"예측 시작월을 입력하세요 (예: 2025-06) [엔터 : {default_start_month}]: "
            ).strip()
            or default_start_month
        ).strip()
        종료월 = (
            input(
                f"예측 종료월을 입력하세요 (예: 2026-05) [엔터 : {default_end_month}]: "
            ).strip()
            or default_end_month
        ).strip()
        # 입력값이 YYYY-MM 형식인지 재확인
        if not valid_yyyymm(시작월) or not valid_yyyymm(종료월):
            print("[경고] 올바른 형식(YYYY-MM)으로 입력해주세요.")
            continue
        if 시작월 >= 종료월:
            print("[경고] 예측 시작월은 종료월보다 이전이어야 합니다.")
            continue
        break

    forecast_visitors_prophet_only(
        csv_path=csv_path, 국적=국적, 목적=목적, 시작월=시작월, 종료월=종료월
    )
