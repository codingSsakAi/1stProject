import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams["font.family"] = "Malgun Gothic"
# plt.rcParams["axes.unicode_minus"] = False


def normalize(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()


선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도 = int(input("예측할 연도 입력 (예: 2025): ").strip())
예측월입력 = input("예측할 월 입력 (1~12 또는 3,6,9 또는 10~12): ").strip()

if "~" in 예측월입력:
    start, end = map(int, 예측월입력.split("~"))
    예측월 = list(range(start, end + 1))
elif "," in 예측월입력:
    예측월 = list(map(int, 예측월입력.split(",")))
else:
    예측월 = [int(예측월입력)]

path = "./data/외국인입국자_전처리완료_딥러닝용.csv"
df = pd.read_csv(path, encoding="utf-8")
df.columns = df.columns.str.strip()
df["국적"] = df["국적"].astype(str).str.strip()
df["목적"] = df["목적"].astype(str).str.strip()

unique_국적 = df["국적"].dropna().unique()
unique_목적 = df["목적"].dropna().unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n\U0001F449 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다.")
        선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다.")
        선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
targets = targets[targets["count"] > 36]
if 선택_국적:
    targets = targets[targets["국적"] == 선택_국적]
if 선택_목적:
    targets = targets[targets["목적"] == 선택_목적]

for _, row in targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = (
        df[(df["국적"] == 국적) & (df["목적"] == 목적)]
        .sort_values(["연도", "월"])
        .reset_index(drop=True)
    )
    if len(df_filtered) < 36:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    df_filtered["연월"] = (
        df_filtered["연도"].astype(str) + "-" + df_filtered["월"].astype(str).str.zfill(2)
    )
    df_filtered["일자"] = pd.to_datetime(df_filtered["연월"])
    last_date = df_filtered["일자"].iloc[-1]

    df_filtered["전월"] = df_filtered["입국자수"].shift(1)
    df_filtered["전년"] = df_filtered["입국자수"].shift(12)
    df_filtered["전월증감률"] = df_filtered["입국자수"].pct_change().shift(1)
    df_filtered["전년증감률"] = (df_filtered["입국자수"] - df_filtered["전년"]) / df_filtered[
        "전년"
    ]
    df_filtered["이동평균3"] = df_filtered["입국자수"].rolling(3).mean().shift(1)
    df_filtered["이동평균6"] = df_filtered["입국자수"].rolling(6).mean().shift(1)
    df_filtered["전월_차이"] = df_filtered["입국자수"] - df_filtered["전월"]
    df_filtered["전년_차이"] = df_filtered["입국자수"] - df_filtered["전년"]

    if 목적 == "유학연수":
        df_filtered["전전월"] = df_filtered["입국자수"].shift(2)
        df_filtered["전전년"] = df_filtered["입국자수"].shift(24)
        df_filtered["이동평균12"] = df_filtered["입국자수"].rolling(12).mean().shift(1)
        df_filtered["누적합3"] = df_filtered["입국자수"].rolling(3).sum().shift(1)
        df_filtered["월"] = df_filtered["월"].astype(int)

    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if 목적 == "유학연수":
        features = [
            "전월",
            "전전월",
            "전년",
            "전전년",
            "전월증감률",
            "전년증감률",
            "이동평균3",
            "이동평균6",
            "이동평균12",
            "누적합3",
            "전월_차이",
            "전년_차이",
            "월",
        ]
        y = np.log1p(df_filtered["입국자수"])
    else:
        features = [
            "전월",
            "전년",
            "전월증감률",
            "전년증감률",
            "이동평균3",
            "이동평균6",
            "전월_차이",
            "전년_차이",
        ]
        y = df_filtered["입국자수"]

    X = df_filtered[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = X_scaled[:-6], X_scaled[-6:]
    y_train, y_test = y[:-6], y[-6:]

    # 성능 개선: 목적이 '상용'인 경우만 하이퍼파라미터 튜닝
    if 목적 == "상용":
        param_grid = {
            "n_estimators": [300, 500, 700],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [3, 5, 7],
        }
        model = XGBRegressor(random_state=42)
        search = RandomizedSearchCV(
            model, param_grid, n_iter=5, cv=3, scoring="neg_mean_squared_error", random_state=42
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if 목적 == "유학연수":
        y_pred_eval = np.expm1(y_pred)
        y_test_eval = np.expm1(y_test)
    else:
        y_pred_eval = y_pred
        y_test_eval = y_test

    r2 = r2_score(y_test_eval, y_pred_eval)
    mape = mean_absolute_percentage_error(y_test_eval, y_pred_eval) * 100
    rmse = mean_squared_error(y_test_eval, y_pred_eval, squared=False)
    dw = durbin_watson(y_test_eval - y_pred_eval)
    신뢰도 = max(0, min(100, 100 - mape))

    print("\u2705 XGBoost 성능 평가")
    print(f" - R^2: {r2:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - 신뢰도: {신뢰도:.1f}%")
    print(f" - Durbin-Watson: {dw:.4f}")

    예측결과들 = []
    for month in 예측월:
        예측_월수 = int(
            (예측연도 - df_filtered["일자"].dt.year.iloc[-1]) * 12
            + (month - df_filtered["일자"].dt.month.iloc[-1])
        )

        future_preds = []
        recent = df_filtered.copy()
        for i in range(예측_월수):
            row = recent.iloc[-1:].copy()
            row["입국자수"] = future_preds[-1] if future_preds else row["입국자수"]
            row["전월"] = recent["입국자수"].iloc[-1]
            row["전전월"] = recent["입국자수"].iloc[-2] if 목적 == "유학연수" else np.nan
            row["전년"] = recent["입국자수"].iloc[-12]
            row["전전년"] = recent["입국자수"].iloc[-24] if 목적 == "유학연수" else np.nan
            row["전월증감률"] = (
                recent["입국자수"].iloc[-1] - recent["입국자수"].iloc[-2]
            ) / recent["입국자수"].iloc[-2]
            row["전년증감률"] = (
                recent["입국자수"].iloc[-1] - recent["입국자수"].iloc[-12]
            ) / recent["입국자수"].iloc[-12]
            row["이동평균3"] = recent["입국자수"].iloc[-3:].mean()
            row["이동평균6"] = recent["입국자수"].iloc[-6:].mean()
            row["이동평균12"] = (
                recent["입국자수"].iloc[-12:].mean() if 목적 == "유학연수" else np.nan
            )
            row["누적합3"] = recent["입국자수"].iloc[-3:].sum() if 목적 == "유학연수" else np.nan
            row["전월_차이"] = row["입국자수"] - row["전월"]
            row["전년_차이"] = row["입국자수"] - row["전년"]
            row["월"] = ((last_date.month + i - 1) % 12) + 1 if 목적 == "유학연수" else np.nan

            row = row[features].fillna(method="ffill", axis=1)
            row_scaled = scaler.transform(row)
            pred = model.predict(row_scaled)[0]
            pred = np.expm1(pred) if 목적 == "유학연수" else pred
            future_preds.append(pred)
            recent = pd.concat([recent, pd.DataFrame([{"입국자수": pred}])], ignore_index=True)

        pred_date = pd.to_datetime(f"{예측연도}-{month:02d}-01")
        예측값 = np.round(future_preds[-1]).astype(int)

        print(f"\n🔮 {예측연도}년 {month}월 예측 입국자 수: {예측값:,}명 (신뢰도: {신뢰도:.1f}%)")
        예측결과들.append((pred_date, 예측값))

    예측_df = pd.DataFrame(예측결과들, columns=["예측월", "예측입국자수"])
    csv_path = f"./output/{국적}_{목적}_예측결과.csv"
    예측_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(14, 6))
    최근기간 = 6
    시각화데이터 = df_filtered[-최근기간:].copy()
    plt.plot(
        시각화데이터["일자"], 시각화데이터["입국자수"], label="최근 실측", color="blue", linewidth=2
    )
    pred_dates = [d for d, _ in 예측결과들]
    pred_values = [v for _, v in 예측결과들]
    if pred_dates:
        plt.plot(
            pred_dates,
            pred_values,
            color="red",
            marker="o",
            linestyle="--",
            linewidth=2,
            label="예측값",
        )
    for pred_date, 예측값 in 예측결과들:
        plt.scatter(pred_date, 예측값, color="red", s=60, edgecolors="black", zorder=5)
        plt.text(
            pred_date,
            예측값 + max(pred_values) * 0.02,
            f"{예측값:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            fontweight="bold",
        )
    start_date = 시각화데이터["일자"].iloc[0]
    end_date = pred_dates[-1] + pd.DateOffset(months=1)
    plt.xlim([start_date, end_date])
    plt.title(f"{국적}/{목적} - {예측연도}년 {예측월}월 예측 결과", fontsize=14, fontweight="bold")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
