import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import platform
import os
from prophet import Prophet
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Mac에서 한글 폰트 설정
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

FEATURE_COLS = ["입국자수"]


# Prophet 예측 함수 (미래0 시나리오만, 외생변수 없이 단순 Prophet)
def predict_with_prophet_future0(df, start_month, end_month):
    df_prophet = (
        df[["날짜", "입국자수"]].rename(columns={"날짜": "ds", "입국자수": "y"}).copy()
    )
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
    pred0 = forecast.loc[pred_mask, "yhat"].values
    return pred_dates, pred0


# XGBoost 예측 함수
def predict_with_xgboost(df, start_month, end_month, window=12):
    import xgboost as xgb

    df = df.copy()
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df[FEATURE_COLS] = df[FEATURE_COLS].clip(lower=0)

    # 시계열 입력/출력 생성
    def prepare_sequences(data, window):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i : i + window])
            y.append(data[i + window])
        return np.array(X), np.array(y)

    X, y = prepare_sequences(df[FEATURE_COLS].values, window)
    if len(X) < 12:
        return None, None
    X_train, y_train = X, y
    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    model.fit(X_train_flat, y_train)
    # 미래 예측
    start_date = pd.to_datetime(start_month)
    end_date = pd.to_datetime(end_month)
    n_months = (
        (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    )
    last_window = df[FEATURE_COLS].values[-window:].tolist()
    preds = []
    for i in range(n_months):
        # last_window의 각 row가 항상 feature 개수(4개)인 float/int로만 구성되도록 보장
        X_input = (
            np.array(last_window, dtype=float).reshape(1, window, -1).reshape(1, -1)
        )
        pred_arr = model.predict(X_input)
        if hasattr(pred_arr, "ndim") and pred_arr.ndim == 1:
            pred_val = float(pred_arr[0])
        elif hasattr(pred_arr, "ndim") and pred_arr.ndim == 2:
            pred_val = float(pred_arr.ravel()[0])
        else:
            pred_val = float(pred_arr)
        preds.append(pred_val)
        # 미래 row feature 생성 (입국자수만)
        pred_row = [pred_val]
        last_window = last_window[1:] + [pred_row]
    future_dates = pd.date_range(start=start_date, periods=n_months, freq="MS")
    return future_dates, np.array(preds)


# LSTM 예측 함수
def predict_with_lstm(df, start_month, end_month, window=12, epochs=100):
    df = df.copy()
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df[FEATURE_COLS] = df[FEATURE_COLS].clip(lower=0)

    def prepare_sequences(data, window):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i : i + window])
            y.append(data[i + window])
        return np.array(X), np.array(y)

    X, y = prepare_sequences(df[FEATURE_COLS].values, window)
    if len(X) < 12:
        return None, None
    X_train, y_train = X, y
    model = Sequential(
        [
            Input(shape=(window, X.shape[2] if X.ndim == 3 else X.shape[1])),
            LSTM(32, return_sequences=False),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    callbacks = [EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)]
    model.fit(
        X_train, y_train, epochs=epochs, batch_size=8, verbose=2, callbacks=callbacks
    )
    # 미래 예측
    start_date = pd.to_datetime(start_month)
    end_date = pd.to_datetime(end_month)
    n_months = (
        (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    )
    last_window = df[FEATURE_COLS].values[-window:].tolist()
    preds = []
    for _ in range(n_months):
        # last_window를 항상 float 타입의 2D array로 변환
        X_input = np.array(last_window, dtype=float).reshape(1, window, -1)
        pred = model.predict(X_input, verbose=2)[0, 0]
        preds.append(pred)
        # 미래 row feature 생성 (입국자수만)
        pred_row = [pred]
        last_window = last_window[1:] + [pred_row]
    future_dates = pd.date_range(start=start_date, periods=n_months, freq="MS")
    return future_dates, np.array(preds)


# 평가 지표 계산 함수
def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    return mae, rmse, mape


# 메인 함수: 대화형 입력, 예측, 시각화, 결과 저장
def hybrid_ensemble_forecast(csv_path):
    RESULT_DIR = "./results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(RESULT_DIR, f"forecast_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["날짜"] = pd.to_datetime(
        df[["연도", "월"]].rename(columns={"연도": "year", "월": "month"}).assign(day=1)
    )
    # 코로나 플래그 및 변형 feature 자동 생성 (한글 주석 추가)
    # 코로나기간: 2020-03-01 ~ 2022-10-31 구간만 1, 나머지 0
    if "코로나기간" not in df.columns:
        df["코로나기간"] = (
            (df["날짜"] >= "2020-03-01") & (df["날짜"] <= "2022-10-31")
        ).astype(int)
    # 코로나기간_sin, 코로나기간_cos 생성
    df["코로나기간_sin"] = np.sin(df["코로나기간"] * np.pi)
    df["코로나기간_cos"] = np.cos(df["코로나기간"] * np.pi)
    목적순서 = ["관광", "유학연수", "공용", "상용"]
    colors = {
        "실제값": "tab:blue",
        "Prophet": "tab:green",
        "XGBoost": "tab:orange",
        "LSTM": "tab:red",
        "단순평균": "black",
        "Residual(Prophet+XGB)": "tab:purple",
        "Residual(Prophet+LSTM)": "tab:brown",
    }

    # 대화형 입력
    available_nations = sorted(df["국적"].dropna().unique())
    available_purposes = sorted(df["목적"].dropna().unique())
    default_nation = "일본"
    default_purpose = "관광"
    default_start_month = "2025-06"
    default_end_month = "2026-05"
    nation_input = (
        input(f"국적을 입력하세요 (예: 일본) [엔터 : {default_nation}]: ").strip()
        or default_nation
    )
    while nation_input not in available_nations:
        print(
            f"[경고] '{nation_input}'은(는) 데이터에 존재하지 않는 국적입니다. 가능한 값: {', '.join(available_nations[:10])} ..."
        )
        nation_input = (
            input(f"국적을 입력하세요 (예: 일본) [엔터 : {default_nation}]: ").strip()
            or default_nation
        )
    purpose_input = input(
        f"목적을 입력하세요 (전체일 경우 엔터) [엔터 : {default_purpose}]: "
    ).strip()
    if not purpose_input:
        purposes = 목적순서
    else:
        purposes = [purpose_input] if purpose_input in available_purposes else 목적순서
    start_month = (
        input(
            f"예측 시작월을 입력하세요 (예: 2025-06) [엔터 : {default_start_month}]: "
        ).strip()
        or default_start_month
    )
    end_month = (
        input(
            f"예측 종료월을 입력하세요 (예: 2026-05) [엔터 : {default_end_month}]: "
        ).strip()
        or default_end_month
    )

    for m in purposes:
        d = df[(df["국적"] == nation_input) & (df["목적"] == m)]
        if len(d) == 0:
            continue
        # Prophet 예측 (미래0 시나리오만)
        prophet_dates, prophet0 = predict_with_prophet_future0(
            d, start_month, end_month
        )
        # XGBoost 예측
        xgb_dates, xgb_preds = predict_with_xgboost(d, start_month, end_month)
        # LSTM 예측
        lstm_dates, lstm_preds = predict_with_lstm(d, start_month, end_month)
        # 단순평균 (Prophet 0 시나리오와 XGB, LSTM)
        preds_list = [
            arr for arr in [prophet0, xgb_preds, lstm_preds] if arr is not None
        ]
        min_len = min(len(arr) for arr in preds_list)
        simple_avg = np.mean([arr[:min_len] for arr in preds_list], axis=0)
        # Residual(Prophet+XGB)
        if xgb_preds is not None:
            # Prophet 잔차 학습용 데이터 생성
            d_train = d.copy().sort_values("날짜")
            prophet_train_dates, prophet_train_preds = predict_with_prophet_future0(
                d_train,
                d_train["날짜"].min().strftime("%Y-%m"),
                d_train["날짜"].max().strftime("%Y-%m"),
            )
            d_train = d_train.reset_index(drop=True)
            d_train["prophet_pred"] = prophet_train_preds
            d_train["residual"] = d_train["입국자수"] - d_train["prophet_pred"]
            # XGBoost로 잔차 예측
            import xgboost as xgb

            window = 12

            def prepare_sequences(data, window):
                X, y = [], []
                for i in range(len(data) - window):
                    X.append(data[i : i + window])
                    y.append(data[i + window])
                return np.array(X), np.array(y)

            X_res, y_res = prepare_sequences(d_train[["residual"]].values, window)
            if len(X_res) > 0:
                model_res = xgb.XGBRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1
                )
                X_res_flat = X_res.reshape(X_res.shape[0], -1)
                model_res.fit(X_res_flat, y_res)
                # 미래 Prophet 예측값 기반 잔차 예측
                last_window = d_train[["residual"]].values[-window:].tolist()
                residual_preds = []
                for _ in range(min_len):
                    X_input = (
                        np.array(last_window).reshape(1, window, -1).reshape(1, -1)
                    )
                    pred = model_res.predict(X_input)[0]
                    residual_preds.append(pred)
                    last_window = last_window[1:] + [[pred]]
                residual_preds = np.array(residual_preds)
                residual_xgb = prophet0[:min_len] + residual_preds
            else:
                residual_xgb = None
        else:
            residual_xgb = None
        # Residual(Prophet+LSTM)
        if lstm_preds is not None:
            d_train = d.copy().sort_values("날짜")
            prophet_train_dates, prophet_train_preds = predict_with_prophet_future0(
                d_train,
                d_train["날짜"].min().strftime("%Y-%m"),
                d_train["날짜"].max().strftime("%Y-%m"),
            )
            d_train = d_train.reset_index(drop=True)
            d_train["prophet_pred"] = prophet_train_preds
            d_train["residual"] = d_train["입국자수"] - d_train["prophet_pred"]
            window = 12

            def prepare_sequences(data, window):
                X, y = [], []
                for i in range(len(data) - window):
                    X.append(data[i : i + window])
                    y.append(data[i + window])
                return np.array(X), np.array(y)

            X_res, y_res = prepare_sequences(d_train[["residual"]].values, window)
            if len(X_res) > 0:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Input, LSTM, Dense
                from tensorflow.keras.callbacks import EarlyStopping

                model_res = Sequential(
                    [
                        Input(shape=(window, 1)),
                        LSTM(16, return_sequences=False),
                        Dense(1),
                    ]
                )
                model_res.compile(optimizer="adam", loss="mse")
                callbacks = [
                    EarlyStopping(
                        monitor="loss", patience=10, restore_best_weights=True
                    )
                ]
                model_res.fit(
                    X_res,
                    y_res,
                    epochs=50,
                    batch_size=8,
                    verbose=2,
                    callbacks=callbacks,
                )
                last_window = d_train[["residual"]].values[-window:].tolist()
                residual_preds = []
                for _ in range(min_len):
                    X_input = np.array(last_window).reshape(1, window, 1)
                    pred = model_res.predict(X_input, verbose=2)[0, 0]
                    residual_preds.append(pred)
                    last_window = last_window[1:] + [[pred]]
                residual_preds = np.array(residual_preds)
                residual_lstm = prophet0[:min_len] + residual_preds
            else:
                residual_lstm = None
        else:
            residual_lstm = None
        # Prophet 추세 보정 (5% 성장률)
        prophet_trend_boost = prophet0[:min_len] * 1.05  # 5% 성장률 보정

        # 실제값(과거) 추출
        hist_data = d.copy().sort_values("날짜")
        start_date = pd.to_datetime(start_month)
        end_date = pd.to_datetime(end_month)
        hist_window_start = start_date - pd.DateOffset(months=13)
        plot_mask = (hist_data["날짜"] >= hist_window_start) & (
            hist_data["날짜"] <= end_date
        )
        plot_data = hist_data[plot_mask].copy()
        # 시각화
        plt.figure(figsize=(12, 7))
        if not plot_data.empty:
            plt.plot(
                plot_data["날짜"],
                plot_data["입국자수"],
                color=colors["실제값"],
                label="실제값(과거)",
                linewidth=2,
                zorder=2,
            )
        plt.plot(
            prophet_dates[:min_len],
            prophet0[:min_len],
            color="tab:green",
            linestyle="-",
            marker="o",
            markersize=5,
            linewidth=2,
            label="Prophet",
            zorder=3,
        )
        if xgb_dates is not None:
            plt.plot(
                xgb_dates[:min_len],
                xgb_preds[:min_len],
                color=colors["XGBoost"],
                linestyle="-",
                marker="o",
                markersize=5,
                linewidth=2,
                label="XGBoost",
                zorder=3,
            )
        if lstm_dates is not None:
            plt.plot(
                lstm_dates[:min_len],
                lstm_preds[:min_len],
                color=colors["LSTM"],
                linestyle="-",
                marker="o",
                markersize=5,
                linewidth=2,
                label="LSTM",
                zorder=3,
            )
        plt.plot(
            prophet_dates[:min_len],
            simple_avg,
            color=colors["단순평균"],
            linestyle="-",
            marker="o",
            markersize=5,
            linewidth=2,
            label="단순평균",
            zorder=4,
        )
        if residual_xgb is not None:
            plt.plot(
                prophet_dates[:min_len],
                residual_xgb,
                color=colors["Residual(Prophet+XGB)"],
                linestyle="-",
                marker="o",
                markersize=5,
                linewidth=2,
                label="Residual(Prophet+XGB)",
                zorder=4,
            )
        if residual_lstm is not None:
            plt.plot(
                prophet_dates[:min_len],
                residual_lstm,
                color=colors["Residual(Prophet+LSTM)"],
                linestyle="-",
                marker="o",
                markersize=5,
                linewidth=2,
                label="Residual(Prophet+LSTM)",
                zorder=4,
            )
        plt.title(f"[{nation_input}] {m} 목적 입국자수 예측 비교", fontsize=16)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fname = os.path.join(run_folder, f"{nation_input}_{m}_hybrid_compare_plot.png")
        plt.savefig(fname)
        plt.close()
        # 신뢰도/검증 수치 저장
        actual = d[(d["날짜"] >= start_date) & (d["날짜"] <= end_date)][
            "입국자수"
        ].values
        metrics_rows = []

        def safe_metrics(pred):
            if pred is not None and len(actual) == len(pred):
                return compute_metrics(actual, pred)
            else:
                return (np.nan, np.nan, np.nan)

        metrics_rows.append(
            {
                "방식": "Prophet",
                "MAE": safe_metrics(prophet0[:min_len])[0],
                "RMSE": safe_metrics(prophet0[:min_len])[1],
                "MAPE": safe_metrics(prophet0[:min_len])[2],
            }
        )
        metrics_rows.append(
            {
                "방식": "XGBoost",
                "MAE": safe_metrics(xgb_preds[:min_len])[0],
                "RMSE": safe_metrics(xgb_preds[:min_len])[1],
                "MAPE": safe_metrics(xgb_preds[:min_len])[2],
            }
        )
        metrics_rows.append(
            {
                "방식": "LSTM",
                "MAE": safe_metrics(lstm_preds[:min_len])[0],
                "RMSE": safe_metrics(lstm_preds[:min_len])[1],
                "MAPE": safe_metrics(lstm_preds[:min_len])[2],
            }
        )
        metrics_rows.append(
            {
                "방식": "단순평균",
                "MAE": safe_metrics(simple_avg)[0],
                "RMSE": safe_metrics(simple_avg)[1],
                "MAPE": safe_metrics(simple_avg)[2],
            }
        )
        metrics_rows.append(
            {
                "방식": "Residual(Prophet+XGB)",
                "MAE": safe_metrics(residual_xgb)[0],
                "RMSE": safe_metrics(residual_xgb)[1],
                "MAPE": safe_metrics(residual_xgb)[2],
            }
        )
        metrics_rows.append(
            {
                "방식": "Residual(Prophet+LSTM)",
                "MAE": safe_metrics(residual_lstm)[0],
                "RMSE": safe_metrics(residual_lstm)[1],
                "MAPE": safe_metrics(residual_lstm)[2],
            }
        )
        pd.DataFrame(metrics_rows).to_csv(
            os.path.join(run_folder, f"{nation_input}_{m}_hybrid_compare_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    csv_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../data_preprocessing/data/processed/외국인입국자_전처리완료_딥러닝용.csv",
        )
    )
    hybrid_ensemble_forecast(csv_path)
