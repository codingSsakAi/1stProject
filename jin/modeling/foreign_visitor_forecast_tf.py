# foreign_visitor_forecast_tf.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import platform
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from adjustText import adjust_text
import warnings

# Suppress FancyArrowPatch warnings globally
warnings.filterwarnings("ignore", message=".*FancyArrowPatch.*")

# Mac에서 한글 폰트 설정
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# 시계열 데이터를 입력받아 (window 개월씩 잘라서) LSTM에 넣을 학습 데이터를 만듭니다
# 예: [1,2,3,4,5]에서 window=3일 경우 → X=[1,2,3], [2,3,4], y=4,5
def prepare_sequences(data, window, exog=None):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


# 실제값과 예측값을 비교하여 평가 지표를 계산합니다
# MAE: 평균 절대 오차, RMSE: 평균 제곱근 오차, MAPE: 평균 절대 백분율 오차
def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    return mae, rmse, mape


# 예측 정확도(MAPE)에 따라 신뢰도 등급 반환
def get_reliability_level(mape):
    if mape < 20:
        return "매우 높음"
    elif mape < 40:
        return "높음"
    elif mape < 60:
        return "보통 (주의 필요)"
    else:
        return "낮음 (신뢰 어려움)"


# 단일 목적에 대해 외국인 입국자 수를 예측하는 함수입니다
def forecast_one(df, label, 시작월, 종료월, window=12, epochs=100, model_type="auto"):
    # 데이터 정렬 및 유효성 검사
    df = df.sort_values("날짜")
    # --- Ratio-based prediction mode for 관광/유학연수 ---
    reference_label = "상용" if label in ["관광", "유학연수"] else None
    use_ratio_mode = False
    reference_series = None

    if reference_label:
        ref_df = df[df["목적"] == reference_label]
        if not ref_df.empty and "입국자수" in ref_df.columns:
            merged_df = pd.merge(
                df, ref_df[["날짜", "입국자수"]], on="날짜", suffixes=("", "_ref")
            )
            if not merged_df["입국자수_ref"].isnull().any():
                use_ratio_mode = True
                df = merged_df.copy()
                df["비율입국자수"] = df["입국자수"] / (df["입국자수_ref"] + 1e-6)
                target_col = "비율입국자수"
                reference_series = df["입국자수_ref"].values.astype(np.float32)
    # 데이터 부족시 예측 생략
    if len(df) < 18:
        print(f"[데이터 부족] '{label}' 목적 데이터가 18개월 미만입니다. 예측 생략.")
        return None, None, label, None, None

    # window size 조정 (유학연수/관광은 더 긴 시계열 활용, 최대 240개월까지 허용)
    if label == "유학연수":
        win = min(48, max(6, len(df) // 2), 240)
    elif label == "관광":
        win = min(48, max(6, len(df) // 2), 240)
    else:
        win = min(window, max(6, len(df) // 2), 240)
    # 위에서 win 계산에 max_possible_win 반영됨 (중복 방지)

    # Stacked LSTM 조건부 도입 (데이터가 60개월 이상인 경우)
    use_stacked = len(df) >= 60

    # 모델 타입 자동 선택 (데이터 짧을 때 fallback)
    if model_type == "auto":
        if len(df) < 30:
            model_type = "linear"  # fallback for short data
        elif label == "공용":
            model_type = "xgboost"
            if len(df) < 36:
                model_type = "linear"
        elif label == "상용":
            model_type = "linear"
        else:
            model_type = "lstm"
    # Enforce model_type to xgboost for '공용'
    if label == "공용":
        model_type = "xgboost"
    if df.empty or "입국자수" not in df.columns or df["입국자수"].dropna().empty:
        print(f"데이터 없음: {label} — 예측 생략")
        return None, None, label, None, None

    # --- Choose series: ratio or raw, depending on use_ratio_mode ---
    if use_ratio_mode:
        series = df[target_col].values.astype(np.float32)
    else:
        series = df["입국자수"].values.astype(np.float32)
    if len(series) == 0:
        print(f"시계열 데이터 없음: {label} — 예측 생략")
        return None, None, label, None, None

    max_val = series.max()
    series_norm = series / max_val
    X, y = prepare_sequences(series_norm, win)
    if len(X) == 0:
        print(f"학습용 시퀀스 없음: {label} — 예측 생략")
        return None, None, label, None, None
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 학습/검증 데이터 분리
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    n_months = (pd.to_datetime(종료월) - pd.to_datetime(시작월)).days // 30 + 1

    if model_type == "lstm":
        print(f"\n[LSTM 학습 시작] 목적: {label} — window={win}, epochs={epochs}")

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = []
        if len(X_val) >= 5:
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
                ),
            ]
        else:
            print(f"[주의] '{label}' 목적: 검증 데이터가 적어 callbacks 생략")

        # 코로나 시기 가중치 약화 처리
        weights = np.ones(len(y_train))
        X_train_idx = np.arange(len(X))[:split_idx]
        if "날짜" in df.columns:
            corona_mask = df["날짜"].between("2020-03-01", "2022-10-01")
            corona_flags = corona_mask.astype(int).values
            weights = np.where(corona_flags[X_train_idx], 0.3, 1.0)

        # Model definition with Dropout for 유학연수 (and Bidirectional for both layers)
        if label == "유학연수":
            model = Sequential(
                [
                    Input(shape=(win, 1)),
                    Bidirectional(LSTM(64, return_sequences=True)),
                    Dropout(0.3),
                    Bidirectional(LSTM(32, return_sequences=False)),
                    Dense(1),
                ]
            )
        elif label == "관광":
            if use_stacked:
                model = Sequential(
                    [
                        Input(shape=(win, 1)),
                        Bidirectional(LSTM(64, return_sequences=True)),
                        Bidirectional(LSTM(32, return_sequences=False)),
                        Dense(1),
                    ]
                )
            else:
                model = Sequential(
                    [
                        Input(shape=(win, 1)),
                        Bidirectional(LSTM(64, return_sequences=False)),
                        Dense(1),
                    ]
                )
        else:
            model = Sequential(
                [Input(shape=(win, 1)), LSTM(64, return_sequences=False), Dense(1)]
            )
        model.compile(optimizer="adam", loss="mse", metrics=["mae", "accuracy"])
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            sample_weight=weights,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # 학습 로그 출력
        history_df = pd.DataFrame(history.history)
        print("\n[학습 로그]")
        print(
            history_df[
                ["loss", "mae", "accuracy", "val_loss", "val_mae", "val_accuracy"]
            ].tail(10)
        )

        # 미래 데이터 예측 생성
        future_input = series_norm[-win:].tolist()
        predictions = []
        for _ in range(n_months):
            input_seq = np.array(future_input[-win:]).reshape((1, win, 1))
            pred = model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(pred)
            future_input.append(pred)

        future_dates = [
            df["날짜"].max() + pd.DateOffset(months=i + 1) for i in range(n_months)
        ]
        pred_values = np.array(predictions) * max_val

        # --- Restore predicted values if using ratio mode ---
        if use_ratio_mode and reference_series is not None:
            # Use last reference as a simple fallback for future reference scaling
            future_ref = reference_series[-1]
            pred_values = np.array(predictions) * future_ref
        # 예측값 후처리
        if label == "유학연수":
            pred_values = np.clip(pred_values, a_min=1000, a_max=None)
            pred_values = (
                pd.Series(pred_values)
                .rolling(window=7, min_periods=1)
                .mean()
                .ewm(span=5)
                .mean()
                .values
            )
            # Flat 보정: 상용 대비 비율 기반 상향 조정
            if reference_series is not None:
                scale = np.mean(reference_series[-3:]) * 0.05
                pred_values = np.maximum(pred_values, scale)
        elif label == "관광":
            pred_values = (
                pd.Series(pred_values).rolling(window=5, min_periods=1).mean().values
            )
        elif label == "공용":
            pred_values = (
                pd.Series(pred_values)
                .rolling(window=5, min_periods=1)
                .mean()
                .ewm(span=3)
                .mean()
                .values
            )

        # Flat prediction warning before return
        pred_std = np.std(pred_values)
        if pred_std < 5:
            print(
                f"[경고] '{label}' 예측값 표준편차 매우 낮음 → flat 경향 의심됨 (보완 필요)"
            )
        # Log-scale std check
        pred_std_log = np.std(np.log1p(pred_values))
        if pred_std_log < 0.015:
            print(
                f"[경고] '{label}' 예측값 로그 변동폭 작음 → flat 경향 의심됨 (보완 필요)"
            )

        # 검증 데이터에 대한 예측 및 평가 지표 계산
        pred_val = model.predict(X_val, verbose=1).flatten() * max_val
        true_val = y_val * max_val
        mae, rmse, mape = compute_metrics(true_val, pred_val)
        reliability = get_reliability_level(mape)
        print(
            f"[검증 결과] {label} — MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}% → 신뢰도: {reliability}"
        )
        if mape < 40:
            print(
                f"[실무 적용 판단] '{label}' 목적 예측은 실무 적용 가능성 높음 (MAPE={mape:.2f}%)"
            )
        elif mape < 60:
            print(
                f"[실무 적용 판단] '{label}' 목적 예측은 실무 적용에 보완 필요 (MAPE={mape:.2f}%)"
            )
        else:
            print(
                f"[실무 적용 판단] '{label}' 목적 예측은 실무 적용 어려움 (MAPE={mape:.2f}%)"
            )

    elif model_type == "xgboost":
        import xgboost as xgb

        X_train_2d = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        X_val_2d = X_val.reshape((X_val.shape[0], X_val.shape[1]))
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
        model.fit(X_train_2d, y_train)
        pred_val = model.predict(X_val_2d)
        future_input = series_norm[-win:].tolist()
        predictions = []
        for _ in range(n_months):
            input_seq = np.array(future_input[-win:]).reshape(1, -1)
            pred = model.predict(input_seq)[0]
            predictions.append(pred)
            future_input.append(pred)
        future_dates = [
            df["날짜"].max() + pd.DateOffset(months=i + 1) for i in range(n_months)
        ]
        pred_values = np.array(predictions) * max_val
        mae, rmse, mape = compute_metrics(y_val * max_val, pred_val * max_val)
        reliability = get_reliability_level(mape)
        print(
            f"[검증 결과] {label} — MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}% → 신뢰도: {reliability}"
        )
        if mape < 40:
            print(
                f"[실무 적용 판단] '{label}' 목적 예측은 실무 적용 가능성 높음 (MAPE={mape:.2f}%)"
            )
        elif mape < 60:
            print(
                f"[실무 적용 판단] '{label}' 목적 예측은 실무 적용에 보완 필요 (MAPE={mape:.2f}%)"
            )
        else:
            print(
                f"[실무 적용 판단] '{label}' 목적 예측은 실무 적용 어려움 (MAPE={mape:.2f}%)"
            )
        # Flat prediction warning before return
        pred_std = np.std(pred_values)
        if pred_std < 5:
            print(
                f"[경고] '{label}' 예측값 표준편차 매우 낮음 → flat 경향 의심됨 (보완 필요)"
            )
        pred_std_log = np.std(np.log1p(pred_values))
        if pred_std_log < 0.015:
            print(
                f"[경고] '{label}' 예측값 로그 변동폭 작음 → flat 경향 의심됨 (보완 필요)"
            )
        return (
            future_dates,
            pred_values,
            label,
            df[["날짜", "입국자수"]],
            (mae, rmse, mape),
        )

    elif model_type == "linear":
        from sklearn.linear_model import LinearRegression

        X_train_2d = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        X_val_2d = X_val.reshape((X_val.shape[0], X_val.shape[1]))
        model = LinearRegression()
        model.fit(X_train_2d, y_train)
        pred_val = model.predict(X_val_2d)
        future_input = series_norm[-win:].tolist()
        predictions = []
        for _ in range(n_months):
            input_seq = np.array(future_input[-win:]).reshape(1, -1)
            pred = model.predict(input_seq)[0]
            predictions.append(pred)
            future_input.append(pred)
        future_dates = [
            df["날짜"].max() + pd.DateOffset(months=i + 1) for i in range(n_months)
        ]
        pred_values = np.array(predictions) * max_val
        mae, rmse, mape = compute_metrics(y_val * max_val, pred_val * max_val)
        reliability = get_reliability_level(mape)
        print(
            f"[검증 결과] {label} — MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}% → 신뢰도: {reliability}"
        )
        if mape < 40:
            print(
                f"[실무 적용 판단] '{label}' 목적 예측은 실무 적용 가능성 높음 (MAPE={mape:.2f}%)"
            )
        elif mape < 60:
            print(
                f"[실무 적용 판단] '{label}' 목적 예측은 실무 적용에 보완 필요 (MAPE={mape:.2f}%)"
            )
        else:
            print(
                f"[실무 적용 판단] '{label}' 목적 예측은 실무 적용 어려움 (MAPE={mape:.2f}%)"
            )
        # Flat prediction warning before return
        pred_std = np.std(pred_values)
        if pred_std < 5:
            print(
                f"[경고] '{label}' 예측값 표준편차 매우 낮음 → flat 경향 의심됨 (보완 필요)"
            )
        pred_std_log = np.std(np.log1p(pred_values))
        if pred_std_log < 0.015:
            print(
                f"[경고] '{label}' 예측값 로그 변동폭 작음 → flat 경향 의심됨 (보완 필요)"
            )
        return (
            future_dates,
            pred_values,
            label,
            df[["날짜", "입국자수"]],
            (mae, rmse, mape),
        )

    # --- 예측값 후처리 (보정 또는 평탄화) ---
    # MAPE가 30~100% 범위이면, 예측값이 불안정한 경우로 간주하여 후처리를 적용합니다.
    if 30 <= mape <= 100:
        # 1) 평균 오차 기반 보정 계수 계산
        correction_factor = true_val.sum() / (
            pred_val.sum() + 1e-6
        )  # 0으로 나누기 방지
        # 2) 예측값 보정 (스케일 보정)
        pred_values = pred_values * correction_factor
        # 3) 이동 평균 보정 (들쭉날쭉한 값을 부드럽게)
        pred_values = (
            pd.Series(pred_values).rolling(window=3, min_periods=1).mean().values
        )

    # Flat prediction warning before return (for auto/other branches)
    pred_std = np.std(pred_values)
    if pred_std < 5:
        print(
            f"[경고] '{label}' 예측값 표준편차 매우 낮음 → flat 경향 의심됨 (보완 필요)"
        )
    pred_std_log = np.std(np.log1p(pred_values))
    if pred_std_log < 0.015:
        print(
            f"[경고] '{label}' 예측값 로그 변동폭 작음 → flat 경향 의심됨 (보완 필요)"
        )

    return future_dates, pred_values, label, df[["날짜", "입국자수"]], (mae, rmse, mape)


# 메인 함수: 데이터 로딩 → 예측 반복 → 시각화 → 결과 저장
def forecast_visitors(csv_path, 국적, 목적, 시작월, 종료월, window=12, epochs=100):
    # 결과 저장 폴더 생성
    BASE_DIR = "../data_preprocessing/data/processed/"
    RESULT_DIR = "./results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(RESULT_DIR, f"forecast_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    # CSV 파일에서 데이터 로딩 및 날짜 컬럼 생성
    df = pd.read_csv(os.path.join(BASE_DIR, csv_path))
    if "연도" in df.columns and "월" in df.columns:
        df["날짜"] = pd.to_datetime(
            df[["연도", "월"]]
            .rename(columns={"연도": "year", "월": "month"})
            .assign(day=1)
        )
    elif "연" in df.columns and "월" in df.columns:
        df["날짜"] = pd.to_datetime(
            df[["연", "월"]].rename(columns={"연": "year", "월": "month"}).assign(day=1)
        )
    else:
        raise ValueError("CSV에 연도/월 정보가 없습니다.")

    all_forecast_rows = []
    metrics_rows = []
    목적순서 = ["관광", "유학연수", "공용", "상용"]
    colors = {
        "관광": "tab:blue",
        "유학연수": "tab:green",
        "공용": "tab:orange",
        "상용": "tab:red",
    }

    # 예측할 목적별 데이터 분리
    if 목적:
        targets = [(목적, df[(df["국적"] == 국적) & (df["목적"] == 목적)])]
    else:
        targets = [(m, df[(df["국적"] == 국적) & (df["목적"] == m)]) for m in 목적순서]

    # 각 목적별로 예측 수행
    for i, (m, d) in enumerate(targets):
        future_dates, pred_values, label, hist_df, metrics = forecast_one(
            d, m, 시작월, 종료월, window, epochs
        )
        if future_dates is None:
            print(f"[SKIP] '{m}' 목적 예측 생략됨 (데이터 부족 또는 모델 실패)")
            continue

        # 결과 저장용 데이터프레임에 추가
        all_forecast_rows.append(
            pd.DataFrame(
                {
                    "국적": 국적,
                    "목적": label,
                    "날짜": future_dates,
                    "예측입국자수": pred_values.astype(int),
                }
            )
        )

        # 평가 지표 저장
        if metrics:
            mae, rmse, mape = metrics
            metrics_rows.append(
                {
                    "국적": 국적,
                    "목적": label,
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2),
                    "MAPE": round(mape, 2),
                    "신뢰도": get_reliability_level(mape),
                }
            )

    # 결과 저장용 데이터프레임에 추가 (이미 위에서 수행)

    # 예측 결과 및 평가 지표 CSV로 저장
    # Define file_prefix before exporting CSVs
    file_prefix = f"{국적}_{목적 or '전체'}"
    if all_forecast_rows:
        result_df = pd.concat(all_forecast_rows, ignore_index=True)
        result_df.to_csv(
            os.path.join(run_folder, file_prefix + ".csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(
            os.path.join(run_folder, file_prefix + "_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # 시각화: 목적별 예측 결과 그래프 (수직 4줄 subplot)
    if all_forecast_rows:
        fig, axs = plt.subplots(
            4, 1, figsize=(16, 20)
        )  # 넓은 가로폭, 4줄 수직 레이아웃
        for i, forecast_df in enumerate(all_forecast_rows):
            m = forecast_df["목적"].iloc[0]
            pred_df = forecast_df.copy()
            hist_data = df[(df["국적"] == 국적) & (df["목적"] == m)].copy()
            hist_data = hist_data.sort_values("날짜")
            start_date = pd.to_datetime(시작월)
            hist_window_start = start_date - pd.DateOffset(months=14)
            hist_window = hist_data[
                (hist_data["날짜"] >= hist_window_start)
                & (hist_data["날짜"] < start_date)
            ].tail(13)
            # Skip plot if both hist_window and pred_df are empty
            if hist_window.empty and pred_df.empty:
                print(f"[그래프 생략] '{m}' 목적: 시각화할 데이터 없음")
                continue
            ax = axs[i]
            ax.set_facecolor("white")
            # 회색 과거 데이터
            ax.plot(
                hist_window["날짜"],
                hist_window["입국자수"],
                color="gray",
                label="이전 13개월",
            )
            # 예측 선: 빨간색 점선
            ax.plot(
                pred_df["날짜"],
                pred_df["예측입국자수"],
                color="red",
                linestyle="--",
                marker="o",
                label="예측값",
            )
            # 예측값 수치 표기 (위아래 번갈아가며)
            for j, (x, y) in enumerate(zip(pred_df["날짜"], pred_df["예측입국자수"])):
                offset = 1000 if j % 2 == 0 else -1000
                va = "bottom" if offset > 0 else "top"
                ax.text(
                    x,
                    y + offset,
                    f"{int(y):,}",
                    ha="center",
                    va=va,
                    fontsize=9,
                    color="red",
                )
            # 노란 배경 표시: 예측시작월~예측시작월+24개월
            forecast_start = pd.to_datetime(시작월)
            forecast_end = forecast_start + pd.DateOffset(months=24)
            ax.axvspan(
                forecast_start,
                forecast_end,
                color="yellow",
                alpha=0.3,
                label="예측 기간 강조",
            )
            ax.set_title(f"[{국적}] {m} 입국자수 예측")
            ax.set_xlabel("날짜")
            ax.set_ylabel("입국자 수")
            ax.legend()
            ax.grid(True)
        fname = os.path.join(run_folder, f"{국적}_전체목적_세로plot.png")
        fig.subplots_adjust(hspace=0.35, top=0.95, bottom=0.05)
        fig.savefig(fname)


if __name__ == "__main__":
    import re

    def valid_yyyymm(date_str):
        return bool(re.match(r"^\d{4}-\d{2}$", date_str))

    print("외국인 입국자 예측 시스템")
    csv_path = "외국인입국자_전처리완료_딥러닝용.csv"

    국적 = input("국적을 입력하세요 (예: 일본): ").strip()
    목적 = input("목적을 입력하세요 (전체일 경우 엔터): ").strip() or None

    while True:
        시작월 = input("예측 시작월을 입력하세요 (예: 2025-06): ").strip()
        종료월 = input("예측 종료월을 입력하세요 (예: 2026-05): ").strip()
        if not valid_yyyymm(시작월) or not valid_yyyymm(종료월):
            print("⚠️ 올바른 형식(YYYY-MM)으로 입력해주세요.")
            continue
        if 시작월 >= 종료월:
            print("⚠️ 예측 시작월은 종료월보다 이전이어야 합니다.")
            continue
        break

    forecast_visitors(
        csv_path=csv_path, 국적=국적, 목적=목적, 시작월=시작월, 종료월=종료월
    )
