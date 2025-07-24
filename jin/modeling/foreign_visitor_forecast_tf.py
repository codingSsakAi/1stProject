import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
import matplotlib.pyplot as plt
from datetime import datetime
import platform
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from sklearn.preprocessing import MinMaxScaler
import joblib
from scipy.stats import pearsonr, spearmanr  # 상관계수 계산용
import logging
from prophet import Prophet  # Prophet 추가
import math

# Mac에서 한글 폰트 설정
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 사용할 feature 컬럼 리스트 (csv 헤더 기준)
FEATURE_COLS = ["입국자수"]


def save_model_and_scaler(
    model, scaler, 국적, 목적, model_type, save_dir="../flask/models"
):
    """모델과 스케일러를 저장하는 함수"""
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 파일명 생성 (특수문자 처리)
    safe_국적 = 국적.replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_목적 = 목적.replace(" ", "_").replace("/", "_").replace("\\", "_")

    # 모델 저장 (Keras 3 경고 제거)
    if model_type == "lstm":
        model_path = os.path.join(save_dir, f"{safe_국적}_{safe_목적}_lstm.keras")
        model.save(model_path)  # save_format 인자 제거
    elif model_type == "xgboost":
        model_path = os.path.join(save_dir, f"{safe_국적}_{safe_목적}_xgb.json")
        model.save_model(model_path)
    # 스케일러 저장
    scaler_path = os.path.join(save_dir, f"{safe_국적}_{safe_목적}_scaler.pkl")
    joblib.dump(scaler, scaler_path)


# 시계열 데이터를 입력받아 (window 개월씩 잘라서) LSTM에 넣을 학습 데이터를 만듭니다
# 예: [1,2,3,4,5]에서 window=3일 경우 → X=[1,2,3], [2,3,4], y=4,5
def prepare_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


# 실제값과 예측값을 비교하여 평가 지표를 계산합니다
# MAE: 평균 절대 오차, RMSE: 평균 제곱근 오차, MAPE: 평균 절대 백분율 오차
def compute_metrics(actual, predicted, exclude_covid=True, dates=None):
    """
    MAE, RMSE, MAPE 계산 (코로나 구간 자동 제외 옵션)
    exclude_covid: True면 2020-03~2022-10 구간 제외
    dates: 실제값/예측값에 대응하는 날짜 시리즈
    """
    if exclude_covid and dates is not None:
        mask = ~(
            (dates >= pd.to_datetime("2020-03-01"))
            & (dates <= pd.to_datetime("2022-10-01"))
        )
        actual = np.array(actual)[mask]
        predicted = np.array(predicted)[mask]
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    return mae, rmse, mape


# 예측 정확도(MAPE)에 따라 신뢰도 등급 반환
def get_reliability_level(mape):
    if mape < 15:
        return "매우 높음"
    elif mape < 30:
        return "높음"
    elif mape < 50:
        return "보통 (주의 필요)"
    else:
        return "낮음 (신뢰 어려움)"


# 예측 구간 월별로 최근 3년 같은 월의 실제 입국자수 평균을 구하는 함수
import pandas as pd


def get_seasonal_average(df, 국적, 목적, target_dates, n_years=3):
    seasonal_avg = []
    for d in target_dates:
        month = d.month
        values = []
        for y in range(1, n_years + 1):
            prev = d - pd.DateOffset(years=y)
            val = df[
                (df["국적"] == 국적) & (df["목적"] == 목적) & (df["날짜"] == prev)
            ]["입국자수"]
            if not val.empty:
                values.append(val.values[0])
        if values:
            seasonal_avg.append(np.mean(values))
    else:
        seasonal_avg.append(np.nan)
    return np.array(seasonal_avg)


# --- Flat/0/음수/비정상 예측 자동 보정 함수 (고도화) ---
def robust_postprocess_predictions(pred_values, hist_data, label):
    # 입력값을 그대로 반환 (보정 로직 무력화)
    return pred_values


# --- 예측 구간 트렌드/성장률/계절성 강제 반영 함수 ---
def enforce_trend_on_predictions(
    pred_values, hist_data, future_dates, label, purpose=None
):
    # 입력값을 그대로 반환 (보정 로직 무력화)
    return pred_values


# --- 계절성 보정: 예측값이 비정상적으로 크거나 flat하면 최근 3년 같은 월 평균과 가중 평균 ---
def seasonal_adjustment(pred_values, hist_data, future_dates, n_years=3):
    seasonal_avg = get_seasonal_average(
        hist_data,
        hist_data["국적"].iloc[0],
        hist_data["목적"].iloc[0],
        future_dates,
        n_years,
    )
    abnormal = (
        np.isnan(pred_values)
        | np.isinf(pred_values)
        | (pred_values < 0)
        | (pred_values > 10 * np.nanmax(seasonal_avg[~np.isnan(seasonal_avg)]))
    )
    if np.std(pred_values) < 1e-3 or np.any(abnormal):
        pred_values = 0.5 * pred_values + 0.5 * np.nan_to_num(
            seasonal_avg, nan=np.nanmean(seasonal_avg)
        )
    return pred_values


# --- gap 구간 보간 함수 (rolling mean/계절성/성장률 기반) ---
def interpolate_gap(gap_dates, last_value, first_pred_value, hist_data):
    """
    gap 구간을 rolling mean, 계절성 평균, 성장률 기반으로 자연스럽게 보간
    """
    if len(gap_dates) == 0:
        return np.array([])
    # 최근 3년 같은 월 평균
    months = [d.month for d in gap_dates]
    seasonal_avg = []
    for i, m in enumerate(months):
        vals = hist_data[hist_data["날짜"].dt.month == m]["입국자수"][-36:]
        if len(vals) > 0:
            seasonal_avg.append(np.mean(vals))
        else:
            seasonal_avg.append(np.mean(hist_data["입국자수"]))
    seasonal_avg = np.array(seasonal_avg)
    # rolling mean
    rolling = (
        pd.Series(hist_data["입국자수"])
        .rolling(window=6, min_periods=1)
        .mean()
        .iloc[-1]
    )
    # 성장률 기반 보간 (마지막값에서 예측 시작값까지 등비/등차)
    if last_value > 0 and first_pred_value > 0:
        growth = (first_pred_value / last_value) ** (1 / max(len(gap_dates), 1))
        growth_curve = last_value * (growth ** np.arange(1, len(gap_dates) + 1))
    else:
        growth_curve = np.linspace(last_value, first_pred_value, len(gap_dates))
    # blending: rolling mean, seasonal avg, growth curve
    gap_y = 0.3 * seasonal_avg + 0.3 * rolling + 0.4 * growth_curve
    return gap_y


# --- flat 경향 감지 함수 ---
def is_flat(pred_values):
    std = np.std(pred_values)
    std_log = np.std(np.log1p(pred_values))
    return std < 5 or std_log < 0.015


# --- feature engineering 함수 실제 파이프라인에 반영 ---
def add_features(df):
    """
    데이터프레임에 추가 파생 feature 없이 '입국자수'만 남김
    """
    df = df.copy()
    # 모든 파생 feature 추가 코드를 제거함
    return df


# --- 실제값-예측값 월별 비교 및 flat/비정상 구간 자동 탐지/리포트 함수 ---
def compare_actual_vs_predicted(actual_df, pred_df, label):
    """
    실제값-예측값 월별 비교, flat/비정상 구간 자동 탐지 및 리포트
    - MAPE, flatness, 급락/폭주/이상치 구간 자동 진단
    - 상관계수(Pearson, Spearman) 자동 평가
    - 한글 리포트/경고 메시지 출력
    - 백테스트 신뢰도(MAPE, Pearson, Spearman) CSV 저장
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    # 날짜 기준 merge
    merged = pd.merge(
        actual_df,
        pred_df,
        on=["국적", "목적", "날짜"],
        how="inner",
        suffixes=("_실제", "_예측"),
    )
    if len(merged) == 0:
        # suppress or conditionally print as per len(merged)
        return
    y_true = merged["입국자수"]
    y_pred = merged["예측입국자수"]
    safe_y_pred = np.nan_to_num(y_pred, nan=0)
    safe_y_pred = np.clip(safe_y_pred, a_min=0, a_max=None)
    safe_y_true = np.nan_to_num(y_true, nan=0)
    safe_y_true = np.clip(safe_y_true, a_min=0, a_max=None)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    try:
        pearson_corr, _ = pearsonr(safe_y_true, safe_y_pred)
    except Exception:
        pearson_corr = np.nan
    try:
        spearman_corr, _ = spearmanr(safe_y_true, safe_y_pred)
    except Exception:
        spearman_corr = np.nan
    # flatness(표준편차, 로그 변동폭, max/min 비율, diff 평균)
    std = np.std(safe_y_pred)
    std_log = np.std(np.log1p(safe_y_pred))
    max_min_ratio = np.max(safe_y_pred) / (np.min(safe_y_pred) + 1e-6)
    diff_mean = np.mean(np.abs(np.diff(safe_y_pred)))
    is_flat = (
        (std < 10) or (std_log < 0.02) or (max_min_ratio < 1.15) or (diff_mean < 5)
    )
    # 급락/폭주/이상치 탐지
    is_crash = np.any(safe_y_pred < 0.5 * np.mean(y_true))
    is_explode = np.any(safe_y_pred > 2.0 * np.mean(y_true))
    # 리포트 출력
    # 월별 실제값-예측값 차이 및 백테스트 신뢰도 저장
    merged["오차"] = y_true - y_pred
    nation = merged["국적"].iloc[0] if "국적" in merged.columns else ""
    purpose = merged["목적"].iloc[0] if "목적" in merged.columns else label
    save_backtest_reliability(nation, purpose, mape, pearson_corr, spearman_corr)
    return merged


# 지원 모델 및 목적 리스트 상수 선언
SUPPORTED_MODELS = ["lstm", "xgboost"]  # flatness가 심한 linear 모델은 제외
SUPPORTED_PURPOSES = ["전체", "관광", "유학연수", "상용", "공용"]


def forecast_one(
    df,
    label,
    시작월,
    종료월,
    window=12,
    epochs=100,
    model_type="lstm",
    postprocess=True,
):
    """
    단일 목적/국적/기간에 대해 예측을 수행하는 함수 (모델 저장 없이)
    - df: 입력 데이터프레임 (특정 국적/목적만 필터링된 상태)
    - label: 목적명(문자열)
    - 시작월, 종료월: 예측 시작/종료 (YYYY-MM)
    - window: LSTM/XGB 입력 시퀀스 길이
    - epochs: 학습 epoch 수
    - model_type: 'lstm', 'xgboost', 'linear' 중 선택
    - postprocess: 후처리(보정) 적용 여부 (True=기존대로, False=순수 예측값)
    """
    import numpy as np
    import pandas as pd
    import warnings

    warnings.filterwarnings("ignore")
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None
    from tensorflow.keras.callbacks import EarlyStopping

    # feature engineering 및 숫자형 변환
    df = add_features(df)
    feature_cols = FEATURE_COLS
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)
    df[feature_cols] = df[feature_cols].clip(lower=0)
    # 데이터가 충분하지 않으면 None 반환
    if len(df) < window + 2:
        return None
    # 정규화
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    # 시계열 입력/출력 생성
    X, y = prepare_sequences(scaled, window)
    n_samples = len(X)
    # 예측 대상 월/기간 계산
    start_date = pd.to_datetime(시작월)
    end_date = pd.to_datetime(종료월)
    n_months = (
        (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    )
    # 학습/검증 분할: X, y와 동일한 길이의 날짜 시리즈 생성
    seq_dates = df["날짜"].iloc[window:].reset_index(drop=True)
    covid_start = pd.to_datetime("2020-03-01")
    covid_end = pd.to_datetime("2022-10-01")
    train_mask = ~((seq_dates >= covid_start) & (seq_dates <= covid_end))
    val_mask = (seq_dates >= covid_start) & (seq_dates <= covid_end)
    if train_mask.sum() < 3:
        train_mask[:] = True
    if val_mask.sum() < 1:
        val_mask[:] = False
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    # 모델 학습
    if model_type == "lstm":
        model = Sequential(
            [
                Input(shape=(window, X.shape[2] if X.ndim == 3 else X.shape[1])),
                LSTM(32, return_sequences=False),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        ]
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=epochs,
            batch_size=8,
            verbose=2,
            callbacks=callbacks,
        )
    elif model_type == "xgboost" and xgb is not None:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        X_flat = X_train.reshape(X_train.shape[0], -1)
        model.fit(X_flat, y_train)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    # 미래 예측
    future_features = [scaled[-window:].tolist()]
    preds_scaled = []
    for i in range(n_months):
        needed = window - len(future_features)
        if needed > 0:
            pad = [[0.0] * len(feature_cols)] * needed
            window_features = pad + [f for f in future_features]
        else:
            window_features = future_features[-window:]
        window_features = [
            (
                f
                if isinstance(f, (list, np.ndarray)) and len(f) == len(feature_cols)
                else [0.0] * len(feature_cols)
            )
            for f in window_features
        ]
        X_input = np.array(window_features).reshape(1, window, -1)
        if model_type == "lstm":
            pred = model.predict(X_input, verbose=0)[0, 0]
        else:
            pred = model.predict(X_input.reshape(1, -1))[0]
        preds_scaled.append(pred)
        # 단일 feature ("입국자수")만 사용하는 구조이므로 pred만 사용
        next_feat = [pred]
        future_features.append(next_feat)
    preds_scaled = np.array(preds_scaled)
    dummy = np.zeros((len(preds_scaled), len(feature_cols)))
    dummy[:, 0] = preds_scaled
    preds_raw = scaler.inverse_transform(dummy)[:, 0]  # 후처리 전 순수 예측값
    # 후처리 적용 여부
    if postprocess:
        preds = robust_postprocess_predictions(preds_raw, df, label)
        future_dates = pd.date_range(start=start_date, periods=n_months, freq="MS")
        preds = enforce_trend_on_predictions(
            preds, df, future_dates, label, purpose=label
        )
    else:
        preds = preds_raw
        future_dates = pd.date_range(start=start_date, periods=n_months, freq="MS")
    # 예측값이 모두 0, NaN, 길이 0이어도 무조건 반환 (None 반환 X)
    if preds is None or len(preds) == 0 or np.all(np.isnan(preds)):
        preds = np.full(n_months, 1)  # 최소값 1로 대체
    # 평가 지표 (백테스트)
    backtest_start = start_date - pd.DateOffset(months=n_months)
    backtest_end = start_date - pd.DateOffset(months=1)
    backtest_mask = (df["날짜"] >= backtest_start) & (df["날짜"] <= backtest_end)
    actual = df.loc[backtest_mask, "입국자수"].values
    pred_for_backtest = preds[: len(actual)] if len(actual) > 0 else np.array([])
    if len(actual) > 0 and len(pred_for_backtest) == len(actual):
        mae, rmse, mape = compute_metrics(
            actual, pred_for_backtest, dates=df.loc[backtest_mask, "날짜"]
        )
    else:
        mae = rmse = mape = np.nan
    return (
        future_dates,
        preds,
        label,
        df[["날짜", "입국자수"]],
        (mae, rmse, mape),
        preds_raw,
    )


def select_best_model_and_predict(
    df, label, 시작월, 종료월, window=12, epochs=100, model_types=None, verbose=True
):
    """
    여러 모델을 학습 후, 최적 모델 또는 blending 결과를 반환하는 함수
    model_types: 사용할 모델 리스트 (None이면 SUPPORTED_MODELS 사용)
    """
    if model_types is None:
        model_types = SUPPORTED_MODELS
    results = {}
    for model_type in model_types:
        try:
            future_dates, pred_values, label, hist_df, metrics, _ = forecast_one(
                df, label, 시작월, 종료월, window, epochs, model_type=model_type
            )
            if metrics is not None and pred_values is not None:
                mae, rmse, mape = metrics
                results[model_type] = {
                    "pred": pred_values,
                    "mape": mape,
                    "mae": mae,
                    "rmse": rmse,
                }
        except Exception as e:
            pass

    if not results:
        return None, None, label, None, None

    # blending
    mapes = {k: v["mape"] for k, v in results.items()}
    preds = {k: v["pred"] for k, v in results.items()}
    weights = np.array([1 / (mapes[k] + 1e-6) for k in preds])
    weights = weights / weights.sum()
    blended_pred = sum(w * p for w, p in zip(weights, preds.values()))

    # 최적 모델 선택
    best_model_type = min(mapes, key=mapes.get)
    best_pred = preds[best_model_type]
    best_mape = mapes[best_model_type]

    # (INFO 출력 제거)

    # blending과 best 중 신뢰도 기준으로 선택
    final_pred = blended_pred if best_mape > 20 else best_pred
    # mae/rmse는 best 모델 기준으로 반환
    best_mae = results[best_model_type]["mae"]
    best_rmse = results[best_model_type]["rmse"]
    return (
        future_dates,
        final_pred,
        label,
        df[["날짜", "입국자수"]],
        (best_mae, best_rmse, best_mape),
    )


# --- 미래 예측 신뢰도 자동 추정/리포트용 CSV 저장/불러오기 함수 ---
def save_backtest_reliability(
    nation,
    purpose,
    mape,
    pearson,
    spearman,
    save_path="./results/backtest_reliability.csv",
):
    """
    백테스트 신뢰도(MAPE, Pearson, Spearman 등)를 목적/국적별로 CSV로 저장
    """
    import os
    import pandas as pd

    row = {
        "국적": nation,
        "목적": purpose,
        "MAPE": mape,
        "Pearson": pearson,
        "Spearman": spearman,
    }
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df = df[~((df["국적"] == nation) & (df["목적"] == purpose))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(save_path, index=False)


def load_backtest_reliability(
    nation, purpose, save_path="./results/backtest_reliability.csv"
):
    """
    미래 예측 시 백테스트 신뢰도 정보를 불러옴
    """
    import os
    import pandas as pd

    if not os.path.exists(save_path):
        return None
    df = pd.read_csv(save_path)
    row = df[(df["국적"] == nation) & (df["목적"] == purpose)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# Prophet 예측 함수


def predict_with_prophet(df, start_month, end_month):
    """
    Prophet 기반 월별 예측 함수
    - df: '날짜', '입국자수' 컬럼 포함 DataFrame (특정 국적/목적만 필터링)
    - start_month, end_month: 예측 시작/종료 (YYYY-MM)
    """
    # Prophet 입력 포맷 변환
    df_prophet = (
        df[["날짜", "입국자수"]].rename(columns={"날짜": "ds", "입국자수": "y"}).copy()
    )
    df_prophet = df_prophet.groupby("ds")["y"].sum().reset_index()
    # 모델 학습
    model = Prophet(yearly_seasonality=True)
    model.fit(df_prophet)
    # 예측 구간 생성
    start_date = pd.to_datetime(start_month)
    end_date = pd.to_datetime(end_month)
    n_months = (
        (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    )
    last_date = df_prophet["ds"].max()
    periods = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)
    future = model.make_future_dataframe(periods=max(periods, n_months), freq="MS")
    forecast = model.predict(future)
    # 예측 구간만 추출
    pred_mask = (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)
    pred_dates = forecast.loc[pred_mask, "ds"].values
    pred_values = forecast.loc[pred_mask, "yhat"].values
    # 성능지표(백테스트)
    hist_mask = (df_prophet["ds"] >= start_date - pd.DateOffset(months=n_months)) & (
        df_prophet["ds"] < start_date
    )
    actual = df_prophet.loc[hist_mask, "y"].values
    pred_for_backtest = forecast.loc[
        forecast["ds"].isin(df_prophet.loc[hist_mask, "ds"]), "yhat"
    ].values
    if len(actual) > 0 and len(pred_for_backtest) == len(actual):
        mae = np.mean(np.abs(actual - pred_for_backtest))
        rmse = np.sqrt(np.mean((actual - pred_for_backtest) ** 2))
        mape = (
            np.mean(np.abs((actual - pred_for_backtest) / np.maximum(actual, 1))) * 100
        )
    else:
        mae = rmse = mape = np.nan
    return pred_dates, pred_values, (mae, rmse, mape)


# 메인 함수: 데이터 로딩 → 예측 반복 → 시각화 → 결과 저장
def forecast_visitors(csv_path, 국적, 목적, 시작월, 종료월, window=12, epochs=100):
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

    all_forecast_rows = []
    metrics_rows = []
    prophet_rows = []
    prophet_metrics_rows = []
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

    # suppress cmdstanpy INFO logs before Prophet prediction
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    # 각 목적별로 예측 수행
    for i, (m, d) in enumerate(targets):
        # 기존 LSTM/XGBoost 예측
        result = forecast_one(d, m, 시작월, 종료월, window, epochs)
        if result is None:
            continue
        future_dates, pred_values, label, hist_df, metrics, _ = result
        # Prophet 예측
        prophet_dates, prophet_preds, prophet_metrics = predict_with_prophet(
            d, 시작월, 종료월
        )
        # 결과 저장용 데이터프레임에 추가 (LSTM/XGBoost)
        all_forecast_rows.append(
            pd.DataFrame(
                {
                    "국적": 국적,
                    "목적": label,
                    "날짜": future_dates,
                    "예측입국자수": np.clip(
                        np.nan_to_num(pred_values, nan=0), a_min=0, a_max=None
                    ).astype(int),
                }
            )
        )
        # Prophet 결과 저장
        prophet_rows.append(
            pd.DataFrame(
                {
                    "국적": 국적,
                    "목적": label + "_Prophet",
                    "날짜": prophet_dates,
                    "예측입국자수": np.clip(
                        np.nan_to_num(prophet_preds, nan=0), a_min=0, a_max=None
                    ).astype(int),
                }
            )
        )
        # 평가 지표 저장 (LSTM/XGBoost)
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
        # 평가 지표 저장 (Prophet)
        if prophet_metrics:
            mae, rmse, mape = prophet_metrics
            prophet_metrics_rows.append(
                {
                    "국적": 국적,
                    "목적": label + "_Prophet",
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2),
                    "MAPE": round(mape, 2),
                    "신뢰도": get_reliability_level(mape),
                }
            )

    # 결과 저장용 데이터프레임에 추가 (이미 위에서 수행)

    # 예측 결과 및 평가 지표 CSV로 저장
    file_prefix = f"{국적}_{목적 or '전체'}"
    if all_forecast_rows:
        result_df = pd.concat(all_forecast_rows + prophet_rows, ignore_index=True)
        result_df["예측입국자수"] = np.nan_to_num(
            result_df["예측입국자수"], nan=0
        ).astype(int)
        result_df.to_csv(
            os.path.join(run_folder, file_prefix + ".csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if metrics_rows or prophet_metrics_rows:
        pd.DataFrame(metrics_rows + prophet_metrics_rows).to_csv(
            os.path.join(run_folder, file_prefix + "_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # 시각화: 목적별 예측 결과 그래프 (2x2 그리드, Prophet 포함)
    import matplotlib.dates as mdates

    if all_forecast_rows:
        n = len(all_forecast_rows)
        n_prophet = len(prophet_rows)
        total_n = n + n_prophet
        if total_n == 1:
            fig, axs = plt.subplots(1, 1, figsize=(8, 6))
            axs = [axs]
        elif total_n == 2:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        elif total_n == 3:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        else:
            ncols = 2 if total_n == 4 else 3
            nrows = math.ceil(total_n / ncols)
            fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
            axs = axs.flatten()
        # LSTM/XGBoost 예측 plot
        for i, forecast_df in enumerate(all_forecast_rows):
            m = forecast_df["목적"].iloc[0]
            pred_df = forecast_df.copy()
            hist_data = (
                df[(df["국적"] == 국적) & (df["목적"] == m)].copy().sort_values("날짜")
            )
            start_date = pd.to_datetime(시작월)
            end_date = pd.to_datetime(종료월)
            hist_window_start = start_date - pd.DateOffset(months=13)
            plot_mask = (hist_data["날짜"] >= hist_window_start) & (
                hist_data["날짜"] <= end_date
            )
            plot_data = hist_data[plot_mask].copy()
            pred_mask = (pred_df["날짜"] >= start_date) & (pred_df["날짜"] <= end_date)
            pred_plot = pred_df[pred_mask].copy()
            ax = axs[i]
            ax.set_facecolor("white")
            # 실제 데이터 plot
            if not plot_data.empty:
                ax.plot(
                    plot_data["날짜"],
                    plot_data["입국자수"],
                    color="tab:blue",
                    label="실제값(과거)",
                    linewidth=2,
                    zorder=2,
                )
            # 예측값 plot
            ax.plot(
                pred_plot["날짜"],
                pred_plot["예측입국자수"],
                color="tab:red",
                linestyle="-",
                marker="o",
                markersize=5,
                linewidth=2,
                label="예측값(LSTM/XGB)",
                zorder=3,
            )
            ax.set_title(f"[{국적}] {m}", fontsize=14)
            ax.legend(loc="upper right", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
        # Prophet 예측 plot
        for j, prophet_df in enumerate(prophet_rows):
            idx = n + j
            m = prophet_df["목적"].iloc[0]
            pred_df = prophet_df.copy()
            ax = axs[idx]
            ax.set_facecolor("white")
            ax.plot(
                pred_df["날짜"],
                pred_df["예측입국자수"],
                color="tab:green",
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
        fig.suptitle(
            f"{국적} 목적별 입국자수 예측 결과 (LSTM/XGB/Prophet)", fontsize=18
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fname = os.path.join(run_folder, f"{국적}_전체목적_2x2plot.png")
        fig.savefig(fname)
        plt.show()

    # 예측 결과와 실제값 비교 (리포트 출력 제거)
    if all_forecast_rows:
        for i, forecast_df in enumerate(all_forecast_rows):
            m = forecast_df["목적"].iloc[0]
            pred_df = forecast_df.copy()
            hist_data = (
                df[(df["국적"] == 국적) & (df["목적"] == m)].copy().sort_values("날짜")
            )
            pred_start_date = pred_df["날짜"].iloc[0]
            pred_end_date = pred_df["날짜"].iloc[-1]
            compare_actual_vs_predicted(hist_data, pred_df, m)

    # 미래 예측 신뢰도 자동 리포트 (출력 제거)


def detailed_analysis_report(actual_df, pred_df, label, save_path=None):
    """
    목적/국가/기간별 오차, 상관계수, flatness, 이상치, 성장률 등 상세 리포트 및 CSV 저장
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    merged = pd.merge(
        actual_df,
        pred_df,
        on=["국적", "목적", "날짜"],
        how="inner",
        suffixes=("_실제", "_예측"),
    )
    if len(merged) == 0:
        return
    y_true = merged["입국자수"]
    y_pred = merged["예측입국자수"]
    safe_y_pred = np.nan_to_num(y_pred, nan=0)
    safe_y_pred = np.clip(safe_y_pred, a_min=0, a_max=None)
    safe_y_true = np.nan_to_num(y_true, nan=0)
    safe_y_true = np.clip(safe_y_true, a_min=0, a_max=None)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    try:
        pearson_corr, _ = pearsonr(safe_y_true, safe_y_pred)
    except Exception:
        pearson_corr = np.nan
    try:
        spearman_corr, _ = spearmanr(safe_y_true, safe_y_pred)
    except Exception:
        spearman_corr = np.nan
    std = np.std(safe_y_pred)
    std_log = np.std(np.log1p(safe_y_pred))
    max_min_ratio = np.max(safe_y_pred) / (np.min(safe_y_pred) + 1e-6)
    diff_mean = np.mean(np.abs(np.diff(safe_y_pred)))
    is_flat = (
        (std < 10) or (std_log < 0.02) or (max_min_ratio < 1.15) or (diff_mean < 5)
    )
    is_crash = np.any(safe_y_pred < 0.5 * np.mean(y_true))
    is_explode = np.any(safe_y_pred > 2.0 * np.mean(y_true))
    # 성장률(예측구간 첫/끝)
    if len(y_pred) > 1:
        growth = (y_pred.iloc[-1] - y_pred.iloc[0]) / max(y_pred.iloc[0], 1)
    else:
        growth = np.nan
    merged["오차"] = y_true - y_pred
    if save_path:
        merged.to_csv(save_path, index=False)
    return merged


if __name__ == "__main__":
    import re

    def valid_yyyymm(date_str):
        return bool(re.match(r"^\d{4}-\d{2}$", date_str))

    # 항상 절대경로로 통일
    csv_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../data_preprocessing/data/processed/외국인입국자_전처리완료_딥러닝용.csv",
        )
    )

    # 데이터에서 국적/목적 리스트 추출
    df_tmp = pd.read_csv(csv_path)
    available_nations = sorted(df_tmp["국적"].dropna().unique())
    available_purposes = sorted(df_tmp["목적"].dropna().unique())

    # 기본값 설정
    default_nation = "일본"
    default_purpose = "관광"
    default_start_month = "2025-06"
    default_end_month = "2026-05"

    # 국적 입력 (존재하는 값만 허용)
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

    # 목적 입력 (존재하는 값만 허용, 전체는 None)
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
        )
        종료월 = (
            input(
                f"예측 종료월을 입력하세요 (예: 2026-05) [엔터 : {default_end_month}]: "
            ).strip()
            or default_end_month
        )
        if not valid_yyyymm(시작월) or not valid_yyyymm(종료월):
            print("[경고] 올바른 형식(YYYY-MM)으로 입력해주세요.")
            continue
        if 시작월 >= 종료월:
            print("[경고] 예측 시작월은 종료월보다 이전이어야 합니다.")
            continue
        break

    forecast_visitors(
        csv_path=csv_path, 국적=국적, 목적=목적, 시작월=시작월, 종료월=종료월
    )
