# model.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings, re, os, joblib
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

warnings.filterwarnings("ignore")


def normalize(text):
    return re.sub(r"[^0-9a-z가-힣]", "", str(text).lower().strip())


def run_forecast(국적_입력, 목적_입력, 예측연도, 예측월리스트):
    파일경로 = "./data/외국인입국자_전처리완료_딥러닝용.csv"
    df = pd.read_csv(파일경로)
    df["년월"] = pd.to_datetime(
        df["연도"].astype(str) + "-" + df["월"].astype(str).str.zfill(2)
    )
    df["입국자수"] = df["입국자수"].astype(int)
    df["국적_norm"] = df["국적"].apply(normalize)
    df["목적_norm"] = df["목적"].apply(normalize)
    국적_norm = normalize(국적_입력)
    목적_norm = normalize(목적_입력)

    유효 = df.groupby(["국적_norm", "목적_norm"]).size().reset_index(name="count")
    유효 = 유효[유효["count"] >= 24]
    if 국적_norm:
        유효 = 유효[유효["국적_norm"] == 국적_norm]
    if 목적_norm:
        유효 = 유효[유효["목적_norm"] == 목적_norm]
    if 유효.empty:
        return [{"error": "⛔ 해당 조건에 맞는 데이터가 없습니다."}]

    results = []
    for _, row in 유효.iterrows():
        국적 = df[df["국적_norm"] == row["국적_norm"]]["국적"].iloc[0]
        목적 = df[df["목적_norm"] == row["목적_norm"]]["목적"].iloc[0]
        data = df[
            (df["국적_norm"] == row["국적_norm"])
            & (df["목적_norm"] == row["목적_norm"])
        ].copy()
        feature_cols = ["연도편차", "월", "월_cos", "월_sin", "성수기여부"]
        if 목적 == "관광":
            feature_cols += ["방학여부"]
        elif 목적 == "유학연수":
            feature_cols += ["학기여부"]

        data["연도"] = data["년월"].dt.year
        data["월"] = data["년월"].dt.month
        data["연도편차"] = data["연도"] - data["연도"].min()
        data["월_cos"] = np.cos(2 * np.pi * data["월"] / 12)
        data["월_sin"] = np.sin(2 * np.pi * data["월"] / 12)
        data["성수기여부"] = data["월"].isin([2, 7, 8]).astype(int)
        data["방학여부"] = data["월"].isin([1, 7, 8, 12]).astype(int)
        data["학기여부"] = data["월"].isin([3, 9]).astype(int)

        use_log = 목적 in ["공용", "관광", "상용"]
        gru_epochs = 250 if 목적 == "유학연수" else (200 if 목적 == "공용" else 150)
        X = data[feature_cols]
        y_raw = data["입국자수"]
        y = np.log1p(y_raw) if use_log else y_raw

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, y_train = X_scaled[:-6], y[:-6]
        X_all = scaler.transform(X)

        param_dict = {
            "공용": {"n_estimators": 400, "max_depth": 3},
            "관광": {"n_estimators": 350, "max_depth": 4},
            "상용": {"n_estimators": 250, "max_depth": 3},
            "유학연수": {"n_estimators": 450, "max_depth": 4},
        }
        params = param_dict.get(목적, {"n_estimators": 250, "max_depth": 2})

        xgb = XGBRegressor(**params)
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_train)
        residuals = y_train.values - pred_xgb

        model_rnn, scaler_resid = None, None
        if len(residuals) >= 18:
            seq_len = 18
            scaler_resid = StandardScaler()
            resid_scaled = scaler_resid.fit_transform(
                residuals.reshape(-1, 1)
            ).flatten()
            X_seq = np.array(
                [
                    resid_scaled[i : i + seq_len]
                    for i in range(len(resid_scaled) - seq_len)
                ]
            )
            y_seq = np.array(
                [resid_scaled[i + seq_len] for i in range(len(resid_scaled) - seq_len)]
            )
            X_seq = X_seq.reshape(-1, seq_len, 1)

            model_rnn = Sequential()
            model_rnn.add(GRU(64, return_sequences=True, input_shape=(seq_len, 1)))
            model_rnn.add(Dropout(0.2))
            model_rnn.add(GRU(32))
            model_rnn.add(Dense(1))
            model_rnn.compile(optimizer="adam", loss="mse")
            model_rnn.fit(X_seq, y_seq, epochs=gru_epochs, verbose=0)
            # 예측에만 사용

        future = pd.date_range(f"{예측연도}-01-01", f"{예측연도}-12-01", freq="MS")
        future = pd.DataFrame({"년월": future})
        future["연도"] = future["년월"].dt.year
        future["월"] = future["년월"].dt.month
        future = future[future["월"].isin(예측월리스트)].copy()
        future["연도편차"] = future["연도"] - data["연도"].min()
        future["월_cos"] = np.cos(2 * np.pi * future["월"] / 12)
        future["월_sin"] = np.sin(2 * np.pi * future["월"] / 12)
        future["성수기여부"] = future["월"].isin([2, 7, 8]).astype(int)
        future["방학여부"] = future["월"].isin([1, 7, 8, 12]).astype(int)
        future["학기여부"] = future["월"].isin([3, 9]).astype(int)
        X_future = scaler.transform(future[feature_cols])
        pred_future = xgb.predict(X_future)

        if model_rnn:
            resid_scaled = scaler_resid.transform(residuals.reshape(-1, 1)).flatten()
            residual_seqs = [
                resid_scaled[-seq_len - i : -i] if i != 0 else resid_scaled[-seq_len:]
                for i in range(len(future))
            ]
            residual_seqs = np.array(residual_seqs).reshape(len(future), seq_len, 1)
            residual_preds_scaled = model_rnn.predict(
                residual_seqs, verbose=0
            ).flatten()
            residual_preds = scaler_resid.inverse_transform(
                residual_preds_scaled.reshape(-1, 1)
            ).flatten()
            pred_total = pred_future + residual_preds
        else:
            pred_total = pred_future

        if use_log:
            pred_total = np.expm1(pred_total)
            전체예측 = np.expm1(xgb.predict(X_all))
        else:
            전체예측 = xgb.predict(X_all)

        future["예측입국자수"] = pred_total

        def safe_mape(y_true, y_pred):
            mask = y_true != 0
            return mean_absolute_percentage_error(y_true[mask], y_pred[mask])

        r2 = r2_score(y_raw, 전체예측)
        mape = safe_mape(y_raw.values, 전체예측)
        신뢰도 = max(0, 100 - mape * 100)

        results.append(
            {
                "country": 국적,
                "purpose": 목적,
                "yms": future["년월"].dt.strftime("%Y-%m").tolist(),
                "values": future["예측입국자수"].astype(int).tolist(),
                "hist_yms": data["년월"].dt.strftime("%Y-%m").tolist(),
                "hist_values": y_raw.astype(int).tolist(),
                "r2": round(r2, 4),
                "mape": round(mape * 100, 2),
                "confidence": round(신뢰도, 1),
            }
        )
    return results
