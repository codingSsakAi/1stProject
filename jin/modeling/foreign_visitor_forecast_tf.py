# foreign_visitor_forecast_tf.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import platform
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from adjustText import adjust_text

# Mac에서 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 시퀀스 준비

def prepare_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# 지표 계산

def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    return mae, rmse, mape

# 개별 목적 예측

def forecast_one(df, label, 시작월, 종료월, window=12, epochs=100):
    df = df.sort_values('날짜')
    if df.empty or '입국자수' not in df.columns or df['입국자수'].dropna().empty:
        print(f"데이터 없음: {label} — 예측 생략")
        return None, None, label, None, None

    series = df['입국자수'].values.astype(np.float32)
    if len(series) == 0:
        print(f"시계열 데이터 없음: {label} — 예측 생략")
        return None, None, label, None, None

    max_val = series.max()
    series_norm = series / max_val
    X, y = prepare_sequences(series_norm, window)
    if len(X) == 0:
        print(f"학습용 시퀀스 없음: {label} — 예측 생략")
        return None, None, label, None, None
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(64, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X, y, epochs=epochs, verbose=2)

    n_months = (pd.to_datetime(종료월) - pd.to_datetime(시작월)).days // 30 + 1
    future_input = series_norm[-window:].tolist()
    predictions = []
    for _ in range(n_months):
        input_seq = np.array(future_input[-window:]).reshape((1, window, 1))
        pred = model.predict(input_seq, verbose=0)[0, 0]
        predictions.append(pred)
        future_input.append(pred)

    future_dates = [df['날짜'].max() + pd.DateOffset(months=i + 1) for i in range(n_months)]
    pred_values = np.array(predictions) * max_val

    pred_train = model.predict(X, verbose=2).flatten() * max_val
    true_train = y * max_val
    mae, rmse, mape = compute_metrics(true_train, pred_train)

    return future_dates, pred_values, label, df[['날짜', '입국자수']], (mae, rmse, mape)

# 메인 함수

def forecast_visitors(csv_path, 국적, 목적, 시작월, 종료월, window=12, epochs=100):
    BASE_DIR = "../data_preprocessing/data/processed/"
    RESULT_DIR = "./results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(RESULT_DIR, f"forecast_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    df = pd.read_csv(os.path.join(BASE_DIR, csv_path))
    if '연도' in df.columns and '월' in df.columns:
        df['날짜'] = pd.to_datetime(df[['연도', '월']].rename(columns={'연도': 'year', '월': 'month'}).assign(day=1))
    elif '연' in df.columns and '월' in df.columns:
        df['날짜'] = pd.to_datetime(df[['연', '월']].rename(columns={'연': 'year', '월': 'month'}).assign(day=1))
    else:
        raise ValueError("CSV에 연도/월 정보가 없습니다.")

    all_forecast_rows = []
    metrics_rows = []
    목적순서 = ['관광', '유학연수', '공용', '상용']
    colors = {'관광': 'tab:blue', '유학연수': 'tab:green', '공용': 'tab:orange', '상용': 'tab:red'}

    if 목적:
        targets = [(목적, df[(df['국적'] == 국적) & (df['목적'] == 목적)])]
    else:
        targets = [(m, df[(df['국적'] == 국적) & (df['목적'] == m)]) for m in 목적순서]

    fig, axs = plt.subplots(nrows=len(targets)+1, figsize=(12, 5*(len(targets)+1)), sharex=True)
    전체_ax = axs[0]
    전체_ax.set_title(f"{국적} 입국자수 예측 ({시작월} ~ {종료월})", fontsize=13, fontweight='bold')
    전체_ax.axvspan(datetime(2020, 3, 1), datetime(2022, 10, 1), color='lightcoral', alpha=0.3, label='코로나 기간')
    전체_ax.axvspan(pd.to_datetime(시작월), pd.to_datetime(종료월), color='lavender', alpha=0.3, label='예측 구간')

    보조_ax = axs[0].twinx()

    # 보조축 y-limit 동적으로 설정
    보조_max = 0
    for m, d in targets:
        max_val = d['입국자수'].max()
        if max_val > 보조_max:
            보조_max = max_val
    보조_ax.set_ylim(0, 10000)

    for i, ax in enumerate(axs):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_xlim([datetime(2020, 1, 1), pd.to_datetime(종료월)])
        ax.axvline(pd.to_datetime(시작월), color='gray', linestyle='--')
        ax.tick_params(axis='both', labelsize=9)

    for i, (m, d) in enumerate(targets):
        future_dates, pred_values, label, hist_df, metrics = forecast_one(d, m, 시작월, 종료월, window, epochs)
        if future_dates is None:
            continue

        max_val = hist_df['입국자수'].max()
        if m == '관광':
            axs[0].plot(hist_df['날짜'], hist_df['입국자수'], label=f"{m} 실측", color=colors[m], alpha=0.8, linewidth=2)
        else:
            보조_ax.plot(hist_df['날짜'], hist_df['입국자수'], label=f"{m} 실측", color=colors[m], alpha=0.8, linewidth=2, linestyle='--', marker='o')
        axs[i+1].plot(hist_df['날짜'], hist_df['입국자수'], color='gray', alpha=0.3, label='실측', linewidth=2)
        axs[i+1].plot(future_dates, pred_values, linestyle='--', marker='o', color=colors[m], label='예측', linewidth=3)

        year_data = hist_df[(hist_df['날짜'] >= '2020-01') & (hist_df['날짜'] <= '2025-05')]
        if not year_data.empty:
            max_row = year_data.loc[year_data['입국자수'].idxmax()]
            min_row = year_data.loc[year_data['입국자수'].idxmin()]
            for row in [max_row, min_row]:
                axs[i+1].scatter(row['날짜'], row['입국자수'], color='black', s=40)
                axs[i+1].annotate(f"{int(row['입국자수'])}", xy=(row['날짜'], row['입국자수']),
                                  xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

        texts = []
        for j, (x, y) in enumerate(zip(future_dates, pred_values)):
            offset = 10 if j % 2 == 0 else -15
            txt = axs[i+1].text(x, y + offset, f"{int(y)}", fontsize=9, ha='center',
                                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', ec='red', lw=1))
            texts.append(txt)
        adjust_text(texts, ax=axs[i+1], arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        axs[i+1].set_ylabel(m, fontsize=10)
        axs[i+1].legend(fontsize=8, loc='upper left')

        all_forecast_rows.append(pd.DataFrame({
            '국적': 국적, '목적': label,
            '날짜': future_dates,
            '예측입국자수': pred_values.astype(int)
        }))

        if metrics:
            mae, rmse, mape = metrics
            metrics_rows.append({
                '국적': 국적, '목적': label, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)
            })

    보조_ax.set_ylabel("보조 목적 입국자수", fontsize=9)

    axs[-1].set_xlabel("날짜", fontsize=10)
    axs[0].legend(loc='upper left', fontsize=9, frameon=True)
    plt.subplots_adjust(hspace=0.4)

    total_forecast = sum([df['예측입국자수'].sum() for df in all_forecast_rows])
    summary = f"{국적} 목적별 입국자 예측 리포트\n기간: {시작월} ~ {종료월} | 총 예측: {int(total_forecast):,}명 | 주요: {목적 or '전체'}"
    fig.suptitle(summary, fontsize=15, fontweight='bold', y=1.03)

    plt.tight_layout()
    file_prefix = f"visitor_forecast_{timestamp}"
    plt.savefig(os.path.join(run_folder, file_prefix + ".png"), dpi=300, bbox_inches='tight')
    plt.show()

    if all_forecast_rows:
        result_df = pd.concat(all_forecast_rows, ignore_index=True)
        result_df.to_csv(os.path.join(run_folder, file_prefix + ".csv"), index=False, encoding='utf-8-sig')
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(os.path.join(run_folder, file_prefix + "_metrics.csv"), index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    print("외국인 입국자 예측 시스템")
    default_csv = "외국인입국자_전처리완료_딥러닝용.csv"
    csv_path = input(f"입력 데이터 파일명(기본값: {default_csv}): ").strip() or default_csv
    국적 = input("국적을 입력하세요 (예: 일본): ")
    목적 = input("목적을 입력하세요 (전체일 경우 엔터): ") or None
    시작월 = input("예측 시작월을 입력하세요 (예: 2025-06): ")
    종료월 = input("예측 종료월을 입력하세요 (예: 2026-05): ")

    forecast_visitors(
        csv_path=csv_path,
        국적=국적,
        목적=목적,
        시작월=시작월,
        종료월=종료월
    )
