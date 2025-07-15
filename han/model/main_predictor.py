import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import matplotlib as mpl
import os

from prophet_predictor import predict_prophet
from xgboost_predictor import predict_xgb
from lstm_predictor import predict_lstm

# ----- 폰트 세팅 -----
def set_korean_font():
    font_paths = [
        "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf"
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            mpl.rc('font', family=fm.FontProperties(fname=fp).get_name())
            mpl.rcParams['axes.unicode_minus'] = False
            print(f"폰트 설정: {fp}")
            return
set_korean_font()

df = pd.read_csv('../data/외국인입국자_전처리완료_딥러닝용.csv')

def normalize_korean_input(input_str):
    return re.sub(r"\s+", "", str(input_str).strip())

def valid_country(input_str, df):
    input_str = normalize_korean_input(input_str)
    if input_str == "": return None
    countries = df['국적'].unique()
    korean_countries = [c for c in countries if re.fullmatch(r"[가-힣]+", c)]
    if input_str in korean_countries: return input_str
    return False

def valid_purpose(input_str, df):
    input_str = normalize_korean_input(input_str)
    if input_str == "": return None
    purposes = df['목적'].unique()
    if input_str in purposes: return input_str
    return False

def valid_ym(input_str, df):
    input_str = normalize_korean_input(input_str)
    try:
        if len(input_str) != 6: return False
        y, m = int(input_str[:4]), int(input_str[4:])
        if not (2005 <= y <= 2030 and 1 <= m <= 12): return False
        return input_str
    except: return False

def get_user_input():
    print("\n[입력 안내]")
    print("▶ 국가명은 반드시 한글로 입력 (예: 일본, 중국, 미국). 띄어쓰기는 자동 제거됩니다.")
    print("▶ 목적은 데이터 기준 한글명(예: 관광, 상용 등). 띄어쓰기는 자동 제거됩니다.")
    print("▶ 예측 날짜는 200501~203012 범위의 6자리(예: 202501)로 입력.")
    while True:
        c = input("\n국가 입력(한글, 없으면 Enter): ")
        country = valid_country(c, df)
        if country is not False: break
        print("❗존재하지 않거나 한글이 아닌 국가입니다. 다시 입력하세요.")
    while True:
        p = input("목적 입력(한글, 없으면 Enter): ")
        purpose = valid_purpose(p, df)
        if purpose is not False: break
        print("❗존재하지 않는 목적입니다. 다시 입력하세요.")
    while True:
        d = input("예측 날짜 입력(YYYYMM): ")
        date = valid_ym(d, df)
        if date: break
        print("❗날짜는 200501~203012(6자리, 예: 202501)로 입력하세요.")
    return country, purpose, date

def interpret_mape(mape):
    if mape == "N/A" or mape is None: return "N/A"
    if mape < 0.1: return "✅ 매우 정확"
    elif mape < 0.2: return "▲ 준수"
    else: return "❗오차 큼"

def plot_all(df, country, purpose, date, result_prophet, result_xgb, result_lstm):
    plt.figure(figsize=(18,7))
    plt.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-01'),
                color='pink', alpha=0.13, label="코로나 구간(2020~2022)")

    real_df = result_prophet['실제_시계열']
    plt.plot(real_df['ds'], real_df['입국자수'], color='black', linewidth=2, label='실제 입국자수')

    # Prophet 전체/장래
    prophet_df = result_prophet['예측_시계열']
    plt.plot(prophet_df['ds'], prophet_df['yhat'], color='blue', linewidth=2, label='Prophet 예측(전체)')
    future = prophet_df[prophet_df['ds'] > real_df['ds'].max()]
    if not future.empty:
        plt.plot(future['ds'], future['yhat'], color='orange', linestyle='--', linewidth=2, label='Prophet 예측(장래)')

    # XGBoost 예측 시계열 (실측 종료점부터 예측점까지)
    if result_xgb.get('예측_시계열') is not None:
        xgb_pred_df = result_xgb['예측_시계열']
        if not xgb_pred_df.empty:
            plt.plot(xgb_pred_df['ds'], xgb_pred_df['입국자수'],
                     color='green', linewidth=2, linestyle='-', label='XGBoost 예측 시계열')
            plt.scatter([xgb_pred_df['ds'].iloc[-1]], [xgb_pred_df['입국자수'].iloc[-1]],
                        color='green', marker='D', s=130, label=None)
            plt.text(xgb_pred_df['ds'].iloc[-1], xgb_pred_df['입국자수'].iloc[-1],
                     f"{int(xgb_pred_df['입국자수'].iloc[-1]):,}", color='green', fontsize=14, fontweight='bold', va='bottom')

    # LSTM 예측 시계열
    if result_lstm.get('예측_시계열') is not None:
        lstm_pred_df = result_lstm['예측_시계열']
        last_real_ds = real_df['ds'].max()
        lstm_pred_zone = lstm_pred_df[lstm_pred_df['ds'] >= last_real_ds]
        if not lstm_pred_zone.empty:
            plt.plot(lstm_pred_zone['ds'], lstm_pred_zone['입국자수'],
                     color='purple', linewidth=2, linestyle=':', label='LSTM 예측(연장)')
            plt.scatter([lstm_pred_zone['ds'].iloc[-1]], [lstm_pred_zone['입국자수'].iloc[-1]],
                        color='purple', marker='^', s=130, label=None)
            plt.text(lstm_pred_zone['ds'].iloc[-1], lstm_pred_zone['입국자수'].iloc[-1],
                     f"{int(lstm_pred_zone['입국자수'].iloc[-1]):,}", color='purple', fontsize=14, fontweight='bold', va='bottom')

    # Prophet 타겟
    if date and result_prophet.get('예측값') is not None:
        target_x = pd.to_datetime(date, format='%Y%m')
        plt.scatter([target_x], [result_prophet['예측값']], color='red', s=150, label='예측 Target', zorder=10)
        plt.text(target_x, result_prophet['예측값'], f"{int(result_prophet['예측값']):,}", color='red',
                 fontsize=16, fontweight='bold', va='bottom')

    # X축: 1년 단위
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Y축: 10만 단위 or 5천 단위
    y_max = max(real_df['입국자수'].max(),
                prophet_df['yhat'].max(),
                result_xgb.get('예측값', 0) or 0,
                result_lstm.get('예측값', 0) or 0)
    if y_max > 100_000:
        step = 100_000
        label_fmt = '{:,.0f}'
    else:
        step = 5000
        label_fmt = '{:,.0f}'
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: label_fmt.format(x)))

    plt.xlabel("연도")
    plt.ylabel("입국자 수")
    plt.legend(fontsize=13)
    plt.title(f"외국인 입국자수 예측(Prophet, XGBoost, LSTM)\n{country or '전체'} - {purpose or '전체'} - {date} 예측", fontsize=21)
    plt.tight_layout()
    plt.show()

# ========== main ==========

if __name__ == "__main__":
    print("⚡외국인 입국자 예측 프로그램")
    country, purpose, date = get_user_input()

    result_prophet = predict_prophet(df, country, purpose, date)
    result_xgb = predict_xgb(df, country, purpose, date)
    result_lstm = predict_lstm(df, country, purpose, date)

    plot_all(df, country, purpose, date, result_prophet, result_xgb, result_lstm)

    print(f"\n[Prophet] 예측값: {result_prophet.get('예측값', 'N/A'):,}")
    print("  └ RMSE: N/A, MAPE: N/A, R2: N/A → Prophet는 미래 오차 없음")

    print(f"[XGBoost] 예측값: {result_xgb.get('예측값', 'N/A'):,}")
    rmse = result_xgb.get('RMSE', 'N/A')
    mape = result_xgb.get('MAPE', 'N/A')
    r2 = result_xgb.get('R2', 'N/A')
    print(f"  └ RMSE: {rmse if isinstance(rmse,float) else rmse}, MAPE: {mape if isinstance(mape,float) else mape}, R2: {r2 if isinstance(r2,float) else r2} → {interpret_mape(mape) if mape!='N/A' else ''}")

    print(f"\n[LSTM] 예측값: {result_lstm.get('예측값', 'N/A'):,}")
    rmse = result_lstm.get('RMSE', 'N/A')
    mae = result_lstm.get('MAE', 'N/A')
    mape = result_lstm.get('MAPE', 'N/A')
    print(f"  └ RMSE: {rmse if isinstance(rmse,float) else rmse}, MAE: {mae if isinstance(mae,float) else mae}, MAPE: {mape if isinstance(mape,float) else mape} → {interpret_mape(mape) if mape!='N/A' else ''}")
