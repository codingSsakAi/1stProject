# %%
##############################################################################################################

# %%
import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("../1stProject/data/목적별_국적별_입국.csv", encoding='cp949')

# "소 계" 포함된 행 제거
df_cleaned = df[~df.apply(lambda row: row.astype(str).str.contains("소 계").any(), axis=1)]

# 결과 저장
df_cleaned.to_csv("../1stProject/data/목적별_국적별_입국_소계제거.csv", index=False, encoding='cp949')

# %%
import pandas as pd

df = pd.read_excel("../1stProject/data/목적별 국적별 입국_250709084025.xls", engine="xlrd")
df.to_csv("../1stProject/data/목적별_국적별_입국.csv", index=False, encoding="cp949")


# %%
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import platform

# [1] 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# [2] 데이터 로딩 및 전처리
df = pd.read_csv('../1stProject/data/목적별_국적별_입국_소계제거.csv', encoding='cp949')
df = df.melt(id_vars=["국적", "목적"], var_name="월", value_name="입국자수")

# ⛔️ "입국자수"에 숫자가 아닌 값 필터링 제거
df = df[~df["입국자수"].str.contains("명|합계|소계|인원", na=False)]

# 날짜 및 수치형 변환
df["월"] = pd.to_datetime(df["월"], format="%Y년%m월")
df["입국자수"] = df["입국자수"].astype(str).str.replace(",", "").astype(float)

# 관광 목적만 필터링
df = df[df["목적"] == "관광"].sort_values("월").reset_index(drop=True)

# [3] Prophet용 데이터
df_prophet = df.groupby("월")["입국자수"].sum().reset_index()
df_prophet.columns = ["ds", "y"]

# [4] Prophet 모델 훈련 및 예측
prophet = Prophet(yearly_seasonality=True)
prophet.fit(df_prophet)
future = prophet.make_future_dataframe(periods=6, freq='MS')
forecast = prophet.predict(future)

# [5] XGBoost용 피처 생성
df_feat = df_prophet.copy()
df_feat["연도"] = df_feat["ds"].dt.year
df_feat["월"] = df_feat["ds"].dt.month
df_feat["계절"] = df_feat["월"] % 12 // 3
df_feat["전월입국자수"] = df_feat["y"].shift(1)
df_feat["전년동월입국자수"] = df_feat["y"].shift(12)
df_feat["전월증감률"] = df_feat["y"].pct_change().shift(1)
df_feat["전년증감률"] = (df_feat["y"] - df_feat["y"].shift(12)) / df_feat["y"].shift(12)
df_feat["이동평균"] = df_feat["y"].rolling(window=3).mean().shift(1)
df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()

# [6] XGBoost 모델 학습 및 예측
features = ["연도", "월", "계절", "전월입국자수", "전년동월입국자수", "전월증감률", "전년증감률", "이동평균"]
X = df_feat[features]
y = df_feat["y"]
X_train, X_test = X[:-6], X[-6:]
y_train, y_test = y[:-6], y[-6:]

model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, None)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# [7] 시각화
plt.figure(figsize=(14, 6))
plt.plot(df_feat["ds"], df_feat["y"], label="실제 입국자수", color='blue')
plt.plot(forecast["ds"], forecast["yhat"], label="Prophet 예측", color='green', linestyle='--')
plt.plot(df_feat["ds"].iloc[-6:], y_pred, label="XGBoost 예측", color='orange', marker='o')
plt.axvline(df_feat["ds"].iloc[-7], color='gray', linestyle='--', label="예측 시작")
plt.title(f"입국자 수 예측 (Prophet + XGBoost 병렬)\nXGBoost RMSE: {rmse:.0f}")
plt.xlabel("월")
plt.ylabel("입국자 수")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# %pip install prophet xgboost

# %%
# GPU 이용
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 음수 기호 깨짐 방지

# 문자열 정규화 함수
def normalize(text):
    return text.replace(" ", "").lower()

# [1] 사용자 입력 및 유사 매칭
선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

# [2] 데이터 로딩 및 전처리
df = pd.read_csv('../1stProject/data/목적별_국적별_입국_소계제거.csv', encoding='cp949')
df = df.melt(id_vars=["국적", "목적"], var_name="월", value_name="입국자수")
df["월"] = pd.to_datetime(df["월"], format="%Y년%m월")
df["입국자수"] = pd.to_numeric(df["입국자수"].astype(str).str.replace(",", "", regex=False), errors="coerce")

# [3] 유사 국적/목적 자동 완성
if 선택_국적:
    match = [nat for nat in df["국적"].dropna().unique()
             if isinstance(nat, str) and normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in df["목적"].dropna().unique()
             if isinstance(pur, str) and normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

# [4] 예측 대상 필터링
targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
valid_targets = targets[targets['count'] > 24]
if 선택_국적:
    valid_targets = valid_targets[valid_targets["국적"] == 선택_국적]
if 선택_목적:
    valid_targets = valid_targets[valid_targets["목적"] == 선택_목적]

# [5] 예측 루프
for idx, row in valid_targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values("월").reset_index(drop=True)

    if len(df_filtered) < 30:
        continue
    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    # 파생 변수 생성
    df_filtered["연도"] = df_filtered["월"].dt.year
    df_filtered["월_숫자"] = df_filtered["월"].dt.month
    df_filtered["계절"] = df_filtered["월"].dt.month % 12 // 3
    df_filtered["전월입국자수"] = df_filtered["입국자수"].shift(1)
    df_filtered["전년동월입국자수"] = df_filtered["입국자수"].shift(12)
    df_filtered["전월증감률"] = df_filtered["입국자수"].pct_change().shift(1)
    df_filtered["전년증감률"] = (df_filtered["입국자수"] - df_filtered["입국자수"].shift(12)) / df_filtered["입국자수"].shift(12)
    df_filtered["이동평균3"] = df_filtered["입국자수"].rolling(window=3).mean().shift(1)
    df_filtered["이동평균6"] = df_filtered["입국자수"].rolling(window=6).mean().shift(1)
    df_filtered["전월_차이"] = df_filtered["입국자수"] - df_filtered["전월입국자수"]
    df_filtered["전년_차이"] = df_filtered["입국자수"] - df_filtered["전년동월입국자수"]
    df_filtered["연중누적합"] = df_filtered.groupby("연도")["입국자수"].cumsum()
    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if df_filtered.empty:
        continue

    # 특징 및 타겟
    features = ["연도", "월_숫자", "계절", "전월입국자수", "전년동월입국자수",
                "전월증감률", "전년증감률", "이동평균3", "이동평균6",
                "전월_차이", "전년_차이", "연중누적합"]
    X = df_filtered[features]
    y = df_filtered["입국자수"]
    dates = df_filtered["월"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 학습/테스트 분리
    X_train, X_test = X_scaled[:-6], X_scaled[-6:]
    y_train, y_test = y[:-6], y[-6:]
    dates_train, dates_test = dates[:-6], dates[-6:]

    if len(X_test) == 0:
        continue

    # GPU XGBoost 모델 및 튜닝
    xgb = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=42)
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10,
                                 cv=TimeSeriesSplit(n_splits=3),
                                 scoring='neg_root_mean_squared_error', random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = np.clip(best_model.predict(X_test), 0, None)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(dates_train, y_train, label="실제값 (학습)", color='blue')
    plt.plot(dates_test, y_test, label="실제값 (테스트)", color='red', linestyle='--')
    plt.plot(dates_test, y_pred, label="예측값 (XGBoost)", color='orange', marker='o')
    plt.title(f"📊 {국적}/{목적} - RMSE: {rmse:.0f}, MAPE: {mape:.1f}%")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Prophet 시계열 예측
    df_prophet = df_filtered[["월", "입국자수"]].rename(columns={"월": "ds", "입국자수": "y"})
    prophet = Prophet(yearly_seasonality=True)
    try:
        prophet.add_country_holidays(country_name='KR')
        prophet.fit(df_prophet)
        future = prophet.make_future_dataframe(periods=6, freq='MS')
        forecast = prophet.predict(future)

        plt.figure(figsize=(14, 5))
        plt.plot(df_prophet['ds'], df_prophet['y'], label='실제값')
        plt.plot(forecast['ds'], forecast['yhat'], label='예측값 (Prophet)', color='green')
        plt.title(f"🔮 Prophet 예측 - {국적}/{목적}")
        plt.xlabel("월")
        plt.ylabel("입국자 수")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[Prophet 오류] {국적}/{목적}: {e}")


# %%
# CPU 이용
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 음수 기호 깨짐 방지

# 문자열 정규화 함수
def normalize(text):
    return text.replace(" ", "").lower()

# [1] 사용자 입력 및 유사 매칭
선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

# [2] 데이터 로딩 및 전처리
df = pd.read_csv('../1stProject/data/목적별_국적별_입국_(05년1월~25년5월).csv', encoding='cp949')
df = df.melt(id_vars=["국적", "목적"], var_name="월", value_name="입국자수")
df["월"] = pd.to_datetime(df["월"], format="%Y년%m월")
df["입국자수"] = pd.to_numeric(df["입국자수"].astype(str).str.replace(",", "", regex=False), errors="coerce")

# [3] 유사 국적/목적 자동 완성
if 선택_국적:
    match = [nat for nat in df["국적"].dropna().unique()
             if isinstance(nat, str) and normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in df["목적"].dropna().unique()
             if isinstance(pur, str) and normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

# [4] 예측 대상 필터링
targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
valid_targets = targets[targets['count'] > 24]
if 선택_국적:
    valid_targets = valid_targets[valid_targets["국적"] == 선택_국적]
if 선택_목적:
    valid_targets = valid_targets[valid_targets["목적"] == 선택_목적]

# [5] 예측 루프
for idx, row in valid_targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values("월").reset_index(drop=True)

    if len(df_filtered) < 30:
        continue
    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    # 파생 변수 생성
    df_filtered["연도"] = df_filtered["월"].dt.year
    df_filtered["월_숫자"] = df_filtered["월"].dt.month
    df_filtered["계절"] = df_filtered["월"].dt.month % 12 // 3
    df_filtered["전월입국자수"] = df_filtered["입국자수"].shift(1)
    df_filtered["전년동월입국자수"] = df_filtered["입국자수"].shift(12)
    df_filtered["전월증감률"] = df_filtered["입국자수"].pct_change().shift(1)
    df_filtered["전년증감률"] = (df_filtered["입국자수"] - df_filtered["입국자수"].shift(12)) / df_filtered["입국자수"].shift(12)
    df_filtered["이동평균3"] = df_filtered["입국자수"].rolling(window=3).mean().shift(1)
    df_filtered["이동평균6"] = df_filtered["입국자수"].rolling(window=6).mean().shift(1)
    df_filtered["전월_차이"] = df_filtered["입국자수"] - df_filtered["전월입국자수"]
    df_filtered["전년_차이"] = df_filtered["입국자수"] - df_filtered["전년동월입국자수"]
    df_filtered["연중누적합"] = df_filtered.groupby("연도")["입국자수"].cumsum()
    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if df_filtered.empty:
        continue

    # 특징 및 타겟
    features = ["연도", "월_숫자", "계절", "전월입국자수", "전년동월입국자수",
                "전월증감률", "전년증감률", "이동평균3", "이동평균6",
                "전월_차이", "전년_차이", "연중누적합"]
    X = df_filtered[features]
    y = df_filtered["입국자수"]
    dates = df_filtered["월"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 학습/테스트 분리
    X_train, X_test = X_scaled[:-6], X_scaled[-6:]
    y_train, y_test = y[:-6], y[-6:]
    dates_train, dates_test = dates[:-6], dates[-6:]

    if len(X_test) == 0:
        continue

    # CPU 기반 XGBoost 모델
    xgb = XGBRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10,
                                 cv=TimeSeriesSplit(n_splits=3),
                                 scoring='neg_root_mean_squared_error', random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = np.clip(best_model.predict(X_test), 0, None)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(dates_train, y_train, label="실제값 (학습)", color='blue')
    plt.plot(dates_test, y_test, label="실제값 (테스트)", color='red', linestyle='--')
    plt.plot(dates_test, y_pred, label="예측값 (XGBoost)", color='orange', marker='o')
    plt.title(f"📊 {국적}/{목적} - RMSE: {rmse:.0f}, MAPE: {mape:.1f}%")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Prophet 시계열 예측
    df_prophet = df_filtered[["월", "입국자수"]].rename(columns={"월": "ds", "입국자수": "y"})
    prophet = Prophet(yearly_seasonality=True)
    try:
        prophet.add_country_holidays(country_name='KR')
        prophet.fit(df_prophet)
        future = prophet.make_future_dataframe(periods=6, freq='MS')
        forecast = prophet.predict(future)

        plt.figure(figsize=(14, 5))
        plt.plot(df_prophet['ds'], df_prophet['y'], label='실제값')
        plt.plot(forecast['ds'], forecast['yhat'], label='예측값 (Prophet)', color='green')
        plt.title(f"🔮 Prophet 예측 - {국적}/{목적}")
        plt.xlabel("월")
        plt.ylabel("입국자 수")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[Prophet 오류] {국적}/{목적}: {e}")


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    return text.replace(" ", "").lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

df = pd.read_csv('../1stProject/data/외국인입국자_전처리완료_딥러닝용.csv', encoding='utf-8')
if '구분' in df.columns:
    df = df[df['구분'] == '합계']
    df = df.drop(columns=['구분'])

df = df.melt(id_vars=['국적', '목적'], var_name='월', value_name='입국자수')
df = df[df['목적'] != '소 계']
df['월'] = pd.to_datetime(df['월'], format='%Y년%m월', errors='coerce')
df = df.dropna(subset=['월'])
df['입국자수'] = pd.to_numeric(df['입국자수'].astype(str).str.replace(",", ""), errors='coerce')

if 선택_국적:
    match = [nat for nat in df["국적"].dropna().unique()
             if isinstance(nat, str) and normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in df["목적"].dropna().unique()
             if isinstance(pur, str) and normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
valid_targets = targets[targets['count'] > 24]
if 선택_국적:
    valid_targets = valid_targets[valid_targets["국적"] == 선택_국적]
if 선택_목적:
    valid_targets = valid_targets[valid_targets["목적"] == 선택_목적]

for idx, row in valid_targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values("월").reset_index(drop=True)
    if len(df_filtered) < 30:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    df_filtered["연도"] = df_filtered["월"].dt.year
    df_filtered["월_숫자"] = df_filtered["월"].dt.month
    df_filtered["계절"] = df_filtered["월"].dt.month % 12 // 3
    df_filtered["전월입국자수"] = df_filtered["입국자수"].shift(1)
    df_filtered["전년동월입국자수"] = df_filtered["입국자수"].shift(12)
    df_filtered["전월증감률"] = df_filtered["입국자수"].pct_change().shift(1)
    df_filtered["전년증감률"] = (df_filtered["입국자수"] - df_filtered["입국자수"].shift(12)) / df_filtered["입국자수"].shift(12)
    df_filtered["이동평균3"] = df_filtered["입국자수"].rolling(window=3).mean().shift(1)
    df_filtered["이동평균6"] = df_filtered["입국자수"].rolling(window=6).mean().shift(1)
    df_filtered["전월_차이"] = df_filtered["입국자수"] - df_filtered["전월입국자수"]
    df_filtered["전년_차이"] = df_filtered["입국자수"] - df_filtered["전년동월입국자수"]
    df_filtered["연중누적합"] = df_filtered.groupby("연도")["입국자수"].cumsum()
    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if df_filtered.empty:
        continue

    features = ["연도", "월_숫자", "계절", "전월입국자수", "전년동월입국자수",
                "전월증감률", "전년증감률", "이동평균3", "이동평균6",
                "전월_차이", "전년_차이", "연중누적합"]
    X = df_filtered[features]
    y = df_filtered["입국자수"]
    dates = df_filtered["월"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = X_scaled[:-6], X_scaled[-6:]
    y_train, y_test = y[:-6], y[-6:]
    dates_train, dates_test = dates[:-6], dates[-6:]

    if len(X_test) == 0:
        continue

    xgb = XGBRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10,
                                 cv=TimeSeriesSplit(n_splits=3),
                                 scoring='neg_root_mean_squared_error', random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = np.clip(best_model.predict(X_test), 0, None)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\n📊 XGBoost 성능 평가")
    print(f"- RMSE: {rmse:.0f}")
    print(f"- MAPE: {mape:.1f}%")
    rel_rmse = rmse / y_test.mean()
    print(f"- 평균 대비 RMSE 비율: {rel_rmse*100:.1f}%")
    if mape < 10:
        print("✅ 매우 우수한 예측 (MAPE < 10%)")
    elif mape < 20:
        print("✅ 양호한 예측 (MAPE < 20%)")
    elif mape < 30:
        print("⚠️ 보통 수준 예측 (MAPE < 30%)")
    else:
        print("❌ 예측 정확도 낮음 (MAPE > 30%)")

    result_df = pd.DataFrame({
        '월': dates_test.dt.strftime('%Y-%m'),
        '실제값': y_test.values,
        '예측값': y_pred.astype(int),
        '오차': (y_pred - y_test).astype(int),
        '오차율(%)': ((y_pred - y_test) / y_test * 100).round(2)
    })
    print(result_df)

    plt.figure(figsize=(14, 6))
    plt.plot(dates_train, y_train, label="실제값 (학습)", color='blue')
    plt.plot(dates_test, y_test, label="실제값 (테스트)", color='red', linestyle='--')
    plt.plot(dates_test, y_pred, label="예측값 (XGBoost)", color='orange', marker='o')
    plt.title(f"📊 {국적}/{목적} - RMSE: {rmse:.0f}, MAPE: {mape:.1f}%")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df_prophet = df_filtered[["월", "입국자수"]].rename(columns={"월": "ds", "입국자수": "y"})
    prophet = Prophet(yearly_seasonality=True)
    try:
        prophet.add_country_holidays(country_name='KR')
        prophet.fit(df_prophet)
        future = prophet.make_future_dataframe(periods=6, freq='MS')
        forecast = prophet.predict(future)

        plt.figure(figsize=(14, 5))
        plt.plot(df_prophet['ds'], df_prophet['y'], label='실제값')
        plt.plot(forecast['ds'], forecast['yhat'], label='예측값 (Prophet)', color='green')
        plt.title(f"🔮 Prophet 예측 - {국적}/{목적}")
        plt.xlabel("월")
        plt.ylabel("입국자 수")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[Prophet 오류] {국적}/{목적}: {e}")


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from prophet import Prophet

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

# 사용자 입력 받기
선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

# ✅ CSV 파일로 읽기
df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv', encoding='utf-8')

# 열 이름 정제
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

# 유효한 입력값 필터링 전처리
unique_국적 = df['국적'].dropna().unique()
unique_목적 = df['목적'].dropna().unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

# 24개월 이상 대상 필터링
targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
valid_targets = targets[targets['count'] > 24]
if 선택_국적:
    valid_targets = valid_targets[valid_targets["국적"] == 선택_국적]
if 선택_목적:
    valid_targets = valid_targets[valid_targets["목적"] == 선택_목적]

for idx, row in valid_targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values(["연도", "월"]).reset_index(drop=True)
    if len(df_filtered) < 36:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    df_filtered['연월'] = df_filtered['연도'].astype(str) + '-' + df_filtered['월'].astype(str).str.zfill(2)
    df_filtered['일자'] = pd.to_datetime(df_filtered['연월'])

    df_prophet = df_filtered[['일자', '입국자수']].rename(columns={"일자": "ds", "입국자수": "y"})
    prophet = Prophet(yearly_seasonality=True)
    try:
        prophet.add_country_holidays(country_name='KR')
        prophet.fit(df_prophet)
        future = prophet.make_future_dataframe(periods=6, freq='MS')
        forecast = prophet.predict(future)

        y_true = df_prophet['y'].iloc[-6:].values
        y_pred = forecast['yhat'].iloc[-6:].values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        print(f"\n📊 Prophet 성능 평가")
        print(f"- RMSE: {rmse:.0f}")
        print(f"- MAPE: {mape:.1f}%")
        if mape < 10:
            print("✅ 매우 우수한 예측 (MAPE < 10%)")
        elif mape < 20:
            print("✅ 양호한 예측 (MAPE < 20%)")
        elif mape < 30:
            print("⚠️ 보통 수준 예측 (MAPE < 30%)")
        else:
            print("❌ 예측 정확도 낮음 (MAPE > 30%)")

        plt.figure(figsize=(14, 6))
        plt.plot(df_prophet['ds'], df_prophet['y'], label='실제값')
        plt.plot(forecast['ds'], forecast['yhat'], label='예측값 (Prophet)', color='green')
        plt.title(f"🔮 Prophet 예측 - {국적}/{목적}")
        plt.xlabel("월")
        plt.ylabel("입국자 수")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[Prophet 오류] {국적}/{목적}: {e}")

    # XGBoost
    df_filtered['전월입국자수'] = df_filtered['입국자수'].shift(1)
    df_filtered['전년동월입국자수'] = df_filtered['입국자수'].shift(12)
    df_filtered['전월증감률'] = df_filtered['입국자수'].pct_change().shift(1)
    df_filtered['전년증감률'] = (df_filtered['입국자수'] - df_filtered['입국자수'].shift(12)) / df_filtered['입국자수'].shift(12)
    df_filtered['이동평균3'] = df_filtered['입국자수'].rolling(window=3).mean().shift(1)
    df_filtered['이동평균6'] = df_filtered['입국자수'].rolling(window=6).mean().shift(1)
    df_filtered['전월_차이'] = df_filtered['입국자수'] - df_filtered['전월입국자수']
    df_filtered['전년_차이'] = df_filtered['입국자수'] - df_filtered['전년동월입국자수']

    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna().copy()

    features = ["전월입국자수", "전년동월입국자수", "전월증감률", "전년증감률", "이동평균3", "이동평균6", "전월_차이", "전년_차이"]
    X = df_filtered[features]
    y = df_filtered["입국자수"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = X_scaled[:-6], X_scaled[-6:]
    y_train, y_test = y[:-6], y[-6:]

    xgb = XGBRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10,
                                 cv=TimeSeriesSplit(n_splits=3),
                                 scoring='neg_root_mean_squared_error', random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = np.clip(best_model.predict(X_test), 0, None)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\n📊 XGBoost 성능 평가")
    print(f"- RMSE: {rmse:.0f}")
    print(f"- MAPE: {mape:.1f}%")
    if mape < 10:
        print("✅ 매우 우수한 예측 (MAPE < 10%)")
    elif mape < 20:
        print("✅ 양호한 예측 (MAPE < 20%)")
    elif mape < 30:
        print("⚠️ 보통 수준 예측 (MAPE < 30%)")
    else:
        print("❌ 예측 정확도 낮음 (MAPE > 30%)")

    전체_일자 = df_filtered['일자']
    전체_입국자수 = y
    전체_예측값 = np.concatenate([np.full(len(y_train), np.nan), y_pred])

    plt.figure(figsize=(14, 6))
    plt.plot(전체_일자, 전체_입국자수, label="실제값", color='blue')
    plt.plot(전체_일자, 전체_예측값, label="예측값 (XGBoost)", color='orange', marker='o')
    plt.title(f"📈 {국적}/{목적} - XGBoost 전체 구간 예측 결과")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv', encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

unique_국적 = df['국적'].dropna().unique()
unique_목적 = df['목적'].dropna().unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
valid_targets = targets[targets['count'] > 24]
if 선택_국적:
    valid_targets = valid_targets[valid_targets["국적"] == 선택_국적]
if 선택_목적:
    valid_targets = valid_targets[valid_targets["목적"] == 선택_목적]

for _, row in valid_targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values(["연도", "월"]).reset_index(drop=True)
    if len(df_filtered) < 36:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    df_filtered['연월'] = df_filtered['연도'].astype(str) + '-' + df_filtered['월'].astype(str).str.zfill(2)
    df_filtered['일자'] = pd.to_datetime(df_filtered['연월'])

    df_filtered['전월'] = df_filtered['입국자수'].shift(1)
    df_filtered['전년'] = df_filtered['입국자수'].shift(12)
    df_filtered['전월증감률'] = df_filtered['입국자수'].pct_change().shift(1)
    df_filtered['전년증감률'] = (df_filtered['입국자수'] - df_filtered['전년']) / df_filtered['전년']
    df_filtered['이동평균3'] = df_filtered['입국자수'].rolling(window=3).mean().shift(1)
    df_filtered['이동평균6'] = df_filtered['입국자수'].rolling(window=6).mean().shift(1)
    df_filtered['전월_차이'] = df_filtered['입국자수'] - df_filtered['전월']
    df_filtered['전년_차이'] = df_filtered['입국자수'] - df_filtered['전년']

    if 목적 == '유학연수':
        df_filtered['전전월'] = df_filtered['입국자수'].shift(2)
        df_filtered['전전년'] = df_filtered['입국자수'].shift(24)
        df_filtered['이동평균12'] = df_filtered['입국자수'].rolling(12).mean().shift(1)
        df_filtered['누적합3'] = df_filtered['입국자수'].rolling(3).sum().shift(1)
        df_filtered['월'] = df_filtered['월'].astype(int)

    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if 목적 == '유학연수':
        features = [
            "전월", "전전월", "전년", "전전년", "전월증감률", "전년증감률",
            "이동평균3", "이동평균6", "이동평균12", "누적합3", "전월_차이", "전년_차이", "월"
        ]
        y = np.log1p(df_filtered["입국자수"])
    else:
        features = [
            "전월", "전년", "전월증감률", "전년증감률",
            "이동평균3", "이동평균6", "전월_차이", "전년_차이"
        ]
        y = df_filtered["입국자수"]

    X = df_filtered[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = X_scaled[:-6], X_scaled[-6:]
    y_train, y_test = y[:-6], y[-6:]

    xgb = XGBRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    n_iter = 30 if 목적 == '유학연수' else 10
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=n_iter,
                                 cv=TimeSeriesSplit(n_splits=3),
                                 scoring='neg_root_mean_squared_error', random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    if 목적 == '유학연수':
        y_pred = np.expm1(best_model.predict(X_test))
        y_test_real = np.expm1(y_test)
    else:
        y_pred = np.clip(best_model.predict(X_test), 0, None)
        y_test_real = y_test

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    mape = mean_absolute_percentage_error(y_test_real, y_pred) * 100

    print(f"\n📊 XGBoost 성능 평가")
    print(f"- RMSE: {rmse:.0f}")
    print(f"- MAPE: {mape:.1f}%")
    if mape < 10:
        print("✅ 매우 우수한 예측 (MAPE < 10%)")
    elif mape < 20:
        print("✅ 양호한 예측 (MAPE < 20%)")
    elif mape < 30:
        print("⚠️ 보통 수준 예측 (MAPE < 30%)")
    else:
        print("❌ 예측 정확도 낮음 (MAPE > 30%)")

    전체_일자 = df_filtered['일자']
    전체_입국자수 = np.expm1(y) if 목적 == '유학연수' else y
    전체_예측값 = np.concatenate([np.full(len(y_train), np.nan), y_pred])

    plt.figure(figsize=(14, 6))
    plt.plot(전체_일자, 전체_입국자수, label="실제값", color='blue')
    plt.plot(전체_일자, 전체_예측값, label="예측값 (XGBoost)", color='orange', marker='o')
    plt.title(f"📈 {국적}/{목적} - XGBoost 예측 결과")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

# 입력
선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

# 데이터 로드
df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv', encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

unique_국적 = df['국적'].dropna().unique()
unique_목적 = df['목적'].dropna().unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

# 대상 필터링
targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
targets = targets[targets["count"] > 36]
if 선택_국적:
    targets = targets[targets["국적"] == 선택_국적]
if 선택_목적:
    targets = targets[targets["목적"] == 선택_목적]

# 모델 반복
for _, row in targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values(["연도", "월"]).reset_index(drop=True)
    if len(df_filtered) < 36:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    df_filtered['연월'] = df_filtered['연도'].astype(str) + '-' + df_filtered['월'].astype(str).str.zfill(2)
    df_filtered['일자'] = pd.to_datetime(df_filtered['연월'])

    last_date = df_filtered['일자'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')

    # 기본 피처
    df_filtered['전월'] = df_filtered['입국자수'].shift(1)
    df_filtered['전년'] = df_filtered['입국자수'].shift(12)
    df_filtered['전월증감률'] = df_filtered['입국자수'].pct_change().shift(1)
    df_filtered['전년증감률'] = (df_filtered['입국자수'] - df_filtered['전년']) / df_filtered['전년']
    df_filtered['이동평균3'] = df_filtered['입국자수'].rolling(3).mean().shift(1)
    df_filtered['이동평균6'] = df_filtered['입국자수'].rolling(6).mean().shift(1)
    df_filtered['전월_차이'] = df_filtered['입국자수'] - df_filtered['전월']
    df_filtered['전년_차이'] = df_filtered['입국자수'] - df_filtered['전년']

    if 목적 == '유학연수':
        df_filtered['전전월'] = df_filtered['입국자수'].shift(2)
        df_filtered['전전년'] = df_filtered['입국자수'].shift(24)
        df_filtered['이동평균12'] = df_filtered['입국자수'].rolling(12).mean().shift(1)
        df_filtered['누적합3'] = df_filtered['입국자수'].rolling(3).sum().shift(1)
        df_filtered['월'] = df_filtered['월'].astype(int)

    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna().copy()

    # 피처 및 타깃
    if 목적 == '유학연수':
        features = [
            "전월", "전전월", "전년", "전전년", "전월증감률", "전년증감률",
            "이동평균3", "이동평균6", "이동평균12", "누적합3", "전월_차이", "전년_차이", "월"
        ]
        y = np.log1p(df_filtered["입국자수"])
    else:
        features = [
            "전월", "전년", "전월증감률", "전년증감률",
            "이동평균3", "이동평균6", "전월_차이", "전년_차이"
        ]
        y = df_filtered["입국자수"]

    X = df_filtered[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 모델 학습 및 평가용 MAPE
    X_train_eval, X_test_eval = X_scaled[:-6], X_scaled[-6:]
    y_train_eval, y_test_eval = y[:-6], y[-6:]

    xgb = XGBRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    n_iter = 30 if 목적 == '유학연수' else 10
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=n_iter,
                                 cv=TimeSeriesSplit(n_splits=3),
                                 scoring='neg_root_mean_squared_error', random_state=42)
    search.fit(X_train_eval, y_train_eval)
    best_model = search.best_estimator_

    # MAPE → 신뢰도
    if 목적 == '유학연수':
        y_pred_eval = np.expm1(best_model.predict(X_test_eval))
        y_test_real = np.expm1(y_test_eval)
    else:
        y_pred_eval = best_model.predict(X_test_eval)
        y_test_real = y_test_eval

    mape = mean_absolute_percentage_error(y_test_real, y_pred_eval) * 100
    신뢰도 = max(0, min(100, 100 - mape))

    # 미래 예측
    recent = df_filtered.copy()
    preds = []
    for i in range(6):
        row = recent.iloc[-1:].copy()
        row['입국자수'] = preds[-1] if preds else row['입국자수']
        row['전월'] = recent['입국자수'].iloc[-1]
        row['전전월'] = recent['입국자수'].iloc[-2] if 목적 == '유학연수' else np.nan
        row['전년'] = recent['입국자수'].iloc[-12]
        row['전전년'] = recent['입국자수'].iloc[-24] if 목적 == '유학연수' else np.nan
        row['전월증감률'] = (recent['입국자수'].iloc[-1] - recent['입국자수'].iloc[-2]) / recent['입국자수'].iloc[-2]
        row['전년증감률'] = (recent['입국자수'].iloc[-1] - recent['입국자수'].iloc[-12]) / recent['입국자수'].iloc[-12]
        row['이동평균3'] = recent['입국자수'].iloc[-3:].mean()
        row['이동평균6'] = recent['입국자수'].iloc[-6:].mean()
        row['이동평균12'] = recent['입국자수'].iloc[-12:].mean() if 목적 == '유학연수' else np.nan
        row['누적합3'] = recent['입국자수'].iloc[-3:].sum() if 목적 == '유학연수' else np.nan
        row['전월_차이'] = row['입국자수'] - row['전월']
        row['전년_차이'] = row['입국자수'] - row['전년']
        row['월'] = ((last_date.month + i - 1) % 12) + 1 if 목적 == '유학연수' else np.nan

        row = row[features]
        row = row.fillna(method='ffill', axis=1)
        row_scaled = scaler.transform(row)
        pred = best_model.predict(row_scaled)[0]
        pred = np.expm1(pred) if 목적 == '유학연수' else pred
        preds.append(pred)
        recent = pd.concat([recent, pd.DataFrame([{'입국자수': pred}])], ignore_index=True)

    # 출력
    result = pd.DataFrame({
        '예측월': future_dates.strftime('%Y-%m'),
        '예측 입국자 수': np.round(preds).astype(int),
        '신뢰도(%)': [round(신뢰도, 1)] * 6
    })
    print(result.to_string(index=False))

    # 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(df_filtered['일자'], df_filtered['입국자수'], label="과거 실측", color='blue')
    plt.plot(future_dates, preds, label="예측값", color='red', marker='o')
    plt.title(f"🔮 {국적}/{목적} - 향후 6개월 예측 (신뢰도: {신뢰도:.1f}%)")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

def make_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv', encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

df['연월'] = df['연도'].astype(str) + '-' + df['월'].astype(str).str.zfill(2)
df['일자'] = pd.to_datetime(df['연월'])

df['입국자수_log'] = np.log1p(df['입국자수'])
df['입국자수_log_smooth'] = df.groupby(['국적', '목적'])['입국자수_log'].transform(lambda x: x.ewm(span=4, adjust=False).mean())
df['월값'] = df['일자'].dt.month / 12.0

def create_sequences(df_filtered, seq_length):
    feature_cols = ['입국자수_log_smooth', '월값']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_filtered[feature_cols])
    Xs, ys = [], []
    for i in range(len(scaled) - seq_length):
        Xs.append(scaled[i:i+seq_length])
        ys.append(scaled[i+seq_length][0])
    return np.array(Xs), np.array(ys), scaler

def get_params(목적):
    목적 = normalize(목적)
    if '유학' in 목적:
        return 30, [128, 64], [0.3, 0.2]
    elif '관광' in 목적:
        return 24, [128, 64], [0.3, 0.2]
    elif '상용' in 목적:
        return 12, [64, 32], [0.2, 0.1]
    else:
        return 18, [64, 32], [0.3, 0.2]

unique_국적 = df['국적'].dropna().unique()
unique_목적 = df['목적'].dropna().unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
valid_targets = targets[targets['count'] > 36]
if 선택_국적:
    valid_targets = valid_targets[valid_targets["국적"] == 선택_국적]
if 선택_목적:
    valid_targets = valid_targets[valid_targets["목적"] == 선택_목적]

for idx, row in valid_targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values("일자").reset_index(drop=True)
    seq_length, lstm_units, dropouts = get_params(목적)

    max_seq_length = min(seq_length, max(6, len(df_filtered) - 6))
    if len(df_filtered) < max_seq_length + 6:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    X, y, scaler = create_sequences(df_filtered, max_seq_length)
    X_train, y_train = X[:-6], y[:-6]
    X_test, y_test = X[-6:], y[-6:]

    model = Sequential()
    model.add(LSTM(lstm_units[0], return_sequences=True, input_shape=(max_seq_length, X.shape[2])))
    model.add(Dropout(dropouts[0]))
    model.add(LSTM(lstm_units[1], return_sequences=False))
    model.add(Dropout(dropouts[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=300, batch_size=8, callbacks=[es], verbose=0)

    y_pred = model.predict(X_test)
    y_test_inv = np.expm1(scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 1))]))[:, 0])
    y_pred_inv = np.expm1(scaler.inverse_transform(np.hstack([y_pred, np.zeros((len(y_pred), 1))]))[:, 0])

    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
    신뢰도 = 100 - mape
    r2 = r2_score(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    print(f"R^2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"신뢰도: {신뢰도:.1f}%")

    future_preds = []
    current_seq = X[-1].copy()
    future_month = df_filtered['월값'].iloc[-1]
    for _ in range(6):
        pred = model.predict(current_seq.reshape(1, max_seq_length, X.shape[2]), verbose=0)
        next_scaled = np.array([[pred[0, 0], future_month]])
        current_seq = np.vstack((current_seq[1:], next_scaled))
        future_preds.append(pred[0, 0])

    future_preds_inv = np.expm1(scaler.inverse_transform(np.hstack([np.array(future_preds).reshape(-1, 1), np.zeros((6, 1))]))[:, 0]).astype(int)

    last_date = df_filtered['일자'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=6, freq='MS')

    result = pd.DataFrame({
        '예측월': future_dates.strftime('%Y-%m'),
        '예측 입국자 수': future_preds_inv.flatten(),
        '신뢰도(%)': [round(신뢰도, 1)] * 6
    })
    print(result.to_string(index=False))

    plt.figure(figsize=(14, 6))
    plt.plot(df_filtered['일자'], np.expm1(df_filtered['입국자수_log']), label="실제값", color='blue')
    plt.plot(future_dates, future_preds_inv, label="예측값 (LSTM)", color='orange', marker='o')
    plt.title(f"🔮 {국적}/{목적} - 향후 6개월 예측 (신뢰도: {신뢰도:.1f}%, R^2: {r2:.4f}, RMSE: {rmse:.2f})")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text): return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

def create_sequences(df_filtered, seq_length):
    feature_cols = ['입국자수_log_smooth', '월값', '연도값']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_filtered[feature_cols])
    Xs, ys = [], []
    for i in range(len(scaled) - seq_length):
        Xs.append(scaled[i:i+seq_length])
        ys.append(scaled[i+seq_length][0])
    return np.array(Xs), np.array(ys), scaler

def get_params(목적):
    목적 = normalize(목적)
    if '유학' in 목적:
        return 30, [128, 64], [0.3, 0.2]
    elif '관광' in 목적:
        return 24, [128, 64], [0.3, 0.2]
    elif '상용' in 목적:
        return 12, [64, 32], [0.2, 0.1]
    else:
        return 18, [64, 32], [0.3, 0.2]

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()
df['연월'] = df['연도'].astype(str) + '-' + df['월'].astype(str).str.zfill(2)
df['일자'] = pd.to_datetime(df['연월'])
df['입국자수_log'] = np.log1p(df['입국자수'])
df['입국자수_log_smooth'] = df.groupby(['국적', '목적'])['입국자수_log'].transform(lambda x: x.ewm(span=4, adjust=False).mean())
df['월값'] = df['일자'].dt.month / 12.0
df['연도값'] = df['일자'].dt.year / 2100

unique_국적 = df['국적'].dropna().unique()
unique_목적 = df['목적'].dropna().unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
valid_targets = targets[targets['count'] > 36]
if 선택_국적:
    valid_targets = valid_targets[valid_targets["국적"] == 선택_국적]
if 선택_목적:
    valid_targets = valid_targets[valid_targets["목적"] == 선택_목적]

for _, row in valid_targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values("일자").reset_index(drop=True)
    seq_length, gru_units, dropouts = get_params(목적)

    if len(df_filtered) < seq_length + 6:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")
    X, y, scaler = create_sequences(df_filtered, seq_length)
    X_train, y_train = X[:-6], y[:-6]
    X_test, y_test = X[-6:], y[-6:]

    model = Sequential([
        GRU(gru_units[0], return_sequences=True, input_shape=(seq_length, X.shape[2])),
        Dropout(dropouts[0]),
        GRU(gru_units[1], return_sequences=False),
        Dropout(dropouts[1]),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=300, batch_size=8, callbacks=[es], verbose=0)

    gru_pred = model.predict(X_test)
    gru_features = np.concatenate([gru_pred, X_test[:, -1, 1:]], axis=1)
    xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    xgb.fit(gru_features, y_train[-6:])

    xgb_pred = xgb.predict(gru_features)
    y_test_inv = np.expm1(scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 2))]))[:, 0])
    y_pred_inv = np.expm1(scaler.inverse_transform(np.hstack([xgb_pred.reshape(-1, 1), np.zeros((len(xgb_pred), 2))]))[:, 0])

    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
    신뢰도 = 100 - mape
    r2 = r2_score(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    print(f"R^2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"신뢰도: {신뢰도:.1f}%")

    future_preds = []
    current_seq = X[-1].copy()
    for _ in range(6):
        gru_out = model.predict(current_seq.reshape(1, seq_length, X.shape[2]), verbose=0)[0, 0]
        future_feat = current_seq[-1, 1:]
        xgb_input = np.hstack([gru_out, future_feat])
        xgb_out = xgb.predict(xgb_input.reshape(1, -1))[0]
        future_preds.append(xgb_out)
        next_scaled = np.hstack([[xgb_out], future_feat])
        current_seq = np.vstack((current_seq[1:], next_scaled))

    future_preds_inv = np.expm1(scaler.inverse_transform(np.hstack([np.array(future_preds).reshape(-1, 1), np.zeros((6, 2))]))[:, 0]).astype(int)
    last_date = df_filtered['일자'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=6, freq='MS')

    result = pd.DataFrame({
        '예측월': future_dates.strftime('%Y-%m'),
        '예측 입국자 수': future_preds_inv,
        '신뢰도(%)': [round(신뢰도, 1)] * 6
    })
    print(result.to_string(index=False))

    plt.figure(figsize=(14, 6))
    plt.plot(df_filtered['일자'], np.expm1(df_filtered['입국자수_log']), label="실제값", color='blue')
    plt.plot(future_dates, future_preds_inv, label="예측값 (GRU+XGBoost)", color='orange', marker='o')
    plt.title(f"🔮 {국적}/{목적} - 향후 6개월 예측 (신뢰도: {신뢰도:.1f}%, R²: {r2:.4f}, RMSE: {rmse:.2f})")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
# GRU vs XGBoost 비교
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text): return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

def create_sequences(df_filtered, seq_length):
    feature_cols = ['입국자수_log_smooth', '월값', '연도값']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_filtered[feature_cols])
    Xs, ys = [], []
    for i in range(len(scaled) - seq_length):
        Xs.append(scaled[i:i+seq_length])
        ys.append(scaled[i+seq_length][0])
    return np.array(Xs), np.array(ys), scaler

def get_params(목적):
    목적 = normalize(목적)
    if '유학' in 목적:
        return 30, [128, 64], [0.3, 0.2]
    elif '관광' in 목적:
        return 24, [128, 64], [0.3, 0.2]
    elif '상용' in 목적:
        return 12, [64, 32], [0.2, 0.1]
    else:
        return 18, [64, 32], [0.3, 0.2]

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()
df['연월'] = df['연도'].astype(str) + '-' + df['월'].astype(str).str.zfill(2)
df['일자'] = pd.to_datetime(df['연월'])
df['입국자수_log'] = np.log1p(df['입국자수'])
df['입국자수_log_smooth'] = df.groupby(['국적', '목적'])['입국자수_log'].transform(lambda x: x.ewm(span=4, adjust=False).mean())
df['월값'] = df['일자'].dt.month / 12.0
df['연도값'] = df['일자'].dt.year / 2100

unique_국적 = df['국적'].dropna().unique()
unique_목적 = df['목적'].dropna().unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
valid_targets = targets[targets['count'] > 36]
if 선택_국적:
    valid_targets = valid_targets[valid_targets["국적"] == 선택_국적]
if 선택_목적:
    valid_targets = valid_targets[valid_targets["목적"] == 선택_목적]

for _, row in valid_targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values("일자").reset_index(drop=True)
    seq_length, gru_units, dropouts = get_params(목적)

    if len(df_filtered) < seq_length + 6:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")
    X, y, scaler = create_sequences(df_filtered, seq_length)
    X_train, y_train = X[:-6], y[:-6]
    X_test, y_test = X[-6:], y[-6:]

    # GRU 모델
    model = Sequential([
        GRU(gru_units[0], return_sequences=True, input_shape=(seq_length, X.shape[2])),
        Dropout(dropouts[0]),
        GRU(gru_units[1], return_sequences=False),
        Dropout(dropouts[1]),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=300, batch_size=8, callbacks=[es], verbose=0)

    gru_pred = model.predict(X_test)

    # XGBoost 모델
    flat_X_train = X_train.reshape(X_train.shape[0], -1)
    flat_X_test = X_test.reshape(X_test.shape[0], -1)
    xgb_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    xgb_model.fit(flat_X_train, y_train)
    xgb_pred = xgb_model.predict(flat_X_test)

    # 역변환
    y_test_inv = np.expm1(scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 2))]))[:, 0])
    gru_inv = np.expm1(scaler.inverse_transform(np.hstack([gru_pred, np.zeros((len(gru_pred), 2))]))[:, 0])
    xgb_inv = np.expm1(scaler.inverse_transform(np.hstack([xgb_pred.reshape(-1, 1), np.zeros((len(xgb_pred), 2))]))[:, 0])

    # 평가
    def evaluate(true, pred, model_name):
        r2 = r2_score(true, pred)
        mape = mean_absolute_percentage_error(true, pred) * 100
        rmse = np.sqrt(mean_squared_error(true, pred))
        print(f"\n✅ {model_name} 성능 평가")
        print(f" - R^2: {r2:.4f}")
        print(f" - MAPE: {mape:.2f}%")
        print(f" - RMSE: {rmse:.2f}")
        print(f" - 신뢰도: {100 - mape:.1f}%")

    evaluate(y_test_inv, gru_inv, "GRU")
    evaluate(y_test_inv, xgb_inv, "XGBoost")

    plt.figure(figsize=(14, 6))
    plt.plot(df_filtered['일자'].iloc[-len(y_test_inv):], y_test_inv, label="실제값", marker='o')
    plt.plot(df_filtered['일자'].iloc[-len(gru_inv):], gru_inv, label="GRU 예측", marker='o')
    plt.plot(df_filtered['일자'].iloc[-len(xgb_inv):], xgb_inv, label="XGBoost 예측", marker='o')
    plt.title(f"🔮 {국적}/{목적} - 모델별 비교 예측")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
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
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도 = int(input("예측할 연도 입력 (예: 2025): ").strip())
예측월입력 = input("예측할 월 입력 (1~12 또는 3,6,9 또는 10~12): ").strip()

if '~' in 예측월입력:
    start, end = map(int, 예측월입력.split('~'))
    예측월 = list(range(start, end + 1))
elif ',' in 예측월입력:
    예측월 = list(map(int, 예측월입력.split(',')))
else:
    예측월 = [int(예측월입력)]

path = './data/외국인입국자_전처리완료_딥러닝용.csv'
df = pd.read_csv(path, encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

unique_국적 = df['국적'].dropna().unique()
unique_목적 = df['목적'].dropna().unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n\U0001F449 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
targets = targets[targets["count"] > 36]
if 선택_국적:
    targets = targets[targets["국적"] == 선택_국적]
if 선택_목적:
    targets = targets[targets["목적"] == 선택_목적]

for _, row in targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values(["연도", "월"]).reset_index(drop=True)
    if len(df_filtered) < 36:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    df_filtered['연월'] = df_filtered['연도'].astype(str) + '-' + df_filtered['월'].astype(str).str.zfill(2)
    df_filtered['일자'] = pd.to_datetime(df_filtered['연월'])
    last_date = df_filtered['일자'].iloc[-1]

    df_filtered['전월'] = df_filtered['입국자수'].shift(1)
    df_filtered['전년'] = df_filtered['입국자수'].shift(12)
    df_filtered['전월증감률'] = df_filtered['입국자수'].pct_change().shift(1)
    df_filtered['전년증감률'] = (df_filtered['입국자수'] - df_filtered['전년']) / df_filtered['전년']
    df_filtered['이동평균3'] = df_filtered['입국자수'].rolling(3).mean().shift(1)
    df_filtered['이동평균6'] = df_filtered['입국자수'].rolling(6).mean().shift(1)
    df_filtered['전월_차이'] = df_filtered['입국자수'] - df_filtered['전월']
    df_filtered['전년_차이'] = df_filtered['입국자수'] - df_filtered['전년']

    if 목적 == '유학연수':
        df_filtered['전전월'] = df_filtered['입국자수'].shift(2)
        df_filtered['전전년'] = df_filtered['입국자수'].shift(24)
        df_filtered['이동평균12'] = df_filtered['입국자수'].rolling(12).mean().shift(1)
        df_filtered['누적합3'] = df_filtered['입국자수'].rolling(3).sum().shift(1)
        df_filtered['월'] = df_filtered['월'].astype(int)

    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if 목적 == '유학연수':
        features = ["전월", "전전월", "전년", "전전년", "전월증감률", "전년증감률",
                    "이동평균3", "이동평균6", "이동평균12", "누적합3", "전월_차이", "전년_차이", "월"]
        y = np.log1p(df_filtered["입국자수"])
    else:
        features = ["전월", "전년", "전월증감률", "전년증감률",
                    "이동평균3", "이동평균6", "전월_차이", "전년_차이"]
        y = df_filtered["입국자수"]

    X = df_filtered[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = X_scaled[:-6], X_scaled[-6:]
    y_train, y_test = y[:-6], y[-6:]

    # 성능 개선: 목적이 '상용'인 경우만 하이퍼파라미터 튜닝
    if 목적 == '상용':
        param_grid = {
            'n_estimators': [300, 500, 700],
            'learning_rate': [0.01, 0.03, 0.05],
            'max_depth': [3, 5, 7]
        }
        model = XGBRegressor(random_state=42)
        search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3, scoring='neg_mean_squared_error', random_state=42)
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if 목적 == '유학연수':
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
        예측_월수 = int((예측연도 - df_filtered['일자'].dt.year.iloc[-1]) * 12 + (month - df_filtered['일자'].dt.month.iloc[-1]))

        future_preds = []
        recent = df_filtered.copy()
        for i in range(예측_월수):
            row = recent.iloc[-1:].copy()
            row['입국자수'] = future_preds[-1] if future_preds else row['입국자수']
            row['전월'] = recent['입국자수'].iloc[-1]
            row['전전월'] = recent['입국자수'].iloc[-2] if 목적 == '유학연수' else np.nan
            row['전년'] = recent['입국자수'].iloc[-12]
            row['전전년'] = recent['입국자수'].iloc[-24] if 목적 == '유학연수' else np.nan
            row['전월증감률'] = (recent['입국자수'].iloc[-1] - recent['입국자수'].iloc[-2]) / recent['입국자수'].iloc[-2]
            row['전년증감률'] = (recent['입국자수'].iloc[-1] - recent['입국자수'].iloc[-12]) / recent['입국자수'].iloc[-12]
            row['이동평균3'] = recent['입국자수'].iloc[-3:].mean()
            row['이동평균6'] = recent['입국자수'].iloc[-6:].mean()
            row['이동평균12'] = recent['입국자수'].iloc[-12:].mean() if 목적 == '유학연수' else np.nan
            row['누적합3'] = recent['입국자수'].iloc[-3:].sum() if 목적 == '유학연수' else np.nan
            row['전월_차이'] = row['입국자수'] - row['전월']
            row['전년_차이'] = row['입국자수'] - row['전년']
            row['월'] = ((last_date.month + i - 1) % 12) + 1 if 목적 == '유학연수' else np.nan

            row = row[features].fillna(method='ffill', axis=1)
            row_scaled = scaler.transform(row)
            pred = model.predict(row_scaled)[0]
            pred = np.expm1(pred) if 목적 == '유학연수' else pred
            future_preds.append(pred)
            recent = pd.concat([recent, pd.DataFrame([{'입국자수': pred}])], ignore_index=True)

        pred_date = pd.to_datetime(f"{예측연도}-{month:02d}-01")
        예측값 = np.round(future_preds[-1]).astype(int)

        print(f"\n🔮 {예측연도}년 {month}월 예측 입국자 수: {예측값:,}명 (신뢰도: {신뢰도:.1f}%)")
        예측결과들.append((pred_date, 예측값))

    예측_df = pd.DataFrame(예측결과들, columns=['예측월', '예측입국자수'])
    예측_df['예측월'] = 예측_df['예측월'].dt.strftime('%Y-%m')  # 날짜를 문자열로 변환
    csv_path = f"./data/{국적}_{목적}_예측결과.csv"
    예측_df.to_csv(csv_path, index=False, encoding='utf-8-sig')


    plt.figure(figsize=(14, 6))
    최근기간 = 6
    시각화데이터 = df_filtered[-최근기간:].copy()
    plt.plot(시각화데이터['일자'], 시각화데이터['입국자수'], label="최근 실측", color='blue', linewidth=2)
    pred_dates = [d for d, _ in 예측결과들]
    pred_values = [v for _, v in 예측결과들]
    if pred_dates:
        plt.plot(pred_dates, pred_values, color='red', marker='o', linestyle='--', linewidth=2, label='예측값')
    for pred_date, 예측값 in 예측결과들:
        plt.scatter(pred_date, 예측값, color='red', s=60, edgecolors='black', zorder=5)
        plt.text(pred_date, 예측값 + max(pred_values) * 0.02, f"{예측값:,}", ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')
    start_date = 시각화데이터['일자'].iloc[0]
    end_date = pred_dates[-1] + pd.DateOffset(months=1)
    plt.xlim([start_date, end_date])
    plt.title(f"{국적}/{목적} - {예측연도}년 {예측월}월 예측 결과", fontsize=14, fontweight='bold')
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
import re, warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()

path = './data/외국인입국자_전처리완료_딥러닝용.csv'
df = pd.read_csv(path, encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

국적목록 = df['국적'].unique()
목적목록 = df['목적'].unique()

if 선택_국적:
    match = [nat for nat in 국적목록 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); exit()

if 선택_목적:
    match = [pur for pur in 목적목록 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); exit()

# 대상 목록
대상목록 = df[df['국적'] == 선택_국적][['국적', '목적']].drop_duplicates()
if 선택_목적:
    대상목록 = 대상목록[대상목록['목적'] == 선택_목적]

# 반복 실행
for _, row in 대상목록.iterrows():
    국적, 목적 = row['국적'], row['목적']
    data = df[(df['국적'] == 국적) & (df['목적'] == 목적)].sort_values(['연도', '월'])
    data['연월'] = data['연도'].astype(str) + '-' + data['월'].astype(str).str.zfill(2)
    data['일자'] = pd.to_datetime(data['연월'])

    data['전월'] = data['입국자수'].shift(1)
    data['전년'] = data['입국자수'].shift(12)
    data['이동평균3'] = data['입국자수'].rolling(3).mean().shift(1)
    data['월'] = data['월'].astype(int)
    data = data.dropna().copy()

    if len(data) < 24:
        print(f"\n❌ {국적} / {목적} 조합은 LSTM 학습을 위한 데이터가 부족합니다. (24개월 이상 필요)")
        continue

    features = ['전월', '전년', '이동평균3', '월']
    target = '입국자수'

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(data[features])
    y_scaled = scaler_y.fit_transform(data[[target]])

    # LSTM 입력 시퀀스 생성
    sequence_length = 12
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y_scaled[i + sequence_length])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    test_size = 6
    X_train, X_test = X_seq[:-test_size], X_seq[-test_size:]
    y_train, y_test = y_seq[:-test_size], y_seq[-test_size:]

    model = Sequential([
        LSTM(32, input_shape=(sequence_length, X_seq.shape[2])),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred).flatten()
    y_test_inv = scaler_y.inverse_transform(y_test).flatten()

    r2 = r2_score(y_test_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
    rmse = mean_squared_error(y_test_inv, y_pred_inv, squared=False)
    dw = durbin_watson(y_test_inv - y_pred_inv)
    신뢰도 = max(0, min(100, 100 - mape))

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(data)} 개월치)")
    print("✅ LSTM 성능 평가")
    print(f" - R^2: {r2:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - 신뢰도: {신뢰도:.1f}%")
    print(f" - Durbin-Watson: {dw:.4f}")

    plt.figure(figsize=(12,6))
    최근일자 = data['일자'].values[-len(y_test_inv):]
    실측값 = data['입국자수'].values[-len(y_test_inv):]

    plt.plot(최근일자, 실측값, label='실측값', color='blue')
    plt.plot(최근일자, y_pred_inv, label='예측값', color='red', linestyle='--', marker='o')
    plt.title(f"{국적} / {목적} - LSTM 예측 결과")
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
import re, warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text): return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도 = int(input("예측할 연도 입력 (예: 2025): ").strip())
예측월입력 = input("예측할 월 입력 (예: 10,11,12 또는 7~9): ").strip()

if '~' in 예측월입력:
    start, end = map(int, 예측월입력.split('~'))
    예측월 = list(range(start, end + 1))
elif ',' in 예측월입력:
    예측월 = list(map(int, 예측월입력.split(',')))
else:
    예측월 = [int(예측월입력)]

path = './data/외국인입국자_전처리완료_딥러닝용.csv'
df = pd.read_csv(path, encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

국적목록 = df['국적'].unique()
목적목록 = df['목적'].unique()

if 선택_국적:
    match = [nat for nat in 국적목록 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); exit()

if 선택_목적:
    match = [pur for pur in 목적목록 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); exit()

대상목록 = df[df['국적'] == 선택_국적][['국적', '목적']].drop_duplicates()
if 선택_목적:
    대상목록 = 대상목록[대상목록['목적'] == 선택_목적]

for _, row in 대상목록.iterrows():
    국적, 목적 = row['국적'], row['목적']
    data = df[(df['국적'] == 국적) & (df['목적'] == 목적)].sort_values(['연도', '월'])
    data['연월'] = data['연도'].astype(str) + '-' + data['월'].astype(str).str.zfill(2)
    data['일자'] = pd.to_datetime(data['연월'])

    data['전월'] = data['입국자수'].shift(1)
    data['전전월'] = data['입국자수'].shift(2)
    data['전년'] = data['입국자수'].shift(12)
    data['전전년'] = data['입국자수'].shift(24)
    data['전월증감률'] = data['입국자수'].pct_change().shift(1)
    data['전년증감률'] = (data['입국자수'] - data['전년']) / data['전년']
    data['이동평균3'] = data['입국자수'].rolling(3).mean().shift(1)
    data['이동평균6'] = data['입국자수'].rolling(6).mean().shift(1)
    data['이동평균12'] = data['입국자수'].rolling(12).mean().shift(1)
    data['누적합3'] = data['입국자수'].rolling(3).sum().shift(1)
    data['전월_차이'] = data['입국자수'] - data['전월']
    data['전년_차이'] = data['입국자수'] - data['전년']
    data['월'] = data['월'].astype(int)

    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(data) < 36:
        print(f"\n❌ {국적} / {목적} 조합은 학습 가능한 데이터가 부족합니다. (36개월 이상 필요)")
        continue

    features = ['전월', '전전월', '전년', '전전년', '전월증감률', '전년증감률',
                '이동평균3', '이동평균6', '이동평균12', '누적합3', '전월_차이', '전년_차이', '월']
    target = '입국자수'

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(data[features])
    y_scaled = scaler_y.fit_transform(np.log1p(data[[target]]))

    sequence_length = 12
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y_scaled[i + sequence_length])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    X_train, y_train = X_seq, y_seq

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False), input_shape=(sequence_length, X_seq.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

    # 모델 성능 평가용
    y_pred = model.predict(X_train)
    y_pred_inv = np.expm1(scaler_y.inverse_transform(y_pred).flatten())
    y_true_inv = np.expm1(scaler_y.inverse_transform(y_train).flatten())

    r2 = r2_score(y_true_inv, y_pred_inv)
    nonzero_idx = y_true_inv != 0
    if np.any(nonzero_idx):
        mape = mean_absolute_percentage_error(y_true_inv[nonzero_idx], y_pred_inv[nonzero_idx]) * 100
    else:
        mape = np.nan
    rmse = mean_squared_error(y_true_inv, y_pred_inv, squared=False)
    dw = durbin_watson(y_true_inv - y_pred_inv)
    신뢰도 = max(0, min(100, 100 - mape))

    print(f"📈 LSTM 성능 평가 ({국적} / {목적})")
    print(f" - R^2: {r2:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - 신뢰도: {신뢰도:.1f}%")
    print(f" - Durbin-Watson: {dw:.4f}")

    future_preds = []
    recent_inputs = X_scaled[-sequence_length:].tolist()
    recent_raw = data.copy()

    for month in 예측월:
        row = recent_raw.iloc[-1:].copy()
        row['입국자수'] = future_preds[-1] if future_preds else row['입국자수']
        row['전월'] = recent_raw['입국자수'].iloc[-1]
        row['전전월'] = recent_raw['입국자수'].iloc[-2]
        row['전년'] = recent_raw['입국자수'].iloc[-12]
        row['전전년'] = recent_raw['입국자수'].iloc[-24]
        row['전월증감률'] = (recent_raw['입국자수'].iloc[-1] - recent_raw['입국자수'].iloc[-2]) / recent_raw['입국자수'].iloc[-2]
        row['전년증감률'] = (recent_raw['입국자수'].iloc[-1] - recent_raw['입국자수'].iloc[-12]) / recent_raw['입국자수'].iloc[-12]
        row['이동평균3'] = recent_raw['입국자수'].iloc[-3:].mean()
        row['이동평균6'] = recent_raw['입국자수'].iloc[-6:].mean()
        row['이동평균12'] = recent_raw['입국자수'].iloc[-12:].mean()
        row['누적합3'] = recent_raw['입국자수'].iloc[-3:].sum()
        row['전월_차이'] = row['입국자수'] - row['전월']
        row['전년_차이'] = row['입국자수'] - row['전년']
        row['월'] = ((recent_raw['일자'].dt.month.iloc[-1] + len(future_preds)) - 1) % 12 + 1

        row = row[features].fillna(method='ffill', axis=1)
        row_scaled = scaler_x.transform(row)
        recent_inputs.append(row_scaled[0])
        input_seq = np.array(recent_inputs[-sequence_length:]).reshape(1, sequence_length, len(features))

        pred_scaled = model.predict(input_seq)[0][0]
        pred = np.expm1(scaler_y.inverse_transform([[pred_scaled]])[0][0])
        future_preds.append(pred)
        recent_raw = pd.concat([recent_raw, pd.DataFrame([{'입국자수': pred}])], ignore_index=True)

        pred_date = pd.to_datetime(f"{예측연도}-{month:02d}-01")
        예측값 = np.round(pred).astype(int)
        print(f"\n🔮 {예측연도}년 {month}월 예측 입국자 수: {예측값:,}명")

    pred_dates = [pd.to_datetime(f"{예측연도}-{m:02d}-01") for m in 예측월]
    pred_values = [np.round(p).astype(int) for p in future_preds[-len(예측월):]]

    plt.figure(figsize=(14, 6))
    plt.plot(data['일자'].iloc[-12:], data['입국자수'].iloc[-12:], label="최근 실측", color='blue', linewidth=2)
    plt.plot(pred_dates, pred_values, color='red', marker='o', linestyle='--', linewidth=2, label='예측값')
    for date, val in zip(pred_dates, pred_values):
        plt.scatter(date, val, color='red', s=60, edgecolors='black', zorder=5)
        plt.text(date, val + max(pred_values)*0.02, f"{val:,}", ha='center', va='bottom', fontsize=9)
    plt.title(f"{국적}/{목적} - {예측연도}년 {예측월}월 예측 결과", fontsize=14, fontweight='bold')
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization, TimeDistributed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
import re, warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text): return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도 = int(input("예측할 연도 입력 (예: 2025): ").strip())
예측월입력 = input("예측할 월 입력 (예: 10,11,12 또는 7~9): ").strip()

if '~' in 예측월입력:
    start, end = map(int, 예측월입력.split('~'))
    예측월 = list(range(start, end + 1))
elif ',' in 예측월입력:
    예측월 = list(map(int, 예측월입력.split(',')))
else:
    예측월 = [int(예측월입력)]

path = './data/외국인입국자_전처리완료_딥러닝용.csv'
df = pd.read_csv(path, encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

국적목록 = df['국적'].unique()
목적목록 = df['목적'].unique()

if 선택_국적:
    match = [nat for nat in 국적목록 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); exit()

if 선택_목적:
    match = [pur for pur in 목적목록 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); exit()

대상목록 = df[df['국적'] == 선택_국적][['국적', '목적']].drop_duplicates()
if 선택_목적:
    대상목록 = 대상목록[대상목록['목적'] == 선택_목적]

for _, row in 대상목록.iterrows():
    국적, 목적 = row['국적'], row['목적']
    data = df[(df['국적'] == 국적) & (df['목적'] == 목적)].sort_values(['연도', '월'])
    data['연월'] = data['연도'].astype(str) + '-' + data['월'].astype(str).str.zfill(2)
    data['일자'] = pd.to_datetime(data['연월'])

    data['전월'] = data['입국자수'].shift(1)
    data['전전월'] = data['입국자수'].shift(2)
    data['전년'] = data['입국자수'].shift(12)
    data['전전년'] = data['입국자수'].shift(24)
    data['전월증감률'] = data['입국자수'].pct_change().shift(1)
    data['전년증감률'] = (data['입국자수'] - data['전년']) / data['전년']
    data['이동평균3'] = data['입국자수'].rolling(3).mean().shift(1)
    data['이동평균6'] = data['입국자수'].rolling(6).mean().shift(1)
    data['이동평균12'] = data['입국자수'].rolling(12).mean().shift(1)
    data['누적합3'] = data['입국자수'].rolling(3).sum().shift(1)
    data['전월_차이'] = data['입국자수'] - data['전월']
    data['전년_차이'] = data['입국자수'] - data['전년']
    data['월'] = data['월'].astype(int)

    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(data) < 36:
        print(f"\n❌ {국적} / {목적} 조합은 학습 가능한 데이터가 부족합니다. (36개월 이상 필요)")
        continue

    features = ['전월', '전전월', '전년', '전전년', '전월증감률', '전년증감률',
                '이동평균3', '이동평균6', '이동평균12', '누적합3', '전월_차이', '전년_차이', '월']
    target = '입국자수'

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(data[features])
    y_scaled = scaler_y.fit_transform(np.log1p(data[[target]]))

    sequence_length = 24
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y_scaled[i + sequence_length])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    X_train, y_train = X_seq, y_seq

    model = Sequential([
        LayerNormalization(input_shape=(sequence_length, X_seq.shape[2])),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(loss='huber', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007))
    model.fit(X_train, y_train, epochs=150, batch_size=8, verbose=0)

    y_pred = model.predict(X_train)
    y_pred_inv = np.expm1(scaler_y.inverse_transform(y_pred).flatten())
    y_true_inv = np.expm1(scaler_y.inverse_transform(y_train).flatten())

    r2 = r2_score(y_true_inv, y_pred_inv)
    nonzero_idx = y_true_inv != 0
    mape = mean_absolute_percentage_error(y_true_inv[nonzero_idx], y_pred_inv[nonzero_idx]) * 100 if np.any(nonzero_idx) else np.nan
    rmse = mean_squared_error(y_true_inv, y_pred_inv, squared=False)
    dw = durbin_watson(y_true_inv - y_pred_inv)
    신뢰도 = max(0, min(100, 100 - mape))

    print(f"\n📈 LSTM 성능 평가 ({국적} / {목적})")
    print(f" - R^2: {r2:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - 신뢰도: {신뢰도:.1f}%")
    print(f" - Durbin-Watson: {dw:.4f}")

    future_preds = []
    recent_inputs = X_scaled[-sequence_length:].tolist()
    recent_raw = data.copy()

    for month in 예측월:
        row = recent_raw.iloc[-1:].copy()
        row['입국자수'] = future_preds[-1] if future_preds else row['입국자수']
        row['전월'] = recent_raw['입국자수'].iloc[-1]
        row['전전월'] = recent_raw['입국자수'].iloc[-2]
        row['전년'] = recent_raw['입국자수'].iloc[-12]
        row['전전년'] = recent_raw['입국자수'].iloc[-24]
        row['전월증감률'] = (recent_raw['입국자수'].iloc[-1] - recent_raw['입국자수'].iloc[-2]) / recent_raw['입국자수'].iloc[-2]
        row['전년증감률'] = (recent_raw['입국자수'].iloc[-1] - recent_raw['입국자수'].iloc[-12]) / recent_raw['입국자수'].iloc[-12]
        row['이동평균3'] = recent_raw['입국자수'].iloc[-3:].mean()
        row['이동평균6'] = recent_raw['입국자수'].iloc[-6:].mean()
        row['이동평균12'] = recent_raw['입국자수'].iloc[-12:].mean()
        row['누적합3'] = recent_raw['입국자수'].iloc[-3:].sum()
        row['전월_차이'] = row['입국자수'] - row['전월']
        row['전년_차이'] = row['입국자수'] - row['전년']
        row['월'] = ((recent_raw['일자'].dt.month.iloc[-1] + len(future_preds)) - 1) % 12 + 1

        row = row[features].fillna(method='ffill', axis=1)
        row_scaled = scaler_x.transform(row)
        recent_inputs.append(row_scaled[0])
        input_seq = np.array(recent_inputs[-sequence_length:]).reshape(1, sequence_length, len(features))

        pred_scaled = model.predict(input_seq)[0][0]
        pred = np.expm1(scaler_y.inverse_transform([[pred_scaled]])[0][0])
        future_preds.append(pred)
        recent_raw = pd.concat([recent_raw, pd.DataFrame([{'입국자수': pred}])], ignore_index=True)

        pred_date = pd.to_datetime(f"{예측연도}-{month:02d}-01")
        예측값 = np.round(pred).astype(int)
        print(f"\n🔮 {예측연도}년 {month}월 예측 입국자 수: {예측값:,}명")

    pred_dates = [pd.to_datetime(f"{예측연도}-{m:02d}-01") for m in 예측월]
    pred_values = [np.round(p).astype(int) for p in future_preds[-len(예측월):]]

    plt.figure(figsize=(14, 6))
    plt.plot(data['일자'].iloc[-12:], data['입국자수'].iloc[-12:], label="최근 실측", color='blue', linewidth=2)
    plt.plot(pred_dates, pred_values, color='red', marker='o', linestyle='--', linewidth=2, label='예측값')
    for date, val in zip(pred_dates, pred_values):
        plt.scatter(date, val, color='red', s=60, edgecolors='black', zorder=5)
        plt.text(date, val + max(pred_values)*0.02, f"{val:,}", ha='center', va='bottom', fontsize=9)
    plt.title(f"{국적}/{목적} - {예측연도}년 {예측월}월 예측 결과", fontsize=14, fontweight='bold')
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
import re, warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text): return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도 = int(input("예측할 연도 입력 (예: 2025): ").strip())
예측월입력 = input("예측할 월 입력 (예: 10,11,12 또는 7~9): ").strip()

if '~' in 예측월입력:
    start, end = map(int, 예측월입력.split('~'))
    예측월 = list(range(start, end + 1))
elif ',' in 예측월입력:
    예측월 = list(map(int, 예측월입력.split(',')))
else:
    예측월 = [int(예측월입력)]

path = './data/외국인입국자_전처리완료_딥러닝용.csv'
df = pd.read_csv(path, encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

국적목록 = df['국적'].unique()
목적목록 = df['목적'].unique()

if 선택_국적:
    match = [nat for nat in 국적목록 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); exit()

if 선택_목적:
    match = [pur for pur in 목적목록 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); exit()

대상목록 = df[df['국적'] == 선택_국적][['국적', '목적']].drop_duplicates()
if 선택_목적:
    대상목록 = 대상목록[대상목록['목적'] == 선택_목적]

for _, row in 대상목록.iterrows():
    국적, 목적 = row['국적'], row['목적']
    data = df[(df['국적'] == 국적) & (df['목적'] == 목적)].sort_values(['연도', '월'])
    data['연월'] = data['연도'].astype(str) + '-' + data['월'].astype(str).str.zfill(2)
    data['일자'] = pd.to_datetime(data['연월'])

    data['전월'] = data['입국자수'].shift(1)
    data['전전월'] = data['입국자수'].shift(2)
    data['전년'] = data['입국자수'].shift(12)
    data['전전년'] = data['입국자수'].shift(24)
    data['전월증감률'] = data['입국자수'].pct_change().shift(1)
    data['전년증감률'] = (data['입국자수'] - data['전년']) / data['전년']
    data['이동평균3'] = data['입국자수'].rolling(3).mean().shift(1)
    data['이동평균6'] = data['입국자수'].rolling(6).mean().shift(1)
    data['이동평균12'] = data['입국자수'].rolling(12).mean().shift(1)
    data['누적합3'] = data['입국자수'].rolling(3).sum().shift(1)
    data['전월_차이'] = data['입국자수'] - data['전월']
    data['전년_차이'] = data['입국자수'] - data['전년']
    data['월'] = data['월'].astype(int)

    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(data) < 36:
        print(f"\n❌ {국적} / {목적} 조합은 학습 가능한 데이터가 부족합니다. (36개월 이상 필요)")
        continue

    features = ['전월', '전전월', '전년', '전전년', '전월증감률', '전년증감률',
                '이동평균3', '이동평균6', '이동평균12', '누적합3', '전월_차이', '전년_차이', '월']
    target = '입국자수'

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(data[features])
    y_scaled = scaler_y.fit_transform(np.log1p(data[[target]]))

    sequence_length = 24
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y_scaled[i + sequence_length])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    X_train, y_train = X_seq, y_seq

    model = Sequential([
        LayerNormalization(input_shape=(sequence_length, X_seq.shape[2])),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='huber', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
    model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0)

    # 이하 동일...

    plt.show()


# %%
# LSTM model 코드
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization, Attention, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
import re, warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text): return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도 = int(input("예측할 연도 입력 (예: 2025): ").strip())
예측월입력 = input("예측할 월 입력 (예: 10,11,12 또는 7~9): ").strip()

if '~' in 예측월입력:
    start, end = map(int, 예측월입력.split('~'))
    예측월 = list(range(start, end + 1))
elif ',' in 예측월입력:
    예측월 = list(map(int, 예측월입력.split(',')))
else:
    예측월 = [int(예측월입력)]

path = './data/외국인입국자_전처리완료_딥러닝용.csv'
df = pd.read_csv(path, encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

국적목록 = df['국적'].unique()
목적목록 = df['목적'].unique()

if 선택_국적:
    match = [nat for nat in 국적목록 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n👉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); exit()

if 선택_목적:
    match = [pur for pur in 목적목록 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); exit()

대상목록 = df[df['국적'] == 선택_국적][['국적', '목적']].drop_duplicates()
if 선택_목적:
    대상목록 = 대상목록[대상목록['목적'] == 선택_목적]

for _, row in 대상목록.iterrows():
    국적, 목적 = row['국적'], row['목적']
    data = df[(df['국적'] == 국적) & (df['목적'] == 목적)].sort_values(['연도', '월'])
    data['연월'] = data['연도'].astype(str) + '-' + data['월'].astype(str).str.zfill(2)
    data['일자'] = pd.to_datetime(data['연월'])

    data['전월'] = data['입국자수'].shift(1)
    data['전전월'] = data['입국자수'].shift(2)
    data['전년'] = data['입국자수'].shift(12)
    data['전전년'] = data['입국자수'].shift(24)
    data['전월증감률'] = data['입국자수'].pct_change().shift(1)
    data['전년증감률'] = (data['입국자수'] - data['전년']) / data['전년']
    data['이동평균3'] = data['입국자수'].rolling(3).mean().shift(1)
    data['이동평균6'] = data['입국자수'].rolling(6).mean().shift(1)
    data['이동평균12'] = data['입국자수'].rolling(12).mean().shift(1)
    data['누적합3'] = data['입국자수'].rolling(3).sum().shift(1)
    data['전월_차이'] = data['입국자수'] - data['전월']
    data['전년_차이'] = data['입국자수'] - data['전년']
    data['월'] = data['월'].astype(int)

    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(data) < 36:
        print(f"\n❌ {국적} / {목적} 조합은 학습 가능한 데이터가 부족합니다. (36개월 이상 필요)")
        continue

    features = ['전월', '전전월', '전년', '전전년', '전월증감률', '전년증감률',
                '이동평균3', '이동평균6', '이동평균12', '누적합3', '전월_차이', '전년_차이', '월']
    target = '입국자수'

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(data[features])
    y_scaled = scaler_y.fit_transform(np.log1p(data[[target]]))

    sequence_length = 24
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y_scaled[i + sequence_length])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    X_train, y_train = X_seq, y_seq

    inputs = Input(shape=(sequence_length, X_seq.shape[2]))
    x = LayerNormalization()(inputs)
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32, return_sequences=True, recurrent_dropout=0.2))(x)
    x = Attention()([x, x])
    x = tf.keras.layers.Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)

    model.compile(loss='huber', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=150, batch_size=8, verbose=0, callbacks=[early_stop])

    y_pred = model.predict(X_train)
    y_pred_inv = np.expm1(scaler_y.inverse_transform(y_pred).flatten())
    y_true_inv = np.expm1(scaler_y.inverse_transform(y_train).flatten())

    r2 = r2_score(y_true_inv, y_pred_inv)
    nonzero_idx = y_true_inv != 0
    mape = mean_absolute_percentage_error(y_true_inv[nonzero_idx], y_pred_inv[nonzero_idx]) * 100
    rmse = mean_squared_error(y_true_inv, y_pred_inv, squared=False)
    dw = durbin_watson(y_true_inv - y_pred_inv)
    신뢰도 = max(0, min(100, 100 - mape))

    print(f"\n📈 LSTM 성능 평가 ({국적} / {목적})")
    print(f" - R^2: {r2:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - 신뢰도: {신뢰도:.1f}%")
    print(f" - Durbin-Watson: {dw:.4f}")

    future_preds = []
    recent_inputs = X_scaled[-sequence_length:].tolist()
    recent_raw = data.copy()

    for month in 예측월:
        row = recent_raw.iloc[-1:].copy()
        row['입국자수'] = future_preds[-1] if future_preds else row['입국자수']
        row['전월'] = recent_raw['입국자수'].iloc[-1]
        row['전전월'] = recent_raw['입국자수'].iloc[-2]
        row['전년'] = recent_raw['입국자수'].iloc[-12]
        row['전전년'] = recent_raw['입국자수'].iloc[-24]
        row['전월증감률'] = (recent_raw['입국자수'].iloc[-1] - recent_raw['입국자수'].iloc[-2]) / recent_raw['입국자수'].iloc[-2]
        row['전년증감률'] = (recent_raw['입국자수'].iloc[-1] - recent_raw['입국자수'].iloc[-12]) / recent_raw['입국자수'].iloc[-12]
        row['이동평균3'] = recent_raw['입국자수'].iloc[-3:].mean()
        row['이동평균6'] = recent_raw['입국자수'].iloc[-6:].mean()
        row['이동평균12'] = recent_raw['입국자수'].iloc[-12:].mean()
        row['누적합3'] = recent_raw['입국자수'].iloc[-3:].sum()
        row['전월_차이'] = row['입국자수'] - row['전월']
        row['전년_차이'] = row['입국자수'] - row['전년']
        row['월'] = ((recent_raw['일자'].dt.month.iloc[-1] + len(future_preds)) - 1) % 12 + 1

        row = row[features].fillna(method='ffill', axis=1)
        row_scaled = scaler_x.transform(row)
        recent_inputs.append(row_scaled[0])
        input_seq = np.array(recent_inputs[-sequence_length:]).reshape(1, sequence_length, len(features))

        pred_scaled = model.predict(input_seq)[0][0]
        pred = np.expm1(scaler_y.inverse_transform([[pred_scaled]])[0][0])
        future_preds.append(pred)
        recent_raw = pd.concat([recent_raw, pd.DataFrame([{'입국자수': pred}])], ignore_index=True)

        pred_date = pd.to_datetime(f"{예측연도}-{month:02d}-01")
        예측값 = np.round(pred).astype(int)
        print(f"\n🔮 {예측연도}년 {month}월 예측 입국자 수: {예측값:,}명")

    pred_dates = [pd.to_datetime(f"{예측연도}-{m:02d}-01") for m in 예측월]
    pred_values = [np.round(p).astype(int) for p in future_preds[-len(예측월):]]

    plt.figure(figsize=(14, 6))
    plt.plot(data['일자'].iloc[-12:], data['입국자수'].iloc[-12:], label="최근 실측", color='blue', linewidth=2)
    plt.plot(pred_dates, pred_values, color='red', marker='o', linestyle='--', linewidth=2, label='예측값')
    for date, val in zip(pred_dates, pred_values):
        plt.scatter(date, val, color='red', s=60, edgecolors='black', zorder=5)
        plt.text(date, val + max(pred_values)*0.02, f"{val:,}", ha='center', va='bottom', fontsize=9)
    plt.title(f"{국적}/{목적} - {예측연도}년 {예측월}월 예측 결과", fontsize=14, fontweight='bold')
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
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
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도 = int(input("예측할 연도 입력 (예: 2025): ").strip())
예측월입력 = input("예측할 월 입력 (1~12 또는 3,6,9 또는 10~12): ").strip()

if '~' in 예측월입력:
    start, end = map(int, 예측월입력.split('~'))
    예측월 = list(range(start, end + 1))
elif ',' in 예측월입력:
    예측월 = list(map(int, 예측월입력.split(',')))
else:
    예측월 = [int(예측월입력)]

예측_목록 = [pd.to_datetime(f"{예측연도}-{m:02d}-01") for m in 예측월]

path = './data/외국인입국자_전처리완료_딥러닝용.csv'
df = pd.read_csv(path, encoding='utf-8')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

unique_국적 = df['국적'].dropna().unique()
unique_목적 = df['목적'].dropna().unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n🕉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
targets = targets[targets["count"] > 36]
if 선택_국적:
    targets = targets[targets["국적"] == 선택_국적]
if 선택_목적:
    targets = targets[targets["목적"] == 선택_목적]

for _, row in targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values(["연도", "월"]).reset_index(drop=True)
    if len(df_filtered) < 36:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    df_filtered['연월'] = df_filtered['연도'].astype(str) + '-' + df_filtered['월'].astype(str).str.zfill(2)
    df_filtered['일자'] = pd.to_datetime(df_filtered['연월'])
    df_filtered['시계열순서'] = np.arange(len(df_filtered))
    df_filtered['전월'] = df_filtered['입국자수'].shift(1)
    df_filtered['전년'] = df_filtered['입국자수'].shift(12)
    df_filtered['전월증감률'] = df_filtered['입국자수'].pct_change().shift(1)
    df_filtered['전년증감률'] = (df_filtered['입국자수'] - df_filtered['전년']) / df_filtered['전년']
    df_filtered['이동평균3'] = df_filtered['입국자수'].rolling(3).mean().shift(1)
    df_filtered['이동평균6'] = df_filtered['입국자수'].rolling(6).mean().shift(1)
    df_filtered['이동평균12'] = df_filtered['입국자수'].rolling(12).mean().shift(1)
    df_filtered['평균증감률'] = (df_filtered['이동평균6'] - df_filtered['이동평균12']) / df_filtered['이동평균12']
    df_filtered['전월_차이'] = df_filtered['입국자수'] - df_filtered['전월']
    df_filtered['전년_차이'] = df_filtered['입국자수'] - df_filtered['전년']
    df_filtered['누적입국자수'] = df_filtered['입국자수'].cumsum()
    df_filtered['12개월편차'] = df_filtered['입국자수'] - df_filtered['이동평균12']
    df_filtered['계절성지표'] = df_filtered['월'].apply(lambda x: np.sin(2 * np.pi * x / 12))
    df_filtered['계절성지표_cos'] = df_filtered['월'].apply(lambda x: np.cos(2 * np.pi * x / 12))
    df_filtered['월'] = df_filtered['월'].astype(int)

    df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna().copy()

    features = ['시계열순서','전월', '전년', '전월증감률', '전년증감률', '이동평균3', '이동평균6', '이동평균12',
                '평균증감률', '전월_차이', '전년_차이', '월', '누적입국자수', '12개월편차', '계절성지표', '계절성지표_cos']
    y = np.log1p(df_filtered["입국자수"]) if 목적 == '유학연수' else df_filtered["입국자수"]

    X = df_filtered[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test = X_scaled[:-6], X_scaled[-6:]
    y_train, y_test = y[:-6], y[-6:]

    model = XGBRegressor(n_estimators=700, learning_rate=0.02, max_depth=5, subsample=0.85, colsample_bytree=0.9, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_eval = np.expm1(y_pred) if 목적 == '유학연수' else y_pred
    y_test_eval = np.expm1(y_test) if 목적 == '유학연수' else y_test

    r2 = r2_score(y_test_eval, y_pred_eval)
    mape = mean_absolute_percentage_error(y_test_eval, y_pred_eval) * 100
    rmse = mean_squared_error(y_test_eval, y_pred_eval, squared=False)
    dw = durbin_watson(y_test_eval - y_pred_eval)
    신뢰도 = max(0, min(100, 100 - mape))

    print("✅ XGBoost 성능 평가")
    print(f" - R^2: {r2:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - 신뢰도: {신뢰도:.1f}%")
    print(f" - Durbin-Watson: {dw:.4f}")

    future_preds = []
    recent = df_filtered.copy()
    for pred_date in 예측_목록:
        row = recent.iloc[-1:].copy()
        row['입국자수'] = future_preds[-1] if future_preds else row['입국자수']
        row['시계열순서'] = len(recent)
        row['전월'] = recent['입국자수'].iloc[-1]
        row['전년'] = recent['입국자수'].iloc[-12]
        row['전월증감률'] = (recent['입국자수'].iloc[-1] - recent['입국자수'].iloc[-2]) / recent['입국자수'].iloc[-2]
        row['전년증감률'] = (recent['입국자수'].iloc[-1] - recent['입국자수'].iloc[-12]) / recent['입국자수'].iloc[-12]
        row['이동평균3'] = recent['입국자수'].iloc[-3:].mean()
        row['이동평균6'] = recent['입국자수'].iloc[-6:].mean()
        row['이동평균12'] = recent['입국자수'].iloc[-12:].mean()
        row['평균증감률'] = (row['이동평균6'] - row['이동평균12']) / row['이동평균12']
        row['전월_차이'] = row['입국자수'] - row['전월']
        row['전년_차이'] = row['입국자수'] - row['전년']
        row['누적입국자수'] = recent['입국자수'].cumsum().iloc[-1] + row['입국자수']
        row['12개월편차'] = row['입국자수'] - row['이동평균12']
        row['계절성지표'] = np.sin(2 * np.pi * pred_date.month / 12)
        row['계절성지표_cos'] = np.cos(2 * np.pi * pred_date.month / 12)
        row['월'] = pred_date.month

        row = row[features].fillna(method='ffill', axis=1)
        row_scaled = scaler.transform(row)
        pred = model.predict(row_scaled)[0]
        pred = np.expm1(pred) if 목적 == '유학연수' else pred
        future_preds.append(pred)
        recent = pd.concat([recent, pd.DataFrame([{'입국자수': pred}])], ignore_index=True)

        예측값 = np.round(pred).astype(int)
        print(f"\n🔮 {pred_date.year}년 {pred_date.month}월 예측 입국자 수: {예측값:,}명 (신뢰도: {신뢰도:.1f}%)")

    pred_values = [np.round(p).astype(int) for p in future_preds]
    plt.figure(figsize=(14, 6))
    시각화데이터 = df_filtered[-6:].copy()
    plt.plot(시각화데이터['일자'], 시각화데이터['입국자수'], label="최근 실측", color='blue', linewidth=2)
    plt.plot(예측_목록, pred_values, color='red', marker='o', linestyle='--', linewidth=2, label='예측값')
    for date, val in zip(예측_목록, pred_values):
        plt.scatter(date, val, color='red', s=60, edgecolors='black', zorder=5)
        plt.text(date, val + max(pred_values)*0.02, f"{val:,}", ha='center', va='bottom', fontsize=9)
    plt.title(f"{국적}/{목적} - {예측연도}년 {예측월}월 예측 결과", fontsize=14, fontweight='bold')
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from statsmodels.stats.stattools import durbin_watson
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"\s+", "", str(text)).strip().lower()

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도 = int(input("예측할 연도 입력 (예: 2025): ").strip())
예측월입력 = input("예측할 월 입력 (1~12 또는 3,6,9 또는 10~12): ").strip()

if '~' in 예측월입력:
    start, end = map(int, 예측월입력.split('~'))
    예측월 = list(range(start, end + 1))
elif ',' in 예측월입력:
    예측월 = list(map(int, 예측월입력.split(',')))
else:
    예측월 = [int(예측월입력)]

예측_목록 = [pd.to_datetime(f"{예측연도}-{m:02d}-01") for m in 예측월]

df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

if 선택_국적:
    match = [nat for nat in df['국적'].unique() if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n🕉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        선택_국적 = ""

if 선택_목적:
    match = [pur for pur in df['목적'].unique() if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index().rename(columns={0: "count"})
targets = targets[targets["count"] > 36]
if 선택_국적:
    targets = targets[targets["국적"] == 선택_국적]
if 선택_목적:
    targets = targets[targets["목적"] == 선택_목적]

for _, row in targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    df_filtered = df[(df["국적"] == 국적) & (df["목적"] == 목적)].sort_values(["연도", "월"]).reset_index(drop=True)
    if len(df_filtered) < 36:
        continue

    print(f"\n📌 예측 대상: {국적} / {목적} ({len(df_filtered)} 개월치)")

    df_filtered['연월'] = df_filtered['연도'].astype(str) + '-' + df_filtered['월'].astype(str).str.zfill(2)
    df_filtered['일자'] = pd.to_datetime(df_filtered['연월'])
    ts = df_filtered.set_index('일자')['입국자수']

    df_filtered['입국자수_lag1'] = df_filtered['입국자수'].shift(1)
    df_filtered['입국자수_lag2'] = df_filtered['입국자수'].shift(2)
    df_filtered['입국자수_lag3'] = df_filtered['입국자수'].shift(3)
    df_filtered['입국자수_lag12'] = df_filtered['입국자수'].shift(12)
    df_filtered['입국자수_ma3'] = df_filtered['입국자수'].rolling(3).mean()
    df_filtered['입국자수_ma6'] = df_filtered['입국자수'].rolling(6).mean()
    df_filtered['입국자수_std3'] = df_filtered['입국자수'].rolling(3).std()
    df_filtered['입국자수_증감률'] = df_filtered['입국자수'].pct_change()
    df_filtered['월'] = df_filtered['월'].astype(int)
    df_filtered['연도'] = df_filtered['연도'].astype(int)
    df_filtered['코로나'] = (df_filtered['연도'].between(2020, 2022)).astype(int)
    df_filtered.dropna(inplace=True)

    feature_cols = ['입국자수_lag1', '입국자수_lag2', '입국자수_lag3', '입국자수_lag12', '입국자수_ma3', '입국자수_ma6', '입국자수_std3', '입국자수_증감률', '월', '연도', '코로나']
    X = df_filtered[feature_cols]
    y = df_filtered['입국자수']
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]  # X에 맞춰 y도 정렬


    model = XGBRegressor(n_estimators=1500, learning_rate=0.01, max_depth=6, subsample=0.9, colsample_bytree=0.9, early_stopping_rounds=30, n_jobs=-1, random_state=42)
    model.fit(X, y, eval_set=[(X, y)], verbose=False)

    preds = []
    temp_df = df_filtered.copy()
    for date in 예측_목록:
        lag1 = temp_df.iloc[-1]['입국자수']
        lag2 = temp_df.iloc[-2]['입국자수']
        lag3 = temp_df.iloc[-3]['입국자수']
        lag12 = temp_df.iloc[-12]['입국자수'] if len(temp_df) >= 12 else lag1
        ma3 = temp_df['입국자수'].iloc[-3:].mean()
        ma6 = temp_df['입국자수'].iloc[-6:].mean()
        std3 = temp_df['입국자수'].iloc[-3:].std()
        roc = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        month = date.month
        year = date.year
        corona = int(2020 <= year <= 2022)

        x_input = pd.DataFrame([[lag1, lag2, lag3, lag12, ma3, ma6, std3, roc, month, year, corona]], columns=feature_cols)
        pred = model.predict(x_input)[0]
        preds.append(pred)

        new_row = pd.DataFrame([[lag1, lag2, lag3, lag12, ma3, ma6, std3, roc, month, year, corona, pred]], columns=feature_cols + ['입국자수'])
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    y_true = y[-12:]
    y_pred = model.predict(X[-12:])
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    dw = durbin_watson(y_true - y_pred)
    신뢰도 = round(100 - mape, 1)

    print(f"✅ XGBoost 성능 평가")
    print(f" - R^2: {r2:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - 신뢰도: {신뢰도}%")
    print(f" - Durbin-Watson: {dw:.4f}\n")

    for date, val in zip(예측_목록, preds):
        print(f"🔮 {date.year}년 {date.month}월 예측 입국자 수: {int(val):,}명 (신뢰도: {신뢰도}%)")

    plt.figure(figsize=(14, 6))
    plt.plot(ts.index[-24:], ts.values[-24:], label='최근 실측', color='blue', linewidth=2)
    plt.plot(예측_목록, preds, label='예측값(XGBoost)', color='red', linestyle='--', marker='o')
    for d, p in zip(예측_목록, preds):
        plt.text(d, p, f"{int(p):,}", ha='center', va='bottom', fontsize=9, color='red')
    plt.title(f"{국적}/{목적} - XGBoost 예측 결과", fontsize=14, fontweight='bold')
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import warnings
import re

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def normalize(text):
    return re.sub(r"\s+", "", str(text)).strip().lower() if pd.notnull(text) else ""

선택_국적 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
선택_목적 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도 = int(input("예측할 연도 입력 (예: 2026): ").strip())
예측월입력 = input("예측할 월 입력 (예: 1~6 또는 3,6,9): ").strip()

if '~' in 예측월입력:
    start, end = map(int, 예측월입력.split('~'))
    예측월 = list(range(start, end + 1))
elif ',' in 예측월입력:
    예측월 = list(map(int, 예측월입력.split(',')))
else:
    예측월 = [int(예측월입력)]

예측_목록 = [pd.to_datetime(f"{예측연도}-{m:02d}-01") for m in 예측월]

df = pd.read_csv('./data/외국인입국자_전처리완료_딥러닝용.csv')
df.columns = df.columns.str.strip()
df['국적'] = df['국적'].astype(str).str.strip()
df['목적'] = df['목적'].astype(str).str.strip()

unique_국적 = df['국적'].unique()
unique_목적 = df['목적'].unique()

if 선택_국적:
    match = [nat for nat in unique_국적 if normalize(선택_국적) in normalize(nat)]
    if match:
        선택_국적 = match[0]
        print(f"\n🕉 입력한 국적과 유사한 값으로 '{선택_국적}' 사용")
    else:
        print("❌ 유효한 국적이 아닙니다."); 선택_국적 = ""

if 선택_목적:
    match = [pur for pur in unique_목적 if normalize(선택_목적) in normalize(pur)]
    if match:
        선택_목적 = match[0]
        print(f"👉 입력한 목적과 유사한 값으로 '{선택_목적}' 사용")
    else:
        print("❌ 유효한 목적이 아닙니다."); 선택_목적 = ""

targets = df.groupby(["국적", "목적"]).size().reset_index(name="count")
targets = targets[targets["count"] > 36]
if 선택_국적:
    targets = targets[targets["국적"] == 선택_국적]
if 선택_목적:
    targets = targets[targets["목적"] == 선택_목적]

for _, row in targets.iterrows():
    국적, 목적 = row["국적"], row["목적"]
    data = df[(df["국적"] == 국적) & (df["목적"] == 목적)].copy()
    data['일자'] = pd.to_datetime(data['연도'].astype(str) + '-' + data['월'].astype(str).str.zfill(2) + '-01')
    data = data.sort_values('일자')

    data['시계열순서'] = np.arange(len(data))
    data['전월'] = data['입국자수'].shift(1)
    data['전년'] = data['입국자수'].shift(12)
    data['전월증감률'] = data['입국자수'].pct_change().shift(1)
    data['전년증감률'] = (data['입국자수'] - data['전년']) / data['전년']
    data['이동평균6'] = data['입국자수'].rolling(6).mean().shift(1)
    data['이동평균12'] = data['입국자수'].rolling(12).mean().shift(1)
    data['월'] = data['월'].astype(int)
    data['계절성'] = np.sin(2 * np.pi * data['월'] / 12)
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    features = ['시계열순서', '전월', '전년', '전월증감률', '전년증감률',
                '이동평균6', '이동평균12', '월', '계절성']
    X = data[features]
    y = data['입국자수']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train = X_scaled
    y_train = y

    xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, random_state=42)
    xgb.fit(X_train, y_train)

    residuals = y_train.values - xgb.predict(X_train)
    seq_len = 12
    def make_sequences(series, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(series) - seq_len):
            X_seq.append(series[i:i+seq_len])
            y_seq.append(series[i+seq_len])
        return np.array(X_seq), np.array(y_seq)

    if len(residuals) < seq_len + 1:
        print(f"\n⚠️ 잔차 데이터가 부족하여 GRU 학습 생략 ({국적}/{목적})")
        continue

    X_seq, y_seq = make_sequences(residuals, seq_len)
    X_seq = X_seq.reshape((X_seq.shape[0], seq_len, 1))

    gru_model = Sequential([
        GRU(64, input_shape=(seq_len, 1)),
        Dense(1)
    ])
    gru_model.compile(optimizer='adam', loss='mse')
    gru_model.fit(X_seq, y_seq, epochs=100, batch_size=8, verbose=0)

    future_preds = []
    recent_df = data.copy()

    for pred_date in 예측_목록:
        row = recent_df.iloc[-1:].copy()
        row['입국자수'] = future_preds[-1] if future_preds else row['입국자수'].values[0]
        row['시계열순서'] = len(recent_df)
        row['전월'] = recent_df['입국자수'].iloc[-1]
        row['전년'] = recent_df['입국자수'].iloc[-12] if len(recent_df) >= 12 else recent_df['입국자수'].mean()
        row['전월증감률'] = (recent_df['입국자수'].iloc[-1] - recent_df['입국자수'].iloc[-2]) / recent_df['입국자수'].iloc[-2]
        row['전년증감률'] = (recent_df['입국자수'].iloc[-1] - recent_df['입국자수'].iloc[-12]) / recent_df['입국자수'].iloc[-12]
        row['이동평균6'] = recent_df['입국자수'].iloc[-6:].mean()
        row['이동평균12'] = recent_df['입국자수'].iloc[-12:].mean()
        row['월'] = pred_date.month
        row['계절성'] = np.sin(2 * np.pi * pred_date.month / 12)
        row = row[features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        row_scaled = scaler.transform(row)

        xgb_pred = xgb.predict(row_scaled)[0]
        recent_residuals = residuals[-seq_len:].tolist()
        for f in future_preds:
            recent_residuals.append(f - xgb.predict(row_scaled)[0])
        recent_residuals = recent_residuals[-seq_len:]
        gru_input = np.array(recent_residuals).reshape(1, seq_len, 1)
        gru_pred = gru_model.predict(gru_input, verbose=0)[0][0]

        최종예측값 = xgb_pred + gru_pred
        future_preds.append(최종예측값)
        recent_df = pd.concat([recent_df, pd.DataFrame([{'입국자수': 최종예측값}])], ignore_index=True)

        print(f"\n🔮 {pred_date.year}년 {pred_date.month}월 예측 입국자 수: {int(round(최종예측값)):,}명")

    mape = mean_absolute_percentage_error(y_train.values[-len(future_preds):], future_preds) * 100
    신뢰도 = max(0, min(100, 100 - mape))
    print(f"\n✅ 예측 신뢰도 (최근값 기준): {신뢰도:.1f}%")

    dates = [d for d in 예측_목록]
    values = [int(round(v)) for v in future_preds]

    plt.figure(figsize=(12, 6))
    plt.plot(data['일자'][-12:], data['입국자수'].values[-12:], label='실제값')
    plt.plot(dates, values, label='예측값', marker='o', linestyle='--', color='red')
    for d, v in zip(dates, values):
        plt.text(d, v, f"{v:,}", ha='center', va='bottom', fontsize=9)
    plt.title(f"{국적} / {목적} - {예측연도}년 예측 결과", fontsize=14)
    plt.xlabel("월")
    plt.ylabel("입국자 수")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


# %%
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import calendar
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ✅ 사용자 입력
국적_입력 = input("예측할 국적 입력 (전체 예측 원하면 Enter): ").strip()
목적_입력 = input("예측할 목적 입력 (전체 예측 원하면 Enter): ").strip()
예측연도_입력 = input("예측할 연도 입력 (예: 2026): ").strip()
예측월_입력 = input("예측할 월 범위 입력 (예: 1~12 또는 3,6,9): ").strip()

if 예측연도_입력.isdigit():
    예측연도 = int(예측연도_입력)
else:
    예측연도 = 2026

if '~' in 예측월_입력:
    start_month, end_month = map(int, 예측월_입력.split('~'))
    예측월리스트 = list(range(start_month, end_month + 1))
elif 예측월_입력:
    예측월리스트 = list(map(int, 예측월_입력.split(',')))
else:
    예측월리스트 = list(range(1, 13))

# ✅ 데이터 불러오기
파일경로 = "./data/외국인입국자_전처리완료_딥러닝용.csv"
df = pd.read_csv(파일경로)

if '년월' not in df.columns:
    if '연도' in df.columns and '월' in df.columns:
        df['년월'] = pd.to_datetime(df['연도'].astype(str) + '-' + df['월'].astype(str).str.zfill(2))
    else:
        raise KeyError("'년월' 또는 '연도', '월' 컬럼이 필요합니다.")

df['입국자수'] = df['입국자수'].astype(int)
df['연도'] = df['년월'].dt.year
df['월'] = df['년월'].dt.month
df['연도편차'] = df['연도'] - df['연도'].min()
df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)

MIN_MONTHS = 24
선택_국적 = 국적_입력 if 국적_입력 else None
선택_목적 = 목적_입력 if 목적_입력 else None

유효조합 = df.groupby(['국적', '목적']).size().reset_index(name='count')
유효조합 = 유효조합[유효조합['count'] >= MIN_MONTHS]

if 선택_국적:
    유효조합 = 유효조합[유효조합['국적'].str.contains(선택_국적)]
if 선택_목적:
    유효조합 = 유효조합[유효조합['목적'].str.contains(선택_목적)]

if 유효조합.empty:
    print("⛔ 해당 조건에 맞는 데이터가 없습니다.")
    exit()

for _, row in 유효조합.iterrows():
    국적, 목적 = row['국적'], row['목적']
    data = df[(df['국적'] == 국적) & (df['목적'] == 목적)].copy()

    feature_cols = ['연도편차', '월', '월_cos', '월_sin']
    X = data[feature_cols]
    y = data['입국자수']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test = X_scaled[:-6], X_scaled[-6:]
    y_train, y_test = y[:-6], y[-6:]

    param_dict = {
        '공용': {'n_estimators': 300, 'max_depth': 2},
        '관광': {'n_estimators': 300, 'max_depth': 3},
        '상용': {'n_estimators': 200, 'max_depth': 2},
        '유학연수': {'n_estimators': 250, 'max_depth': 2}
    }
    params = param_dict.get(목적, {'n_estimators': 250, 'max_depth': 2})

    xgb = XGBRegressor(**params)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_train)
    residuals = y_train.values - pred_xgb

    # GRU 보정 모델 학습
    model_gru = None
    if len(residuals) >= 18:
        seq_len = 18
        X_seq, y_seq = [], []
        for i in range(len(residuals) - seq_len):
            X_seq.append(residuals[i:i + seq_len])
            y_seq.append(residuals[i + seq_len])
        X_seq = np.array(X_seq).reshape(-1, seq_len, 1)
        y_seq = np.array(y_seq)

        model_gru = Sequential([
            GRU(64, return_sequences=True, input_shape=(seq_len, 1)),
            Dropout(0.2),
            GRU(32),
            Dense(1)
        ])
        model_gru.compile(optimizer='adam', loss='mse')
        model_gru.fit(X_seq, y_seq, epochs=150, verbose=0)

    # 미래 예측
    data_min_year = df['연도'].min()
    미래_기간 = pd.date_range(start=f"{예측연도}-01-01", end=f"{예측연도}-12-01", freq="MS")
    future = pd.DataFrame({
        '년월': 미래_기간,
        '연도': 미래_기간.year,
        '월': 미래_기간.month
    })
    future = future[future['월'].isin(예측월리스트)].copy()
    future['연도편차'] = future['연도'] - data_min_year
    future['월_cos'] = np.cos(2 * np.pi * future['월'] / 12)
    future['월_sin'] = np.sin(2 * np.pi * future['월'] / 12)

    X_future = scaler.transform(future[feature_cols])
    pred_future = xgb.predict(X_future)

    if model_gru:
        recent_residuals = residuals[-seq_len:].tolist()
        residual_preds = []
        for _ in range(len(future)):
            seq_input = np.array(recent_residuals[-seq_len:]).reshape(1, seq_len, 1)
            pred_resid = model_gru.predict(seq_input, verbose=0)[0][0]
            residual_preds.append(pred_resid)
            recent_residuals.append(pred_resid)
        pred_total = pred_future + residual_preds
    else:
        pred_total = pred_future

    future['예측입국자수'] = pred_total

    print(f"\n📌 예측 대상: {국적} / {목적} → XGBoost + GRU 보정 방식 사용" if model_gru else f"\n📌 예측 대상: {국적} / {목적} → XGBoost 단독 예측")
    for i, row in future.iterrows():
        print(f"\n🔮 {row['년월'].strftime('%Y년 %m월')} 예측 입국자 수: {int(round(row['예측입국자수'], 0))}명")

    # 평가 지표
    전체_예측값 = xgb.predict(X_scaled)
    if model_gru:
        recent_residuals = residuals[-seq_len:].tolist()
        residual_preds_train = []
        for _ in range(len(X_scaled)):
            seq_input = np.array(recent_residuals[-seq_len:]).reshape(1, seq_len, 1)
            pred_resid = model_gru.predict(seq_input, verbose=0)[0][0]
            residual_preds_train.append(pred_resid)
            recent_residuals.append(pred_resid)
        전체_예측값 += residual_preds_train

    r2 = r2_score(y, 전체_예측값)
    mape = mean_absolute_percentage_error(y, 전체_예측값)
    신뢰도 = max(0, 100 - mape * 100)

    print(f"\n✅ 예측 신뢰도 (최근값 기준): {신뢰도:.1f}%")
    print(f"📊 R²: {r2:.4f} | MAPE: {mape * 100:.2f}%")

    # 예측값 시각화 (실제 + 미래)
    시각화_df = pd.concat([data[['년월', '입국자수']], future[['년월', '예측입국자수']].rename(columns={'예측입국자수': '입국자수'})])
    시각화_df['구분'] = ['실제'] * len(data) + ['예측'] * len(future)

    plt.figure(figsize=(10, 4))
    for label, d in 시각화_df.groupby('구분'):
        plt.plot(d['년월'], d['입국자수'], marker='o', label=label)
    plt.title(f"{국적} - {목적} {예측연도}년 예측 결과")
    plt.xlabel("년월")
    plt.ylabel("입국자 수")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%



