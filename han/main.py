import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import platform
import matplotlib as mpl

# 한글 폰트 자동설정
if platform.system() == "Windows":
    font_family = "Malgun Gothic"
elif platform.system() == "Darwin":
    font_family = "AppleGothic"
else:
    font_family = "NanumGothic"
mpl.rcParams['font.family'] = font_family
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = font_family
plt.rcParams['axes.unicode_minus'] = False

# 1. 예측결과 파일 로드
pred12 = pd.read_csv("12개월_국적목적별_향후12개월_예측.csv")
pred24 = pd.read_csv("24개월_국적목적별_향후24개월_예측.csv")
for df in [pred12, pred24]:
    df['국적'] = df['국적'].astype(str).str.strip()
    df['목적'] = df['목적'].astype(str).str.strip()
    df['ds'] = df['ds'].astype(str).str.strip()

# 2. 국가/목적 선택
print("\n=== 예측 가능한 국가(샘플) ===")
print(pred12['국적'].unique())
nation = input("\n국가명을 위에서 복사해 붙여넣으세요(정확하게): ").strip()
possible_purposes = pred12[pred12['국적']==nation]['목적'].unique()
print("\n=== 예측 가능한 목적(해당 국가) ===")
print(possible_purposes)
purpose = input("\n목적명을 위에서 복사해 붙여넣으세요(정확하게): ").strip()

# 3. 데이터 필터
pred12_ = pred12[(pred12['국적']==nation) & (pred12['목적']==purpose)].copy()
pred24_ = pred24[(pred24['국적']==nation) & (pred24['목적']==purpose)].copy()
if pred12_.empty or pred24_.empty:
    print("\n[오류] 해당 국가/목적 조합 데이터가 없습니다.")
    exit()

# 4. 날짜 기준 merge
df = pd.merge(
    pred12_[['ds', 'prophet_pred', 'xgb_pred', 'lstm_pred', 'stacking_pred']],
    pred24_[['ds', 'prophet_pred', 'xgb_pred', 'lstm_pred', 'stacking_pred']],
    on='ds', suffixes=('_12', '_24')
)

# 5. 실측값 추가 (있는 경우)
if '입국자수' in pred12_.columns:
    real = pred12_[['ds', '입국자수']]
    df = pd.merge(df, real, on='ds', how='left')

# 6. 평균/메타-앙상블
df['blend_mean'] = (df['stacking_pred_12'] + df['stacking_pred_24']) / 2

meta_pred_name = None
meta_fit = df.dropna(subset=['입국자수']) if '입국자수' in df.columns else pd.DataFrame()
if not meta_fit.empty:
    meta_X = meta_fit[['stacking_pred_12','stacking_pred_24']]
    meta_y = meta_fit['입국자수']
    meta_model = LinearRegression()
    meta_model.fit(meta_X, meta_y)
    df['meta_pred'] = meta_model.predict(df[['stacking_pred_12','stacking_pred_24']])
    meta_pred_name = 'meta_pred'
    print("\n[안내] Meta-Ensemble(LinearRegression) 적용됨")
else:
    print("\n[안내] 실측 구간이 없어 Meta-Ensemble은 '단순평균'으로 대체")
    df['meta_pred'] = df['blend_mean']
    meta_pred_name = 'blend_mean'

# 7. 날짜 질의 및 해당 예측값(실측 있으면 같이) 즉시 출력
while True:
    input_date = input("\n예측 날짜(YYYY-MM 형식, ex: 2026-03) 또는 Enter: 그래프 출력 : ").strip()
    if input_date == "":
        break
    row = df[df['ds'].astype(str).str.startswith(input_date)]
    if row.empty:
        print("해당 날짜 예측 데이터 없음.")
        continue
    result = row.iloc[0]
    print(f"\n[{nation}-{purpose}-{input_date}]")
    if '입국자수' in result and not pd.isnull(result['입국자수']):
        print(f"실제 입국자수: {result['입국자수']:.0f}")
    print(f"12개월 Stacking 예측: {result['stacking_pred_12']:.1f}")
    print(f"24개월 Stacking 예측: {result['stacking_pred_24']:.1f}")
    print(f"단순평균(Blending): {result['blend_mean']:.1f}")
    print(f"Meta-Ensemble: {result[meta_pred_name]:.1f}")
    # 오차 계산(실측 있는 경우)
    if '입국자수' in result and not pd.isnull(result['입국자수']):
        err12 = abs(result['입국자수'] - result['stacking_pred_12'])
        err24 = abs(result['입국자수'] - result['stacking_pred_24'])
        err_blend = abs(result['입국자수'] - result['blend_mean'])
        err_meta = abs(result['입국자수'] - result[meta_pred_name])
        print(f"[오차] 12M: {err12:.1f}, 24M: {err24:.1f}, Blending: {err_blend:.1f}, Meta: {err_meta:.1f}")

# 8. 전체 구간 시각화 (각 예측값 annotate)
plt.figure(figsize=(19,8))
x = df['ds']
if '입국자수' in df.columns and df['입국자수'].notna().any():
    plt.plot(x, df['입국자수'], label='실제', color='black', linewidth=2, marker='o', markersize=6)
plt.plot(x, df['stacking_pred_12'], label='12개월 Stacking', marker='s')
plt.plot(x, df['stacking_pred_24'], label='24개월 Stacking', marker='^')
plt.plot(x, df['blend_mean'], label='평균(Blending)', linestyle='--', linewidth=2, marker='P')
plt.plot(x, df['meta_pred'], label='Meta-Ensemble', linestyle='-.', linewidth=2, marker='X')

# 각 예측값 annotate (최근 5개월만 깔끔하게 출력)
for i in range(-5, 0):
    ds = df.iloc[i]['ds']
    plt.annotate(f"{df.iloc[i]['meta_pred']:.0f}", (ds, df.iloc[i]['meta_pred']),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='blue')
    if '입국자수' in df.columns and not pd.isnull(df.iloc[i]['입국자수']):
        plt.annotate(f"{df.iloc[i]['입국자수']:.0f}", (ds, df.iloc[i]['입국자수']),
                     textcoords="offset points", xytext=(0,-18), ha='center', fontsize=10, color='black')

plt.xticks(rotation=60)
plt.legend(fontsize=13)
plt.title(f"{nation}-{purpose} 입국자수 예측 (예측값 실시간 표기)", fontsize=20)
plt.xlabel("연월")
plt.ylabel("입국자수")
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. 모델 평가값 별도 출력 (실측구간 기준)
if '입국자수' in df.columns and df['입국자수'].notna().any():
    def eval_metric(y_true, y_pred):
        return {
            "RMSE": mean_squared_error(y_true, y_pred, squared=False),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred)
        }
    metrics = {}
    for col in ['stacking_pred_12','stacking_pred_24','blend_mean','meta_pred']:
        sub = df.dropna(subset=['입국자수',col])
        if not sub.empty:
            metrics[col] = eval_metric(sub['입국자수'], sub[col])
    print("\n[모델별 실측 성능 비교] (RMSE 낮을수록, MAPE 낮을수록 좋음)")
    for key, v in metrics.items():
        print(f"{key}: RMSE={v['RMSE']:.2f}, MAPE={v['MAPE']:.4f}")

    best_model = min(metrics, key=lambda k: metrics[k]['RMSE'])
    print(f"\n[Best] RMSE 기준 최적모델: {best_model} (RMSE={metrics[best_model]['RMSE']:.2f})")
else:
    print("\n[안내] 실측구간이 없어 성능평가를 생략합니다.")
