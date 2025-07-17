import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform

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

# 1. 결과 파일에서 12/24개월 stacking 예측만 불러오기
# 각 파일의 test_df에 stacking_pred, ds, 입국자수 칼럼이 반드시 있어야 함
test12 = pd.read_csv("data/12개월_test_df.csv")  # 파일명, 경로 실제에 맞게
test24 = pd.read_csv("data/24개월_test_df.csv")

# 2. 날짜(ds) 기준으로 merge
df = pd.merge(
    test12[['ds', '입국자수', 'stacking_pred']],
    test24[['ds', 'stacking_pred']],
    on='ds',
    suffixes=('_12', '_24')
)

# 3. 결합 예측값 생성
df['blend_mean'] = (df['stacking_pred_12'] + df['stacking_pred_24']) / 2

# 4. Meta-ensemble (LinearRegression)
meta_X = df[['stacking_pred_12', 'stacking_pred_24']]
meta_y = df['입국자수']
meta_model = LinearRegression()
meta_model.fit(meta_X, meta_y)
df['meta_pred'] = meta_model.predict(meta_X)

# 5. 평가 함수
def eval_metric(y_true, y_pred):
    return {
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }

metrics = {}
for col in ['stacking_pred_12', 'stacking_pred_24', 'blend_mean', 'meta_pred']:
    metrics[col] = eval_metric(df['입국자수'], df[col])

# 6. 시각화
plt.figure(figsize=(18,8))
plt.plot(df['ds'], df['입국자수'], label='실제', color='black', linewidth=2)
plt.plot(df['ds'], df['stacking_pred_12'], label='12개월 Stacking', linestyle='-')
plt.plot(df['ds'], df['stacking_pred_24'], label='24개월 Stacking', linestyle='-')
plt.plot(df['ds'], df['blend_mean'], label='단순평균(Blending)', linestyle='--', linewidth=2)
plt.plot(df['ds'], df['meta_pred'], label='Meta-Ensemble', linestyle='-.', linewidth=2)
plt.legend(fontsize=13)
plt.title("12개월/24개월 예측 & 앙상블 비교", fontsize=20)
plt.xlabel("연월")
plt.ylabel("입국자수")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. 평가 결과 출력
print("\n[모델별 성능 비교]")
for key, v in metrics.items():
    print(f"{key}: RMSE={v['RMSE']:.2f}, MAPE={v['MAPE']:.4f}")

