import numpy as np
import matplotlib.pyplot as plt
import os

# 샘플 데이터 (실제값, LSTM/XGB 마지막 값, Prophet 예측값)
actual = [250000, 270000, 300000, 350000, 370000]  # 2025-01~2025-05
lstm_last = 370000  # 2025-05
prophet_pred = [120000, 130000, 170000, 160000, 150000, 140000]  # 2025-06~2025-11

# Prophet 패턴을 그대로 따르는 blending: LSTM/XGB 마지막 값을 anchor로 Prophet 증감률을 적용
blended = [lstm_last]
for i in range(1, len(prophet_pred)):
    prev_prophet = prophet_pred[i - 1]
    curr_prophet = prophet_pred[i]
    # Prophet의 증감률(전월 대비 비율) 적용
    ratio = curr_prophet / prev_prophet if prev_prophet != 0 else 1
    blended.append(blended[-1] * ratio)

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), actual, label="실제값(2025-01~05)", color="blue", marker="o")
plt.plot([5], [lstm_last], "bo")
plt.plot(
    range(6, 12),
    prophet_pred,
    label="Prophet 예측(원본)",
    color="green",
    marker="o",
    linestyle="--",
)
plt.plot(
    range(6, 12), blended, label="Blending 예측(Prophet 패턴)", color="red", marker="o"
)
plt.axvline(5.5, color="gray", linestyle=":", alpha=0.5)
plt.xticks(
    range(1, 12),
    ["2025-01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
)
plt.legend()
plt.title("Prophet 패턴을 그대로 따르는 Blending 예측 그래프 예시")
plt.tight_layout()

# 저장 경로
save_path = os.path.join(os.path.dirname(__file__), "blending_sample.png")
plt.savefig(save_path)
plt.close()

print(f"이미지 저장 완료: {save_path}")
