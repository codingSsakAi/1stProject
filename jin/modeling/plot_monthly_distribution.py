import pandas as pd
import matplotlib.pyplot as plt
import os
import platform

# Mac에서 한글 폰트 설정
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = "./results/국적_목적별_월별입국자수.csv"  # 예시로 첫 번째 파일만 사용
SAVE_DIR = "./results/visualize/"
os.makedirs(SAVE_DIR, exist_ok=True)


# 데이터 불러오기
df = pd.read_csv(CSV_PATH)

# 날짜 컬럼이 문자열이면 datetime으로 변환
df["날짜"] = pd.to_datetime(df["날짜"])

# 시각화 대상 목적 리스트
target_purposes = ["관광", "상용", "공용", "유학연수"]

# 국적별로 반복
for nationality in df["국적"].unique():
    plt.figure(figsize=(16, 6))
    for purpose in target_purposes:
        sub = df[(df["국적"] == nationality) & (df["목적"] == purpose)]
        if sub.empty:
            continue
        # 월별 입국자수 라인 그래프
        plt.plot(sub["날짜"], sub["입국자수"], marker="o", label=purpose)
    plt.title(f"{nationality} - 목적별 월별 입국자수 분포", fontsize=16)
    plt.xlabel("날짜", fontsize=13)
    plt.ylabel("입국자수", fontsize=13)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    # 파일명 안전하게
    safe_name = f"{nationality}_목적별_월별입국자수.png".replace("/", "_")
    plt.savefig(os.path.join(SAVE_DIR, safe_name))
    plt.close()
    print(f"저장 완료: {os.path.join(SAVE_DIR, safe_name)}")

# 한글 주석: 목적별(관광/상용/공용/유학연수) 월별 입국자수 분포를 국적별로 한 번에 시각화합니다.
