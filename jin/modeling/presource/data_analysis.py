# -*- coding: utf-8 -*-
"""
국적별, 목적별 입국자 예측을 위한 데이터 분석 (GPU 전용)
Author: Jin
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from datetime import datetime


# GPU 설정
def setup_gpu():
    """M1 GPU 최적화 설정"""
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("🍎 M1 GPU 최적화 완료")
            return True
        else:
            print("⚠️ GPU 미발견 - CPU 모드로 실행")
            return False
    except Exception as e:
        print(f"⚠️ GPU 설정 실패: {e}")
        return False


# GPU 설정 실행
gpu_available = setup_gpu()

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 결과 저장 폴더
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def analyze_data():
    """데이터 분석 메인 함수"""
    print("🔍 국적별, 목적별 입국자 데이터 분석 시작 (GPU 전용)")
    print("=" * 60)

    # 현재 시간 (파일명 구분용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 데이터 로드
    df = pd.read_csv("../../data_preprocessing/data/processed/외국인입국자_전처리완료_딥러닝용.csv")
    print(f"✅ 전체 데이터: {df.shape}")
    print(f"📋 컬럼: {list(df.columns)}")

    # 2. 기본 정보 확인
    if "국적" in df.columns and "목적" in df.columns:
        print(f"\n📊 기본 정보:")
        print(f"- 국적 종류: {df['국적'].nunique()}개")
        print(f"- 목적 종류: {df['목적'].nunique()}개")
        print(f"- 전체 조합: {len(df.groupby(['국적', '목적']))}")

        # 3. 상위 국적 확인 (입국자수 기준)
        print(f"\n🌍 상위 국적 (총 입국자수 기준):")
        top_nationalities = (
            df.groupby("국적")["입국자수"].sum().sort_values(ascending=False).head(15)
        )
        nationality_summary = []

        for i, (nationality, total) in enumerate(top_nationalities.items(), 1):
            print(f"  {i:2d}. {nationality}: {total:,}명")
            nationality_summary.append(
                {"rank": i, "nationality": nationality, "total_visitors": total}
            )

        # 4. 목적별 확인
        print(f"\n🎯 목적별 입국자 분포:")
        purpose_totals = df.groupby("목적")["입국자수"].sum().sort_values(ascending=False)
        purpose_summary = []

        for i, (purpose, total) in enumerate(purpose_totals.items(), 1):
            percentage = (total / df["입국자수"].sum()) * 100
            print(f"  {i}. {purpose}: {total:,}명 ({percentage:.1f}%)")
            purpose_summary.append(
                {"rank": i, "purpose": purpose, "total_visitors": total, "percentage": percentage}
            )

        # 5. 상위 조합 확인 (데이터 량 기준)
        print(f"\n📈 상위 국적-목적 조합 (데이터 량 기준):")
        combination_counts = (
            df.groupby(["국적", "목적"]).size().sort_values(ascending=False).head(20)
        )
        combination_summary = []

        for i, ((nationality, purpose), count) in enumerate(combination_counts.items(), 1):
            total_visitors = df[(df["국적"] == nationality) & (df["목적"] == purpose)][
                "입국자수"
            ].sum()
            avg_monthly = total_visitors / count if count > 0 else 0

            print(f"  {i:2d}. {nationality} - {purpose}")
            print(
                f"      📊 {count}개월 데이터, 총 {total_visitors:,}명, 월평균 {avg_monthly:.0f}명"
            )

            combination_summary.append(
                {
                    "rank": i,
                    "nationality": nationality,
                    "purpose": purpose,
                    "data_points": count,
                    "total_visitors": total_visitors,
                    "avg_monthly": avg_monthly,
                }
            )

        # 6. 예측 모델 구축에 적합한 조합 분석
        print(f"\n🎯 예측 모델 구축 적합성 분석:")
        print("-" * 80)

        predictable_combinations = []

        for (nationality, purpose), count in combination_counts.head(15).items():
            subset = df[(df["국적"] == nationality) & (df["목적"] == purpose)].copy()

            if len(subset) >= 36:  # 최소 3년 데이터 (예측 모델용)
                # 시계열 순서로 정렬
                if "시계열순서" in subset.columns:
                    subset = subset.sort_values("시계열순서")
                elif "연도" in subset.columns and "월" in subset.columns:
                    subset = subset.sort_values(["연도", "월"])

                visitors = subset["입국자수"].values

                # 기본 통계
                mean_visitors = np.mean(visitors)
                std_visitors = np.std(visitors)
                cv = std_visitors / mean_visitors if mean_visitors > 0 else float("inf")

                # 예측 가능성 지표들
                # 1. 자기상관 (시계열 패턴)
                if len(visitors) > 12:
                    autocorr_1 = (
                        np.corrcoef(visitors[:-1], visitors[1:])[0, 1] if len(visitors) > 1 else 0
                    )
                    autocorr_3 = (
                        np.corrcoef(visitors[:-3], visitors[3:])[0, 1] if len(visitors) > 3 else 0
                    )
                    autocorr_12 = (
                        np.corrcoef(visitors[:-12], visitors[12:])[0, 1]
                        if len(visitors) > 12
                        else 0
                    )
                else:
                    autocorr_1 = autocorr_3 = autocorr_12 = 0

                # 2. 트렌드 분석
                if len(visitors) > 24:
                    x = np.arange(len(visitors))
                    trend_slope = np.polyfit(x, visitors, 1)[0]
                    trend_r2 = np.corrcoef(x, visitors)[0, 1] ** 2 if len(visitors) > 1 else 0
                else:
                    trend_slope = trend_r2 = 0

                # 3. 코로나 영향 분석
                covid_impact = covid_recovery = 0
                if "연도" in subset.columns:
                    pre_covid = subset[subset["연도"] <= 2019]["입국자수"]
                    covid_period = subset[(subset["연도"] >= 2020) & (subset["연도"] <= 2021)][
                        "입국자수"
                    ]
                    post_covid = subset[subset["연도"] >= 2022]["입국자수"]

                    if len(pre_covid) > 0 and len(covid_period) > 0:
                        covid_impact = (
                            (covid_period.mean() - pre_covid.mean()) / pre_covid.mean() * 100
                        )

                    if len(covid_period) > 0 and len(post_covid) > 0:
                        covid_recovery = (
                            (post_covid.mean() - covid_period.mean()) / covid_period.mean() * 100
                        )

                # 4. 예측 적합성 점수 계산
                suitability_score = 0

                # 데이터 양 (3점)
                if len(subset) >= 60:
                    suitability_score += 3
                elif len(subset) >= 48:
                    suitability_score += 2
                elif len(subset) >= 36:
                    suitability_score += 1

                # 안정성 (2점)
                if cv < 1.0:
                    suitability_score += 2
                elif cv < 2.0:
                    suitability_score += 1

                # 예측가능성 (3점)
                if autocorr_1 > 0.5:
                    suitability_score += 2
                elif autocorr_1 > 0.3:
                    suitability_score += 1
                if autocorr_12 > 0.3:
                    suitability_score += 1

                # 데이터 크기 (1점)
                if mean_visitors > 500:
                    suitability_score += 1

                # 트렌드 명확성 (1점)
                if trend_r2 > 0.3:
                    suitability_score += 1

                predictable_combinations.append(
                    {
                        "nationality": nationality,
                        "purpose": purpose,
                        "data_points": len(subset),
                        "mean_visitors": mean_visitors,
                        "std_visitors": std_visitors,
                        "cv": cv,
                        "autocorr_1m": autocorr_1 if not np.isnan(autocorr_1) else 0,
                        "autocorr_3m": autocorr_3 if not np.isnan(autocorr_3) else 0,
                        "autocorr_12m": autocorr_12 if not np.isnan(autocorr_12) else 0,
                        "trend_slope": trend_slope,
                        "trend_r2": trend_r2 if not np.isnan(trend_r2) else 0,
                        "covid_impact": covid_impact if not np.isnan(covid_impact) else 0,
                        "covid_recovery": covid_recovery if not np.isnan(covid_recovery) else 0,
                        "suitability_score": suitability_score,
                        "pre_covid_avg": pre_covid.mean() if len(pre_covid) > 0 else 0,
                        "covid_avg": covid_period.mean() if len(covid_period) > 0 else 0,
                        "post_covid_avg": post_covid.mean() if len(post_covid) > 0 else 0,
                    }
                )

                print(f"📊 {nationality} - {purpose}:")
                print(f"   📈 데이터: {len(subset)}개월, 월평균: {mean_visitors:.0f}명")
                print(
                    f"   📉 변동성: {cv:.2f} ({'안정적' if cv < 1.5 else '변동적' if cv < 3.0 else '매우변동적'})"
                )
                print(
                    f"   🔄 자기상관: 1개월({autocorr_1:.3f}), 3개월({autocorr_3:.3f}), 12개월({autocorr_12:.3f})"
                )
                print(f"   📈 트렌드: 기울기 {trend_slope:.1f}, R² {trend_r2:.3f}")
                print(f"   🦠 코로나: 영향 {covid_impact:+.0f}%, 회복 {covid_recovery:+.0f}%")
                print(f"   ⭐ 예측 적합도: {suitability_score}/10점")
                print()

        # 7. 추천 조합 (예측 모델 구축용)
        print(f"\n🏆 예측 모델 구축 추천 조합 (적합도 순):")

        if predictable_combinations:
            predictable_df = pd.DataFrame(predictable_combinations)
            top_predictable = predictable_df.sort_values("suitability_score", ascending=False).head(
                10
            )

            print("\n🎯 TOP 10 추천 조합:")
            for idx, (_, row) in enumerate(top_predictable.iterrows(), 1):
                print(
                    f"  {idx:2d}. [{row['suitability_score']:.0f}/10점] {row['nationality']} - {row['purpose']}"
                )
                print(
                    f"      📊 {row['data_points']}개월, 평균 {row['mean_visitors']:.0f}명, CV {row['cv']:.2f}"
                )
                print(
                    f"      🔄 자기상관 {row['autocorr_1m']:.3f}, 트렌드 R² {row['trend_r2']:.3f}"
                )
        else:
            top_predictable = pd.DataFrame()
            print("⚠️ 예측 모델 구축에 적합한 조합을 찾지 못했습니다.")

        # 8. 결과 저장
        save_analysis_results(
            nationality_summary,
            purpose_summary,
            combination_summary,
            predictable_combinations,
            timestamp,
        )

        # 9. 시각화 생성
        create_visualizations(df, top_predictable, timestamp)

        return top_predictable if len(predictable_combinations) > 0 else pd.DataFrame()

    else:
        print("❌ '국적' 또는 '목적' 컬럼이 없습니다!")
        return pd.DataFrame()


def save_analysis_results(
    nationality_summary, purpose_summary, combination_summary, predictable_combinations, timestamp
):
    """분석 결과를 results 폴더에 저장"""
    print(f"\n💾 분석 결과 저장 중...")

    # 1. 국적별 분석 결과
    nationality_df = pd.DataFrame(nationality_summary)
    nationality_path = os.path.join(RESULTS_DIR, f"nationality_analysis_{timestamp}.csv")
    nationality_df.to_csv(nationality_path, index=False, encoding="utf-8-sig")

    # 2. 목적별 분석 결과
    purpose_df = pd.DataFrame(purpose_summary)
    purpose_path = os.path.join(RESULTS_DIR, f"purpose_analysis_{timestamp}.csv")
    purpose_df.to_csv(purpose_path, index=False, encoding="utf-8-sig")

    # 3. 조합별 분석 결과
    combination_df = pd.DataFrame(combination_summary)
    combination_path = os.path.join(RESULTS_DIR, f"combination_analysis_{timestamp}.csv")
    combination_df.to_csv(combination_path, index=False, encoding="utf-8-sig")

    # 4. 예측 적합성 분석 결과
    if predictable_combinations:
        predictable_df = pd.DataFrame(predictable_combinations)
        predictable_path = os.path.join(RESULTS_DIR, f"prediction_suitability_{timestamp}.csv")
        predictable_df.to_csv(predictable_path, index=False, encoding="utf-8-sig")

        print(f"✅ 분석 결과 저장 완료:")
        print(f"   📄 국적별 분석: {nationality_path}")
        print(f"   📄 목적별 분석: {purpose_path}")
        print(f"   📄 조합별 분석: {combination_path}")
        print(f"   📄 예측 적합성: {predictable_path}")
    else:
        print("⚠️ 예측 적합성 분석 결과 없음")


def create_visualizations(df, top_predictable, timestamp):
    """분석 결과 시각화 생성 및 저장"""
    print(f"\n📊 시각화 생성 중...")

    # 큰 화면으로 설정
    plt.figure(figsize=(20, 15))

    # 1. 상위 국적별 입국자 수 (막대 차트)
    plt.subplot(3, 3, 1)
    top_nationalities = df.groupby("국적")["입국자수"].sum().sort_values(ascending=False).head(10)
    plt.barh(range(len(top_nationalities)), top_nationalities.values, color="skyblue")
    plt.yticks(range(len(top_nationalities)), top_nationalities.index)
    plt.xlabel("총 입국자수 (명)")
    plt.title("상위 10개 국적별 총 입국자수")
    plt.gca().invert_yaxis()

    # 2. 목적별 입국자 비율 (파이 차트)
    plt.subplot(3, 3, 2)
    purpose_totals = df.groupby("목적")["입국자수"].sum().sort_values(ascending=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(purpose_totals)))
    plt.pie(purpose_totals.values, labels=purpose_totals.index, autopct="%1.1f%%", colors=colors)
    plt.title("목적별 입국자 비율")

    # 3. 연도별 총 입국자 트렌드
    plt.subplot(3, 3, 3)
    if "연도" in df.columns:
        yearly_trend = df.groupby("연도")["입국자수"].sum()
        plt.plot(yearly_trend.index, yearly_trend.values, marker="o", linewidth=2, markersize=6)
        plt.xlabel("연도")
        plt.ylabel("총 입국자수 (명)")
        plt.title("연도별 입국자 트렌드 (코로나 영향 포함)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 코로나 기간 표시
        plt.axvspan(2020, 2021, alpha=0.3, color="red", label="코로나 기간")
        plt.legend()

    # 4. 월별 입국자 패턴 (박스플롯)
    plt.subplot(3, 3, 4)
    if "월" in df.columns:
        monthly_data = [df[df["월"] == month]["입국자수"].values for month in range(1, 13)]
        plt.boxplot(monthly_data, labels=range(1, 13))
        plt.xlabel("월")
        plt.ylabel("입국자수 (명)")
        plt.title("월별 입국자수 분포 (박스플롯)")
        plt.grid(True, alpha=0.3)

    # 5. 예측 적합성 점수 분포
    plt.subplot(3, 3, 5)
    if len(top_predictable) > 0:
        scores = top_predictable["suitability_score"].head(10)
        names = [
            f"{row['nationality'][:5]}-{row['purpose'][:5]}"
            for _, row in top_predictable.head(10).iterrows()
        ]
        bars = plt.bar(range(len(scores)), scores, color="lightgreen")
        plt.xlabel("조합 순위")
        plt.ylabel("예측 적합성 점수")
        plt.title("상위 10개 조합의 예측 적합성")
        plt.xticks(range(len(names)), names, rotation=45, ha="right")

        # 점수 표시
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{score}",
                ha="center",
                va="bottom",
            )
    else:
        plt.text(
            0.5,
            0.5,
            "예측 적합한 조합 없음",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=12,
        )
        plt.title("예측 적합성 점수")

    # 6. 변동성 vs 평균 입국자수 산점도
    plt.subplot(3, 3, 6)
    if len(top_predictable) > 0:
        plt.scatter(
            top_predictable["mean_visitors"],
            top_predictable["cv"],
            s=top_predictable["suitability_score"] * 20,
            alpha=0.6,
            c="coral",
        )
        plt.xlabel("평균 월별 입국자수 (명)")
        plt.ylabel("변동계수 (CV)")
        plt.title("평균 입국자수 vs 변동성\n(원 크기 = 예측 적합성)")
        plt.grid(True, alpha=0.3)

        # 기준선 표시
        plt.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="변동성 기준 (CV=2.0)")
        plt.legend()

    # 7. 자기상관 히트맵
    plt.subplot(3, 3, 7)
    if len(top_predictable) >= 5:
        top_5 = top_predictable.head(5)
        autocorr_data = top_5[["autocorr_1m", "autocorr_3m", "autocorr_12m"]].values
        names = [f"{row['nationality'][:8]}-{row['purpose'][:8]}" for _, row in top_5.iterrows()]

        im = plt.imshow(autocorr_data, cmap="RdYlBu_r", aspect="auto")
        plt.colorbar(im)
        plt.yticks(range(len(names)), names)
        plt.xticks([0, 1, 2], ["1개월", "3개월", "12개월"])
        plt.title("상위 5개 조합의 자기상관 패턴")

        # 수치 표시
        for i in range(len(names)):
            for j in range(3):
                plt.text(
                    j,
                    i,
                    f"{autocorr_data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if autocorr_data[i, j] > 0.5 else "black",
                )

    # 8. 코로나 전후 비교
    plt.subplot(3, 3, 8)
    if len(top_predictable) >= 5:
        top_5 = top_predictable.head(5)
        pre_covid = top_5["pre_covid_avg"].values
        post_covid = top_5["post_covid_avg"].values
        names = [f"{row['nationality'][:6]}-{row['purpose'][:6]}" for _, row in top_5.iterrows()]

        x = np.arange(len(names))
        width = 0.35

        plt.bar(x - width / 2, pre_covid, width, label="코로나 이전", color="lightblue")
        plt.bar(x + width / 2, post_covid, width, label="코로나 이후", color="lightcoral")

        plt.xlabel("조합")
        plt.ylabel("월평균 입국자수 (명)")
        plt.title("코로나 전후 입국자수 비교 (상위 5개 조합)")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 9. 데이터 포인트 수 분포
    plt.subplot(3, 3, 9)
    if len(top_predictable) > 0:
        data_points = top_predictable["data_points"].head(10)
        names = [
            f"{row['nationality'][:5]}-{row['purpose'][:5]}"
            for _, row in top_predictable.head(10).iterrows()
        ]

        bars = plt.bar(range(len(data_points)), data_points, color="lightpink")
        plt.xlabel("조합 순위")
        plt.ylabel("데이터 포인트 수 (개월)")
        plt.title("상위 10개 조합의 데이터 양")
        plt.xticks(range(len(names)), names, rotation=45, ha="right")

        # 기준선 표시
        plt.axhline(y=36, color="red", linestyle="--", alpha=0.5, label="최소 기준 (36개월)")
        plt.axhline(y=60, color="green", linestyle="--", alpha=0.5, label="충분 기준 (60개월)")
        plt.legend()

    plt.tight_layout()

    # 저장
    viz_path = os.path.join(RESULTS_DIR, f"comprehensive_analysis_{timestamp}.png")
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 종합 시각화 저장: {viz_path}")


if __name__ == "__main__":
    print("🚀 국적별, 목적별 입국자 예측 분석 시작 (GPU 전용)")
    print(f"🍎 GPU 사용 가능: {'Yes' if gpu_available else 'No'}")
    print("=" * 60)

    # 분석 실행
    top_combinations = analyze_data()

    if len(top_combinations) > 0:
        print(f"\n🎉 분석 완료!")
        print(f"📊 예측 가능한 {len(top_combinations)}개 조합 발견")
        print(f"📁 결과 저장: ./results/ 폴더")
        print(f"🏆 최고 점수: {top_combinations.iloc[0]['suitability_score']}/10점")
        print(
            f"🥇 1위 조합: {top_combinations.iloc[0]['nationality']} - {top_combinations.iloc[0]['purpose']}"
        )
    else:
        print(f"\n⚠️ 예측 적합한 조합을 찾지 못했습니다.")

    print(f"\n👋 분석 완료 - 다음 단계: 예측 모델 구축")
