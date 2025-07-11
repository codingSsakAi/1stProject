# -*- coding: utf-8 -*-
"""
데이터 탐색 및 시각화 클래스
Author: Jin
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import os
from typing import List, Dict, Any

# 프로젝트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.config import *
from scripts.utils import *
from scripts.data_loader import DataLoader

# 한글 폰트 설정
setup_plotting()

logger = logging.getLogger(__name__)


class DataExplorer:
    """데이터 탐색 및 시각화 클래스"""

    def __init__(self, data_loader: DataLoader = None):
        """
        초기화

        Args:
            data_loader: 데이터 로더 인스턴스
        """
        self.data_loader = data_loader or DataLoader()
        self.data = None

        logger.info("📊 데이터 탐색기 초기화")

    def load_and_explore(self):
        """데이터 로드 및 기본 탐색"""
        self.data = self.data_loader.preprocess_data()

        # 기본 정보 출력
        self.data_loader.print_data_info()

        # 기본 탐색 실행
        self.explore_target_distribution()
        self.explore_time_trends()
        self.explore_country_patterns()
        self.explore_purpose_patterns()
        self.explore_covid_impact()

        logger.info("✅ 데이터 탐색 완료")

    def explore_target_distribution(self):
        """타겟 변수 분포 탐색"""
        logger.info("🔍 타겟 변수 분포 분석 중...")

        # 한글 폰트 강제 적용
        ensure_korean_font()

        plt.figure(figsize=(15, 10))

        # 전체 분포
        plt.subplot(2, 3, 1)
        self.data[TARGET_COLUMN].hist(bins=50, alpha=0.7)
        plt.title("입국자수 분포")
        plt.xlabel("입국자수")
        plt.ylabel("빈도")

        # 로그 변환 분포
        plt.subplot(2, 3, 2)
        log_values = np.log1p(self.data[TARGET_COLUMN])
        log_values.hist(bins=50, alpha=0.7)
        plt.title("입국자수 분포 (로그변환)")
        plt.xlabel("log(입국자수+1)")
        plt.ylabel("빈도")

        # 박스플롯
        plt.subplot(2, 3, 3)
        plt.boxplot(self.data[TARGET_COLUMN])
        plt.title("입국자수 박스플롯")
        plt.ylabel("입국자수")

        # 연도별 분포
        plt.subplot(2, 3, 4)
        yearly_data = self.data.groupby("연도")[TARGET_COLUMN].sum()
        yearly_data.plot(kind="bar")
        plt.title("연도별 총 입국자수")
        plt.xlabel("연도")
        plt.ylabel("총 입국자수")
        plt.xticks(rotation=45)

        # 월별 분포
        plt.subplot(2, 3, 5)
        monthly_data = self.data.groupby("월")[TARGET_COLUMN].mean()
        monthly_data.plot(kind="bar")
        plt.title("월별 평균 입국자수")
        plt.xlabel("월")
        plt.ylabel("평균 입국자수")

        # 코로나 기간 비교
        plt.subplot(2, 3, 6)
        covid_comparison = self.data.groupby("코로나기간")[TARGET_COLUMN].mean()
        covid_comparison.plot(kind="bar")
        plt.title("코로나 기간별 평균 입국자수")
        plt.xlabel("코로나기간 (0:이전, 1:기간중)")
        plt.ylabel("평균 입국자수")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "target_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("✅ 타겟 변수 분포 분석 완료")

    def explore_time_trends(self):
        """시간별 트렌드 분석"""
        logger.info("🔍 시간별 트렌드 분석 중...")

        # 한글 폰트 강제 적용
        ensure_korean_font()

        # 월별 시계열 데이터 생성
        monthly_data = (
            self.data.groupby(["연도", "월"])
            .agg({TARGET_COLUMN: "sum", "코로나기간": "first"})
            .reset_index()
        )

        monthly_data["날짜"] = pd.to_datetime(
            monthly_data["연도"].astype(str)
            + "-"
            + monthly_data["월"].astype(str).str.zfill(2)
            + "-01"
        )

        plt.figure(figsize=(15, 10))

        # 전체 시계열
        plt.subplot(2, 2, 1)
        plt.plot(monthly_data["날짜"], monthly_data[TARGET_COLUMN])
        plt.title("전체 시계열 트렌드")
        plt.xlabel("날짜")
        plt.ylabel("월별 총 입국자수")
        plt.xticks(rotation=45)

        # 코로나 기간 구분
        plt.subplot(2, 2, 2)
        covid_data = monthly_data[monthly_data["코로나기간"] == 1]
        normal_data = monthly_data[monthly_data["코로나기간"] == 0]

        plt.plot(normal_data["날짜"], normal_data[TARGET_COLUMN], label="정상기간", alpha=0.7)
        plt.plot(
            covid_data["날짜"],
            covid_data[TARGET_COLUMN],
            label="코로나기간",
            alpha=0.7,
            color="red",
        )
        plt.title("코로나 기간별 구분")
        plt.xlabel("날짜")
        plt.ylabel("월별 총 입국자수")
        plt.legend()
        plt.xticks(rotation=45)

        # 계절성 패턴
        plt.subplot(2, 2, 3)
        seasonal_data = self.data.groupby("월")[TARGET_COLUMN].mean()
        seasonal_data.plot(kind="bar")
        plt.title("월별 계절성 패턴")
        plt.xlabel("월")
        plt.ylabel("평균 입국자수")

        # 연도별 트렌드
        plt.subplot(2, 2, 4)
        yearly_trend = self.data.groupby("연도")[TARGET_COLUMN].sum()
        yearly_trend.plot(kind="line", marker="o")
        plt.title("연도별 총 입국자수 트렌드")
        plt.xlabel("연도")
        plt.ylabel("총 입국자수")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "time_trends.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("✅ 시간별 트렌드 분석 완료")

    def explore_country_patterns(self):
        """국가별 패턴 분석"""
        logger.info("🔍 국가별 패턴 분석 중...")

        # 상위 10개 국가
        top_countries = self.data.groupby("국적")[TARGET_COLUMN].sum().nlargest(10)

        plt.figure(figsize=(15, 10))

        # 상위 국가 막대그래프
        plt.subplot(2, 2, 1)
        top_countries.plot(kind="bar")
        plt.title("상위 10개 국가별 총 입국자수")
        plt.xlabel("국적")
        plt.ylabel("총 입국자수")
        plt.xticks(rotation=45)

        # 상위 국가 시계열
        plt.subplot(2, 2, 2)
        top_5_countries = top_countries.head(5).index

        for country in top_5_countries:
            country_data = self.data[self.data["국적"] == country]
            monthly_data = country_data.groupby(["연도", "월"])[TARGET_COLUMN].sum().reset_index()
            monthly_data["날짜"] = pd.to_datetime(
                monthly_data["연도"].astype(str)
                + "-"
                + monthly_data["월"].astype(str).str.zfill(2)
                + "-01"
            )
            plt.plot(
                monthly_data["날짜"],
                monthly_data[TARGET_COLUMN],
                label=f"국적_{country}",
                alpha=0.7,
            )

        plt.title("상위 5개 국가 시계열")
        plt.xlabel("날짜")
        plt.ylabel("월별 입국자수")
        plt.legend()
        plt.xticks(rotation=45)

        # 국가별 코로나 영향
        plt.subplot(2, 2, 3)
        covid_impact = []
        for country in top_5_countries:
            country_data = self.data[self.data["국적"] == country]
            pre_covid = country_data[country_data["코로나기간"] == 0][TARGET_COLUMN].mean()
            during_covid = country_data[country_data["코로나기간"] == 1][TARGET_COLUMN].mean()
            impact = (during_covid - pre_covid) / pre_covid * 100 if pre_covid > 0 else 0
            covid_impact.append(impact)

        plt.bar(range(len(top_5_countries)), covid_impact)
        plt.title("상위 5개 국가 코로나 영향 (%)")
        plt.xlabel("국적")
        plt.ylabel("변화율 (%)")
        plt.xticks(range(len(top_5_countries)), [f"국적_{c}" for c in top_5_countries], rotation=45)

        # 국가별 계절성
        plt.subplot(2, 2, 4)
        seasonal_data = self.data[self.data["국적"].isin(top_5_countries)]
        seasonal_pivot = seasonal_data.pivot_table(
            values=TARGET_COLUMN, index="월", columns="국적", aggfunc="mean"
        )

        for country in top_5_countries:
            if country in seasonal_pivot.columns:
                plt.plot(
                    seasonal_pivot.index,
                    seasonal_pivot[country],
                    label=f"국적_{country}",
                    marker="o",
                )

        plt.title("상위 5개 국가 계절성 패턴")
        plt.xlabel("월")
        plt.ylabel("평균 입국자수")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "country_patterns.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("✅ 국가별 패턴 분석 완료")

    def explore_purpose_patterns(self):
        """목적별 패턴 분석"""
        logger.info("🔍 목적별 패턴 분석 중...")

        # 목적별 총 입국자수
        purpose_data = self.data.groupby("목적")[TARGET_COLUMN].sum().sort_values(ascending=False)

        plt.figure(figsize=(15, 8))

        # 목적별 막대그래프
        plt.subplot(2, 2, 1)
        purpose_data.plot(kind="bar")
        plt.title("목적별 총 입국자수")
        plt.xlabel("목적")
        plt.ylabel("총 입국자수")
        plt.xticks(rotation=45)

        # 목적별 시계열
        plt.subplot(2, 2, 2)
        top_purposes = purpose_data.head(5).index

        for purpose in top_purposes:
            purpose_data_ts = self.data[self.data["목적"] == purpose]
            monthly_data = (
                purpose_data_ts.groupby(["연도", "월"])[TARGET_COLUMN].sum().reset_index()
            )
            monthly_data["날짜"] = pd.to_datetime(
                monthly_data["연도"].astype(str)
                + "-"
                + monthly_data["월"].astype(str).str.zfill(2)
                + "-01"
            )
            plt.plot(
                monthly_data["날짜"],
                monthly_data[TARGET_COLUMN],
                label=f"목적_{purpose}",
                alpha=0.7,
            )

        plt.title("상위 5개 목적별 시계열")
        plt.xlabel("날짜")
        plt.ylabel("월별 입국자수")
        plt.legend()
        plt.xticks(rotation=45)

        # 목적별 코로나 영향
        plt.subplot(2, 2, 3)
        covid_impact = []
        for purpose in top_purposes:
            purpose_data_covid = self.data[self.data["목적"] == purpose]
            pre_covid = purpose_data_covid[purpose_data_covid["코로나기간"] == 0][
                TARGET_COLUMN
            ].mean()
            during_covid = purpose_data_covid[purpose_data_covid["코로나기간"] == 1][
                TARGET_COLUMN
            ].mean()
            impact = (during_covid - pre_covid) / pre_covid * 100 if pre_covid > 0 else 0
            covid_impact.append(impact)

        plt.bar(range(len(top_purposes)), covid_impact)
        plt.title("상위 5개 목적별 코로나 영향 (%)")
        plt.xlabel("목적")
        plt.ylabel("변화율 (%)")
        plt.xticks(range(len(top_purposes)), [f"목적_{p}" for p in top_purposes], rotation=45)

        # 목적별 계절성
        plt.subplot(2, 2, 4)
        seasonal_data = self.data[self.data["목적"].isin(top_purposes)]
        seasonal_pivot = seasonal_data.pivot_table(
            values=TARGET_COLUMN, index="월", columns="목적", aggfunc="mean"
        )

        for purpose in top_purposes:
            if purpose in seasonal_pivot.columns:
                plt.plot(
                    seasonal_pivot.index,
                    seasonal_pivot[purpose],
                    label=f"목적_{purpose}",
                    marker="o",
                )

        plt.title("상위 5개 목적별 계절성 패턴")
        plt.xlabel("월")
        plt.ylabel("평균 입국자수")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "purpose_patterns.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("✅ 목적별 패턴 분석 완료")

    def explore_covid_impact(self):
        """코로나 영향 분석"""
        logger.info("🔍 코로나 영향 분석 중...")

        plt.figure(figsize=(15, 8))

        # 코로나 전후 비교
        plt.subplot(2, 2, 1)
        covid_comparison = self.data.groupby("코로나기간")[TARGET_COLUMN].agg(["mean", "sum"])
        covid_comparison.plot(kind="bar")
        plt.title("코로나 기간별 입국자수 비교")
        plt.xlabel("코로나기간 (0:이전, 1:기간중)")
        plt.ylabel("입국자수")
        plt.xticks(rotation=0)

        # 연도별 코로나 영향
        plt.subplot(2, 2, 2)
        yearly_covid = self.data.groupby(["연도", "코로나기간"])[TARGET_COLUMN].sum().unstack()
        yearly_covid.plot(kind="bar", stacked=True)
        plt.title("연도별 코로나 기간 구분")
        plt.xlabel("연도")
        plt.ylabel("총 입국자수")
        plt.legend(["정상기간", "코로나기간"])
        plt.xticks(rotation=45)

        # 회복 추세 분석
        plt.subplot(2, 2, 3)
        recovery_data = self.data[self.data["연도"] >= 2020]
        monthly_recovery = recovery_data.groupby(["연도", "월"])[TARGET_COLUMN].sum().reset_index()
        monthly_recovery["날짜"] = pd.to_datetime(
            monthly_recovery["연도"].astype(str)
            + "-"
            + monthly_recovery["월"].astype(str).str.zfill(2)
            + "-01"
        )

        plt.plot(monthly_recovery["날짜"], monthly_recovery[TARGET_COLUMN], marker="o")
        plt.title("2020년 이후 회복 추세")
        plt.xlabel("날짜")
        plt.ylabel("월별 총 입국자수")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 코로나 이전 대비 회복률
        plt.subplot(2, 2, 4)
        pre_covid_avg = self.data[self.data["코로나기간"] == 0][TARGET_COLUMN].mean()
        recent_data = self.data[self.data["연도"] >= 2023]
        recovery_rate = recent_data[TARGET_COLUMN].mean() / pre_covid_avg * 100

        plt.bar(["코로나 이전", "최근(2023+)"], [100, recovery_rate])
        plt.title(f"회복률: {recovery_rate:.1f}%")
        plt.ylabel("회복률 (%)")
        plt.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="코로나 이전 수준")
        plt.legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "covid_impact.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("✅ 코로나 영향 분석 완료")

    def generate_summary_report(self):
        """요약 보고서 생성"""
        logger.info("📋 요약 보고서 생성 중...")

        summary = self.data_loader.get_data_summary()

        report = f"""
# 외국인 입국자 데이터 탐색 보고서

## 📊 기본 통계
- 총 데이터 수: {summary['total_rows']:,}행
- 총 컬럼 수: {summary['total_columns']}개
- 기간: {summary['date_range']}
- 국적 수: {summary['unique_countries']}개
- 목적 수: {summary['unique_purposes']}개

## 🎯 타겟 변수 통계
- 평균: {summary['target_stats']['mean']:.2f}
- 표준편차: {summary['target_stats']['std']:.2f}
- 최소값: {summary['target_stats']['min']:.2f}
- 최대값: {summary['target_stats']['max']:.2f}
- 중간값: {summary['target_stats']['median']:.2f}

## 🦠 코로나 영향
- 코로나 기간 데이터 비율: {summary['covid_period_ratio']:.2%}

## 📈 주요 발견사항
1. 시계열 트렌드: 코로나 기간 중 급격한 감소 후 회복 추세
2. 계절성 패턴: 특정 월에 입국자 수 집중
3. 국가별 차이: 상위 몇 개 국가가 전체 입국자의 대부분 차지
4. 목적별 차이: 목적에 따라 코로나 영향 정도 상이

## 🔮 모델링 시사점
- 시계열 특성과 계절성 고려 필요
- 코로나 기간을 별도 변수로 활용
- 국가별, 목적별 개별 모델 vs 통합 모델 비교 필요
- 지연 변수와 이동평균 활용 효과적

---
생성일: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # 보고서 저장
        with open(RESULTS_DIR / "exploration_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("✅ 요약 보고서 생성 완료")
        print(report)


# 실행 예시
if __name__ == "__main__":
    # 데이터 탐색기 생성
    explorer = DataExplorer()

    # 전체 탐색 실행
    explorer.load_and_explore()

    # 요약 보고서 생성
    explorer.generate_summary_report()

    logger.info("✅ 데이터 탐색 완료")
