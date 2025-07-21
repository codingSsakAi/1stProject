# -*- coding: utf-8 -*-
"""
외국인 입국자 데이터 전처리 스크립트 (코로나 시기 포함)
Author: Jin
Created: 2025-01-15

이 스크립트는 다음과 같은 전처리 작업을 수행합니다:
1. 원본 데이터 로드 및 기본 클리닝
2. 소계/합계/교포/소개 등 불필요한 항목 제거 (딥러닝 모델에 불필요)
3. Long format으로 변환 (딥러닝 모델에 적합)
4. 시계열 특성 변수 생성
5. 최종 전처리 데이터 저장
"""

import pandas as pd
import numpy as np
import os
import re
import warnings

warnings.filterwarnings("ignore")


class ForeignVisitorDataProcessor:
    """외국인 입국자 데이터 전처리 클래스"""

    # 제거할 키워드 목록 (딥러닝 모델에 불필요한 집계성 데이터)
    KEYWORDS_TO_REMOVE = [
        "소 계",
        "소계",
        "합 계",
        "합계",
        "계",
        "교포",
        "소개",
        "아시아주",
        "미 주",
        "구 주",
        "아프리카주",
        "오세아니아주",
        "기타",
        "미주",
        "구주",
        "아시아",
        "아프리카",
        "오세아니아",
        "전체",
        "총계",
        "총 계",
        "전 체",
    ]

    # 계절 매핑
    SEASON_MAP = {
        12: "겨울",
        1: "겨울",
        2: "겨울",
        3: "봄",
        4: "봄",
        5: "봄",
        6: "여름",
        7: "여름",
        8: "여름",
        9: "가을",
        10: "가을",
        11: "가을",
    }

    # 코로나 시기 정의
    COVID_START = "2020-03-01"
    COVID_END = "2022-06-30"

    def __init__(
        self,
        input_file="/Volumes/DATA/mbc_project/1stProject/jin/data_preprocessing/data/1_2_(로우데이터_합본.csv)목적별 국적별 입국_(05년1월~25년5월).csv",
    ):
        """
        초기화 함수

        Args:
            input_file (str): 입력 파일 경로
        """
        self.input_file = input_file
        self.output_dir = "./data/processed/"
        self.raw_data = None
        self.processed_data = None

        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        데이터 로드 함수 (여러 인코딩 시도)

        Returns:
            bool: 로드 성공 여부
        """
        encodings_to_try = ["utf-8", "cp949", "euc-kr", "latin-1"]

        for encoding in encodings_to_try:
            try:
                print(f"📂 '{encoding}' 인코딩으로 데이터 로드 시도 중...")
                self.raw_data = pd.read_csv(self.input_file, encoding=encoding)
                print(
                    f"✅ 데이터 로드 성공! ({self.raw_data.shape[0]}행 × {self.raw_data.shape[1]}열)"
                )
                return True
            except Exception as e:
                print(f"❌ '{encoding}' 인코딩 실패: {str(e)}")
                continue

        print("❌ 모든 인코딩 시도 실패")
        return False

    def clean_data(self):
        """
        데이터 클리닝 함수
        """
        print("\n🧹 데이터 클리닝 시작...")

        if self.raw_data is None:
            print("❌ 데이터가 로드되지 않았습니다.")
            return

        # 두 번째 행(단위 정보) 제거
        if len(self.raw_data) > 1:
            second_row = self.raw_data.iloc[1]
            if pd.isna(second_row.iloc[0]) or "인원(명)" in str(second_row.iloc[2]):
                self.raw_data = self.raw_data.drop(self.raw_data.index[1]).reset_index(drop=True)
                print("✅ 단위 정보 행 제거 완료")

        # 컬럼명 설정
        if "국적" not in self.raw_data.columns or "목적" not in self.raw_data.columns:
            columns = list(self.raw_data.columns)
            columns[0] = "국적"
            columns[1] = "목적"
            self.raw_data.columns = columns

        # NaN 값이 있는 행 제거 및 공백 정리
        self.raw_data = self.raw_data.dropna(subset=["국적", "목적"])

        # 공백 문자 제거 (양끝 + 중간 공백 정리)
        for col in ["국적", "목적"]:
            self.raw_data[col] = (
                self.raw_data[col].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
            )

        print(f"✅ 기본 클리닝 완료: {self.raw_data.shape[0]}행")

    def remove_aggregated_rows(self):
        """
        소계/합계/교포/소개 등 불필요한 행 제거 (최적화 버전)
        """
        print("\n🗑️ 소계/합계/불필요한 항목 제거 중...")

        original_count = len(self.raw_data)

        # 모든 키워드를 한 번에 정규식으로 처리
        pattern = "|".join(self.KEYWORDS_TO_REMOVE)

        # 국적과 목적에서 키워드가 포함되지 않은 행만 유지
        mask = (
            ~self.raw_data["국적"].str.contains(pattern, na=False, case=False)
            & ~self.raw_data["목적"].str.contains(pattern, na=False, case=False)
            & (self.raw_data["국적"].str.strip() != "")
            & (self.raw_data["목적"].str.strip() != "")
        )

        self.raw_data = self.raw_data[mask]

        removed_count = original_count - len(self.raw_data)
        print(f"✅ {removed_count}개 행 제거 완료 (잔여: {len(self.raw_data)}행)")

    def reshape_to_long_format(self):
        """
        Wide format을 Long format으로 변환 (최적화 버전)
        """
        print("\n🔄 Long format으로 변환 중...")

        # 국적, 목적 컬럼을 제외한 나머지가 날짜 컬럼
        date_columns = [col for col in self.raw_data.columns if col not in ["국적", "목적"]]

        # Wide to Long 변환
        long_data = pd.melt(
            self.raw_data,
            id_vars=["국적", "목적"],
            value_vars=date_columns,
            var_name="연월",
            value_name="입국자수",
        )

        # 문자열 처리 통합 (쉼표, 따옴표 제거 + 연월 형식 변환)
        long_data["입국자수"] = (
            long_data["입국자수"].astype(str).str.replace(",", "").str.replace('"', "")
        )
        long_data["연월"] = long_data["연월"].str.replace("년", "-").str.replace("월", "")

        # 숫자가 아닌 값들과 "계" 포함 연월 제거
        numeric_mask = long_data["입국자수"].str.isnumeric()
        date_mask = long_data["연월"].str.match(r"^\d{4}-\d{2}$", na=False)

        long_data = long_data[numeric_mask & date_mask]
        long_data["입국자수"] = pd.to_numeric(long_data["입국자수"])

        self.processed_data = long_data
        print(f"✅ Long format 변환 완료: {len(self.processed_data)}행")

    def add_date_features(self):
        """
        날짜 관련 특성 변수 추가
        """
        print("\n📅 날짜 특성 변수 생성 중...")

        # 연월을 날짜로 변환
        self.processed_data["날짜"] = pd.to_datetime(
            self.processed_data["연월"] + "-01", format="%Y-%m-%d"
        )

        # 연도, 월, 분기, 계절 추가
        self.processed_data["연도"] = self.processed_data["날짜"].dt.year
        self.processed_data["월"] = self.processed_data["날짜"].dt.month
        self.processed_data["분기"] = self.processed_data["날짜"].dt.quarter
        self.processed_data["계절"] = self.processed_data["월"].map(self.SEASON_MAP)

        # 코로나 시기 구분
        covid_start = pd.to_datetime(self.COVID_START)
        covid_end = pd.to_datetime(self.COVID_END)
        self.processed_data["코로나기간"] = (
            (self.processed_data["날짜"] >= covid_start)
            & (self.processed_data["날짜"] <= covid_end)
        ).astype(int)

        # 시계열 순서 (딥러닝 모델용)
        self.processed_data = self.processed_data.sort_values(["국적", "목적", "날짜"])
        self.processed_data["시계열순서"] = (
            self.processed_data.groupby(["국적", "목적"]).cumcount() + 1
        )

        print("✅ 날짜 특성 변수 생성 완료")

    def add_lag_features(self):
        """
        지연 특성 변수 추가 및 결측치/inf 처리
        """
        print("\n⏰ 지연 특성 변수 생성 및 결측치/inf 처리 중...")

        def create_lag_features(group):
            """그룹별 지연 특성 생성 함수"""
            group = group.sort_values("날짜").copy()

            # 지연 변수 및 이동평균 컬럼 목록
            lag_cols = []
            ma_cols = []

            # 지연 변수들 생성
            for lag in [1, 3, 12]:
                col_name = f"입국자수_{lag}개월전"
                group[col_name] = group["입국자수"].shift(lag)
                lag_cols.append(col_name)

            # 이동평균들 생성
            for window in [3, 12]:
                col_name = f"입국자수_{window}개월평균"
                group[col_name] = (
                    group["입국자수"].rolling(window=window, min_periods=1).mean()
                )
                ma_cols.append(col_name)

            # 전년동월대비 증감률
            group["전년동월대비증감률"] = (
                (group["입국자수"] - group["입국자수_12개월전"]) / group["입국자수_12개월전"] * 100
            )
            
            # NaN 및 inf 값 처리
            group[lag_cols] = group[lag_cols].fillna(0)
            group[ma_cols] = group[ma_cols].fillna(0)
            group["전년동월대비증감률"] = group["전년동월대비증감률"].replace([np.inf, -np.inf], 0).fillna(0)

            return group

        # 그룹별 처리 후 결합
        self.processed_data = (
            self.processed_data.groupby(["국적", "목적"])
            .apply(create_lag_features)
            .reset_index(drop=True)
        )

        print("✅ 지연 특성 변수 생성 및 결측치/inf 처리 완료")

    def normalize_features(self):
        """
        수치형 특성 정규화 (로그 변환)
        """
        print("\n⚖️ 수치형 특성 정규화 (로그 변환) 중...")

        # 로그 변환을 적용할 컬럼 목록
        cols_to_normalize = [
            "입국자수",
            "입국자수_1개월전",
            "입국자수_3개월전",
            "입국자수_12개월전",
            "입국자수_3개월평균",
            "입국자수_12개월평균",
        ]

        # 컬럼이 데이터에 존재하는지 확인 후 로그 변환 적용
        for col in cols_to_normalize:
            if col in self.processed_data.columns:
                self.processed_data[col] = np.log1p(self.processed_data[col])
            else:
                print(f"⚠️ 정규화 대상 컬럼 '{col}'이 존재하지 않습니다.")

        print("✅ 수치형 특성 정규화 완료")

    def save_processed_data(self):
        """
        전처리된 데이터 저장
        """
        print("\n💾 전처리 데이터 저장 중...")

        output_file = os.path.join(self.output_dir, "외국인입국자_전처리완료_딥러닝용.csv")

        # 최종 컬럼 순서 (날짜, 연월 제외)
        column_order = [
            "국적",
            "목적",
            "연도",
            "월",
            "분기",
            "계절",
            "코로나기간",
            "시계열순서",
            "입국자수",
            "입국자수_1개월전",
            "입국자수_3개월전",
            "입국자수_12개월전",
            "입국자수_3개월평균",
            "입국자수_12개월평균",
            "전년동월대비증감률",
        ]

        # 존재하는 컬럼만 선택하여 저장
        available_columns = [col for col in column_order if col in self.processed_data.columns]
        final_data = self.processed_data[available_columns]

        final_data.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"✅ 저장 완료: {output_file}")
        print(f"📊 최종 데이터 형태: {final_data.shape[0]}행 × {final_data.shape[1]}열")
        print(f"\n📋 **최종 전처리 데이터 샘플:**")
        print(final_data.head(10).to_string())

        return output_file

    def get_data_summary(self):
        """
        데이터 요약 정보 출력 (최적화 버전)
        """
        if self.processed_data is not None:
            data = self.processed_data

            print(f"\n📈 **데이터 요약 정보**")
            print(f"- 총 데이터 행수: {len(data):,}")
            print(f"- 국적 수: {data['국적'].nunique()}")
            print(f"- 목적 수: {data['목적'].nunique()}")

            # 날짜 범위 계산 (간소화)
            year_month = data[["연도", "월"]].drop_duplicates().sort_values(["연도", "월"])
            min_date = year_month.iloc[0]
            max_date = year_month.iloc[-1]
            print(
                f"- 날짜 범위: {min_date['연도']}년 {min_date['월']:02d}월 ~ {max_date['연도']}년 {max_date['월']:02d}월"
            )

            covid_counts = data["코로나기간"].value_counts()
            print(f"- 코로나 기간 데이터: {covid_counts.get(1, 0):,}행")
            print(f"- 비코로나 기간 데이터: {covid_counts.get(0, 0):,}행")

            print(f"\n🏷️ **국적 목록 (상위 10개):**")
            print(data["국적"].value_counts().head(10))

            print(f"\n🎯 **목적 목록:**")
            print(data["목적"].value_counts())

    def run_preprocessing(self):
        """
        전체 전처리 프로세스 실행

        Returns:
            bool: 전처리 성공 여부
        """
        try:
            print("=" * 60)
            print("🚀 외국인 입국자 데이터 전처리 시작 (코로나 시기 포함)")
            print("=" * 60)

            # 전처리 단계별 실행
            steps = [
                (self.load_data, "데이터 로드"),
                (self.clean_data, "데이터 클리닝"),
                (self.remove_aggregated_rows, "소계/합계 제거"),
                (self.reshape_to_long_format, "Long format 변환"),
                (self.add_date_features, "날짜 특성 추가"),
                (self.add_lag_features, "지연 특성 추가 및 결측치/inf 처리"),
                (self.normalize_features, "수치형 특성 정규화"),
            ]

            # 데이터 로드 단계는 별도 처리 (반환값 확인 필요)
            if not steps[0][0]():
                return False

            # 나머지 단계들 실행
            for step_func, step_name in steps[1:]:
                step_func()

            # 최종 저장 및 요약
            self.save_processed_data()
            self.get_data_summary()

            print("\n" + "=" * 60)
            print("✅ 전처리 완료! 🎉")
            print("=" * 60)
            return True

        except Exception as e:
            print(f"\n❌ 전처리 중 오류 발생: {str(e)}")
            import traceback

            traceback.print_exc()
            return False


# 실행 부분
if __name__ == "__main__":
    processor = ForeignVisitorDataProcessor()
    success = processor.run_preprocessing()

    if success:
        print(f"\n🎯 **다음 단계:** 생성된 CSV 파일로 딥러닝 모델 학습을 진행하세요!")
    else:
        print(f"\n⚠️ 전처리 실패. 오류를 확인하고 다시 시도해주세요.")
