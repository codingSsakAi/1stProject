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
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class ForeignVisitorDataProcessor:
    """외국인 입국자 데이터 전처리 클래스"""

    def __init__(
        self,
        input_file="../han/data/1_2_(로우데이터_합본.csv)목적별_국적별_입국(05년1월~25년5월).csv",
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

        # 데이터가 로드되었는지 확인
        if self.raw_data is None:
            print("❌ 데이터가 로드되지 않았습니다.")
            return

        # 두 번째 행(단위 정보) 제거
        if len(self.raw_data) > 1:
            # 두 번째 행이 단위 정보인지 확인
            second_row = self.raw_data.iloc[1]
            if pd.isna(second_row.iloc[0]) or "인원(명)" in str(second_row.iloc[2]):
                self.raw_data = self.raw_data.drop(self.raw_data.index[1])
                self.raw_data = self.raw_data.reset_index(drop=True)
                print("✅ 단위 정보 행 제거 완료")

        # 컬럼명이 이미 '국적', '목적'으로 되어 있는지 확인
        if "국적" not in self.raw_data.columns or "목적" not in self.raw_data.columns:
            # 첫 번째와 두 번째 컬럼을 국적, 목적으로 설정
            columns = list(self.raw_data.columns)
            columns[0] = "국적"
            columns[1] = "목적"
            self.raw_data.columns = columns

        # NaN 값이 있는 행 제거
        self.raw_data = self.raw_data.dropna(subset=["국적", "목적"])

        # 공백 문자 제거 (양끝 + 중간 공백 정리)
        self.raw_data["국적"] = (
            self.raw_data["국적"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
        )
        self.raw_data["목적"] = (
            self.raw_data["목적"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
        )

        print(f"✅ 기본 클리닝 완료: {self.raw_data.shape[0]}행")

    def remove_aggregated_rows(self):
        """
        소계/합계/교포/소개 등 불필요한 행 제거
        """
        print("\n🗑️ 소계/합계/불필요한 항목 제거 중...")

        # 제거할 키워드 목록 (딥러닝 모델에 불필요한 집계성 데이터)
        keywords_to_remove = [
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

        original_count = len(self.raw_data)

        # 국적과 목적에서 키워드 제거
        for keyword in keywords_to_remove:
            # 국적에서 키워드 포함된 행 제거
            mask_nationality = ~self.raw_data["국적"].str.contains(keyword, na=False, case=False)
            # 목적에서 키워드 포함된 행 제거
            mask_purpose = ~self.raw_data["목적"].str.contains(keyword, na=False, case=False)

            # 두 조건 모두 만족하는 행만 유지
            self.raw_data = self.raw_data[mask_nationality & mask_purpose]

        # "계"만 있는 행 제거 (정확히 "계"인 경우)
        self.raw_data = self.raw_data[
            (self.raw_data["국적"].str.strip() != "계")
            & (self.raw_data["목적"].str.strip() != "계")
        ]

        # 빈 값이나 공백만 있는 행 제거
        self.raw_data = self.raw_data[
            (self.raw_data["국적"].str.strip() != "") & (self.raw_data["목적"].str.strip() != "")
        ]

        removed_count = original_count - len(self.raw_data)
        print(f"✅ {removed_count}개 행 제거 완료 (잔여: {len(self.raw_data)}행)")

    def reshape_to_long_format(self):
        """
        Wide format을 Long format으로 변환
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

        # 쉼표가 포함된 숫자 처리
        long_data["입국자수"] = (
            long_data["입국자수"].astype(str).str.replace(",", "").str.replace('"', "")
        )

        # 숫자가 아닌 값들 제거
        long_data = long_data[long_data["입국자수"].str.isnumeric()]
        long_data["입국자수"] = pd.to_numeric(long_data["입국자수"])

        # 연월 정리 (예: "2005년01월" -> "2005-01")
        long_data["연월"] = long_data["연월"].str.replace("년", "-").str.replace("월", "")

        # "계" 값이 포함된 연월 데이터 제거
        long_data = long_data[~long_data["연월"].str.contains("계", na=False)]

        # 연월이 올바른 날짜 형식인지 확인 (YYYY-MM 형태)
        import re

        date_pattern = r"^\d{4}-\d{2}$"
        long_data = long_data[long_data["연월"].str.match(date_pattern, na=False)]

        self.processed_data = long_data
        print(f"✅ Long format 변환 완료: {len(self.processed_data)}행")

    def add_date_features(self):
        """
        날짜 관련 특성 변수 추가
        """
        print("\n📅 날짜 특성 변수 생성 중...")

        # 연월을 날짜로 변환 (예: 2015-01 -> 2015-01-01)
        try:
            self.processed_data["날짜"] = pd.to_datetime(
                self.processed_data["연월"] + "-01", format="%Y-%m-%d"
            )
        except:
            # 다른 형식 시도
            self.processed_data["날짜"] = pd.to_datetime(
                self.processed_data["연월"], infer_datetime_format=True
            )

        # 연도, 월, 분기, 계절 추가
        self.processed_data["연도"] = self.processed_data["날짜"].dt.year
        self.processed_data["월"] = self.processed_data["날짜"].dt.month
        self.processed_data["분기"] = self.processed_data["날짜"].dt.quarter

        # 계절 정의 (3-5월: 봄, 6-8월: 여름, 9-11월: 가을, 12-2월: 겨울)
        season_map = {
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
        self.processed_data["계절"] = self.processed_data["월"].map(season_map)

        # 코로나 시기 구분 (2020년 3월 ~ 2022년 6월)
        covid_start = pd.to_datetime("2020-03-01")
        covid_end = pd.to_datetime("2022-06-30")
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
        지연 특성 변수 추가 (딥러닝 모델용)
        """
        print("\n⏰ 지연 특성 변수 생성 중...")

        # 국적-목적별로 그룹화
        grouped = self.processed_data.groupby(["국적", "목적"])

        # 각 그룹별로 지연 변수 생성
        lag_features = []

        for name, group in grouped:
            group = group.sort_values("날짜").copy()

            # 1개월전, 3개월전, 12개월전 입국자수
            group["입국자수_1개월전"] = group["입국자수"].shift(1)
            group["입국자수_3개월전"] = group["입국자수"].shift(3)
            group["입국자수_12개월전"] = group["입국자수"].shift(12)

            # 이동평균 (3개월, 12개월)
            group["입국자수_3개월평균"] = group["입국자수"].rolling(window=3, min_periods=1).mean()
            group["입국자수_12개월평균"] = (
                group["입국자수"].rolling(window=12, min_periods=1).mean()
            )

            # 전년동월대비 증감률
            group["전년동월대비증감률"] = (
                (group["입국자수"] - group["입국자수_12개월전"]) / group["입국자수_12개월전"] * 100
            ).fillna(0)

            lag_features.append(group)

        # 모든 그룹 합치기
        self.processed_data = pd.concat(lag_features, ignore_index=True)

        print("✅ 지연 특성 변수 생성 완료")

    def save_processed_data(self):
        """
        전처리된 데이터 저장
        """
        print("\n💾 전처리 데이터 저장 중...")

        # 파일명 생성
        output_file = os.path.join(self.output_dir, "외국인입국자_전처리완료_딥러닝용.csv")

        # 컬럼 순서 정리 (날짜, 연월 컬럼 제거)
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

        # 존재하는 컬럼만 선택
        available_columns = [col for col in column_order if col in self.processed_data.columns]
        final_data = self.processed_data[available_columns]

        # CSV 저장
        final_data.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"✅ 저장 완료: {output_file}")
        print(f"📊 최종 데이터 형태: {final_data.shape[0]}행 × {final_data.shape[1]}열")

        # 샘플 데이터 출력
        print(f"\n📋 **최종 전처리 데이터 샘플:**")
        print(final_data.head(10).to_string())

        return output_file

    def get_data_summary(self):
        """
        데이터 요약 정보 출력
        """
        if self.processed_data is not None:
            print(f"\n📈 **데이터 요약 정보**")
            print(f"- 총 데이터 행수: {len(self.processed_data):,}")
            print(f"- 국적 수: {self.processed_data['국적'].nunique()}")
            print(f"- 목적 수: {self.processed_data['목적'].nunique()}")
            print(
                f"- 날짜 범위: {self.processed_data['날짜'].min()} ~ {self.processed_data['날짜'].max()}"
            )
            print(f"- 코로나 기간 데이터: {(self.processed_data['코로나기간'] == 1).sum():,}행")
            print(f"- 비코로나 기간 데이터: {(self.processed_data['코로나기간'] == 0).sum():,}행")

            print(f"\n🏷️ **국적 목록 (상위 10개):**")
            print(self.processed_data["국적"].value_counts().head(10))

            print(f"\n🎯 **목적 목록:**")
            print(self.processed_data["목적"].value_counts())

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

            # 1. 데이터 로드
            if not self.load_data():
                return False

            # 2. 데이터 클리닝
            self.clean_data()

            # 3. 소계/합계 제거
            self.remove_aggregated_rows()

            # 4. Long format 변환
            self.reshape_to_long_format()

            # 5. 날짜 특성 추가
            self.add_date_features()

            # 6. 지연 특성 추가
            self.add_lag_features()

            # 7. 데이터 저장
            output_file = self.save_processed_data()

            # 8. 요약 정보 출력
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
    # 데이터 전처리 실행
    processor = ForeignVisitorDataProcessor()
    success = processor.run_preprocessing()

    if success:
        print(f"\n🎯 **다음 단계:** 생성된 CSV 파일로 딥러닝 모델 학습을 진행하세요!")
    else:
        print(f"\n⚠️ 전처리 실패. 오류를 확인하고 다시 시도해주세요.")
