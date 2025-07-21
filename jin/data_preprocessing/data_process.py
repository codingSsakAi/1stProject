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
            pd.to_numeric(long_data["입국자수"].astype(str).str.replace(",", "").str.replace('"', ""), errors="coerce").fillna(0)
        )
        long_data["연월"] = long_data["연월"].str.replace("년", "-").str.replace("월", "")

        # 숫자가 아닌 값들과 "계" 포함 연월 제거
        # numeric_mask = long_data["입국자수"].str.isnumeric()
        numeric_mask = pd.to_numeric(long_data["입국자수"], errors="coerce").notnull()
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
            
            # 개선된 결측치 처리 (그룹별 특성 반영)
            group = self.handle_missing_values_intelligently(group, lag_cols, ma_cols)

            return group

        # 그룹별 처리 후 결합
        self.processed_data = (
            self.processed_data.groupby(["국적", "목적"])
            .apply(create_lag_features)
            .reset_index(drop=True)
        )

        print("✅ 지연 특성 변수 생성 및 결측치/inf 처리 완료")

    def analyze_data_characteristics(self):
        """
        국가별/목적별 데이터 특성 분석 및 소규모 국가 감지
        - 데이터 크기, 변동성, 0값 비율 등 분석
        - 홍콩과 같은 문제 국가 사전 감지
        - 최적 변환 방법 추천
        """
        print("\n🔍 국가별/목적별 데이터 특성 분석 중...")
        
        # 분석 결과 저장용 딕셔너리
        analysis_results = {}
        problem_cases = []
        
        # 국가별/목적별 그룹 분석
        groups = self.processed_data.groupby(["국적", "목적"])
        
        print(f"  📊 총 {len(groups)}개 그룹 분석 중...")
        
        for (country, purpose), group in groups:
            # 기본 통계
            data_size = len(group)
            mean_visitors = group["입국자수"].mean()
            std_visitors = group["입국자수"].std()
            min_visitors = group["입국자수"].min()
            max_visitors = group["입국자수"].max()
            zero_count = (group["입국자수"] == 0).sum()
            zero_ratio = zero_count / data_size * 100
            
            # 변동성 계산 (변동계수)
            cv = std_visitors / mean_visitors if mean_visitors > 0 else 0
            
            # 데이터 품질 등급 판정
            if zero_ratio > 40:
                quality_grade = "매우나쁨"
                problem_type = "높은_0값_비율"
            elif cv < 0.3:
                quality_grade = "나쁨" 
                problem_type = "낮은_변동성"
            elif data_size < 100:
                quality_grade = "보통"
                problem_type = "소규모_데이터"
            elif cv > 2.0:
                quality_grade = "주의"
                problem_type = "높은_변동성"
            else:
                quality_grade = "좋음"
                problem_type = "정상"
            
            # 분석 결과 저장
            analysis_results[(country, purpose)] = {
                'data_size': data_size,
                'mean': mean_visitors,
                'std': std_visitors,
                'min': min_visitors,
                'max': max_visitors,
                'zero_count': zero_count,
                'zero_ratio': zero_ratio,
                'cv': cv,
                'quality_grade': quality_grade,
                'problem_type': problem_type
            }
            
            # 문제 케이스 수집
            if quality_grade in ["매우나쁨", "나쁨", "주의"]:
                problem_cases.append({
                    'country': country,
                    'purpose': purpose,
                    'problem_type': problem_type,
                    'zero_ratio': zero_ratio,
                    'cv': cv,
                    'data_size': data_size
                })
        
        # 분석 결과를 클래스 속성으로 저장
        self.data_analysis = analysis_results
        self.problem_cases = problem_cases
        
        # 문제 케이스 리포트
        print(f"\n📋 데이터 품질 분석 결과:")
        print(f"  - 전체 그룹: {len(analysis_results)}개")
        print(f"  - 문제 그룹: {len(problem_cases)}개")
        
        # 품질 등급별 분류
        quality_counts = {}
        for result in analysis_results.values():
            grade = result['quality_grade']
            quality_counts[grade] = quality_counts.get(grade, 0) + 1
            
        print(f"  📊 품질 등급 분포:")
        for grade, count in quality_counts.items():
            print(f"    - {grade}: {count}개")
        
        # 상위 10개 문제 케이스 출력
        if problem_cases:
            print(f"\n⚠️ 주요 문제 케이스 (상위 10개):")
            sorted_problems = sorted(problem_cases, 
                                   key=lambda x: (x['zero_ratio'], -x['cv']), 
                                   reverse=True)[:10]
            
            for i, case in enumerate(sorted_problems, 1):
                print(f"  {i:2d}. {case['country']}-{case['purpose']}: "
                      f"{case['problem_type']} (0값: {case['zero_ratio']:.1f}%, "
                      f"변동성: {case['cv']:.2f})")
        
        print("✅ 데이터 특성 분석 완료")
        return analysis_results, problem_cases

    def add_advanced_features(self):
        """
        향상된 피처 엔지니어링 (소규모 국가 예측 성능 개선용)
        - 변동성 지표 (Rolling std, 변동계수)
        - 모멘텀 지표 (상승/하락 추세)
        - 상대변화율 (전월대비, 전년동월대비)
        - 계절성 지표 (계절별 평균 대비)
        - 트렌드 지표 (증가/감소 패턴)
        """
        print("\n🚀 향상된 피처 엔지니어링 시작...")
        
        def create_advanced_features(group):
            """그룹별 고급 피처 생성 함수"""
            group = group.sort_values("날짜").copy()
            
            # 1. 변동성 지표들
            print(f"    📊 변동성 지표 생성: {group.iloc[0]['국적']}-{group.iloc[0]['목적']}")
            
            # Rolling 표준편차 (3개월, 6개월, 12개월)
            for window in [3, 6, 12]:
                col_name = f"변동성_{window}개월"
                group[col_name] = group["입국자수"].rolling(window=window, min_periods=1).std().fillna(0)
            
            # 변동계수 (평균 대비 표준편차)
            group["변동계수_3개월"] = (
                group["변동성_3개월"] / group["입국자수_3개월평균"].replace(0, 1)
            ).fillna(0)
            
            # 2. 모멘텀 지표들
            # 전월대비 변화율
            group["전월대비변화율"] = (
                (group["입국자수"] - group["입국자수_1개월전"]) / 
                group["입국자수_1개월전"].replace(0, 1) * 100
            ).replace([np.inf, -np.inf], 0).fillna(0)
            
            # 3개월 트렌드 (상승=1, 하락=-1, 유지=0)
            group["트렌드_3개월"] = np.where(
                group["입국자수"] > group["입국자수_3개월전"], 1,
                np.where(group["입국자수"] < group["입국자수_3개월전"], -1, 0)
            )
            
            # 연속 상승/하락 개월 수
            trend_changes = group["트렌드_3개월"].diff().fillna(0) != 0
            group["트렌드연속성"] = (~trend_changes).cumsum()
            
            # 3. 계절성 지표들
            # 월별 평균 계산 (과거 데이터 기반)
            월별_평균 = group.groupby("월")["입국자수"].expanding().mean().reset_index(level=0, drop=True)
            group["월별평균대비"] = (group["입국자수"] / 월별_평균.replace(0, 1)).fillna(1)
            
            # 4. 상대적 크기 지표들
            # 전체 기간 대비 현재값 위치 (백분위)
            group["상대적크기"] = group["입국자수"].rank(pct=True)
            
            # 최근 12개월 최대값 대비 비율
            group["최근최대값대비"] = (
                group["입국자수"] / 
                group["입국자수"].rolling(window=12, min_periods=1).max()
            ).fillna(0)
            
            # 5. 안정성 지표들
            # 최근 6개월 안정성 (표준편차/평균)
            recent_stability = (
                group["입국자수"].rolling(window=6, min_periods=1).std() /
                group["입국자수"].rolling(window=6, min_periods=1).mean()
            )
            group["안정성지수"] = (1 / (1 + recent_stability)).fillna(0)
            
            # 6. 코로나 관련 지표들 (코로나 전후 비교)
            pre_covid_mean = group[group["코로나기간"] == 0]["입국자수"].mean()
            if pd.notna(pre_covid_mean) and pre_covid_mean > 0:
                group["코로나전대비"] = (group["입국자수"] / pre_covid_mean).fillna(0)
            else:
                group["코로나전대비"] = 1.0
            
            return group
        
        # 그룹별 처리
        print("  🔄 국가별/목적별 고급 피처 생성 중...")
        self.processed_data = (
            self.processed_data.groupby(["국적", "목적"])
            .apply(create_advanced_features)
            .reset_index(drop=True)
        )
        
        # 생성된 피처 목록
        advanced_features = [
            "변동성_3개월", "변동성_6개월", "변동성_12개월",
            "변동계수_3개월", "전월대비변화율", "트렌드_3개월", 
            "트렌드연속성", "월별평균대비", "상대적크기",
            "최근최대값대비", "안정성지수", "코로나전대비"
        ]
        
        print(f"\n✅ 향상된 피처 엔지니어링 완료!")
        print(f"📊 생성된 고급 피처: {len(advanced_features)}개")
        for feature in advanced_features:
            print(f"  - {feature}")
        
        return advanced_features

    def apply_adaptive_transformation(self, column_name):
        """
        데이터 특성에 따른 적응형 변환 적용
        - 높은 0값 비율: Square root + 상수 변환
        - 낮은 변동성: Robust scaling 
        - 높은 변동성: Log + clipping
        - 정상 데이터: 기본 로그변환
        """
        if not hasattr(self, 'data_analysis'):
            # 분석 결과가 없으면 기본 로그변환 적용
            return np.log1p(self.processed_data[column_name])
        
        # 국가별/목적별로 적응형 변환 적용
        result_series = self.processed_data[column_name].copy()
        
        for (country, purpose), analysis in self.data_analysis.items():
            # 해당 국가/목적의 행들 마스크
            mask = (
                (self.processed_data['국적'] == country) & 
                (self.processed_data['목적'] == purpose)
            )
            
            if not mask.any():
                continue
                
            # 해당 그룹의 데이터
            group_data = self.processed_data.loc[mask, column_name].copy()
            
            # 문제 유형에 따른 변환 선택
            problem_type = analysis['problem_type']
            zero_ratio = analysis['zero_ratio']
            cv = analysis['cv']
            
            if problem_type == "높은_0값_비율":
                # Square root + 상수 변환 (0값에 강함)
                transformed = np.sqrt(group_data + 1)
                print(f"      🟡 {country}-{purpose}: 제곱근 변환 (0값비율: {zero_ratio:.1f}%)")
                
            elif problem_type == "낮은_변동성":
                # 민감한 변환 (작은 변화도 확대)
                if group_data.std() > 0:
                    # Robust scaling (이상치에 덜 민감)
                    median = group_data.median()
                    mad = np.median(np.abs(group_data - median))
                    if mad > 0:
                        transformed = (group_data - median) / mad
                    else:
                        transformed = np.log1p(group_data)
                    print(f"      🔵 {country}-{purpose}: 로버스트 스케일링 (변동성: {cv:.2f})")
                else:
                    transformed = np.log1p(group_data)
                    
            elif problem_type == "높은_변동성":
                # Log + clipping (극값 제한)
                log_data = np.log1p(group_data)
                q99 = log_data.quantile(0.99)
                q01 = log_data.quantile(0.01)
                transformed = np.clip(log_data, q01, q99)
                print(f"      🔴 {country}-{purpose}: 로그+클리핑 변환 (변동성: {cv:.2f})")
                
            elif problem_type == "소규모_데이터":
                # 안정적인 변환 (과적합 방지)
                transformed = np.log1p(group_data * 0.9 + group_data.mean() * 0.1)
                print(f"      🟤 {country}-{purpose}: 안정화 변환 (데이터수: {analysis['data_size']})")
                
            else:  # 정상 케이스
                # 기본 로그변환
                transformed = np.log1p(group_data)
                print(f"      🟢 {country}-{purpose}: 기본 로그변환 (정상)")
            
            # 변환된 값을 결과에 적용
            result_series.loc[mask] = transformed
            
        return result_series

    def handle_missing_values_intelligently(self, group, lag_cols, ma_cols):
        """
        그룹별 특성을 반영한 정교한 결측치 처리
        - 계절성 고려한 보간
        - 트렌드 반영한 예측값 대체
        - 그룹 평균 및 최빈값 활용
        - 이상치 및 inf 값 적절한 처리
        """
        country = group.iloc[0]['국적']
        purpose = group.iloc[0]['목적']
        
        print(f"      🔧 결측치 처리: {country}-{purpose}")
        
        # 1. 지연 변수들 처리 (계절성 고려)
        for lag_col in lag_cols:
            if lag_col in group.columns:
                # 결측치 개수 확인
                missing_count = group[lag_col].isna().sum()
                if missing_count > 0:
                    print(f"        📝 {lag_col}: {missing_count}개 결측치 처리")
                    
                    # 계절성 고려한 보간
                    if len(group) >= 12 and missing_count < len(group) * 0.5:
                        # 같은 월의 평균값으로 대체
                        group[lag_col] = group[lag_col].fillna(
                            group.groupby('월')[lag_col].transform('mean')
                        )
                    
                    # 여전히 결측치가 있으면 전후 값의 평균으로 보간
                    if group[lag_col].isna().any():
                        group[lag_col] = group[lag_col].interpolate(method='linear')
                    
                    # 맨 앞/뒤 결측치는 0으로 대체
                    group[lag_col] = group[lag_col].fillna(0)
        
        # 2. 이동평균들 처리 (트렌드 반영)
        for ma_col in ma_cols:
            if ma_col in group.columns:
                missing_count = group[ma_col].isna().sum()
                if missing_count > 0:
                    print(f"        📊 {ma_col}: {missing_count}개 결측치 처리")
                    
                    # 이동평균은 forward fill 후 backward fill
                    group[ma_col] = group[ma_col].fillna(method='ffill').fillna(method='bfill')
                    
                    # 여전히 결측치가 있으면 전체 평균으로 대체
                    if group[ma_col].isna().any():
                        overall_mean = group[ma_col].mean()
                        if pd.notna(overall_mean):
                            group[ma_col] = group[ma_col].fillna(overall_mean)
                        else:
                            group[ma_col] = group[ma_col].fillna(0)
        
        # 3. 전년동월대비증감률 특별 처리
        if "전년동월대비증감률" in group.columns:
            # inf 값 처리 (매우 큰/작은 값으로 제한)
            group["전년동월대비증감률"] = group["전년동월대비증감률"].replace([np.inf], 1000)
            group["전년동월대비증감률"] = group["전년동월대비증감률"].replace([-np.inf], -1000)
            
            # 이상치 처리 (99%tile 기준으로 클리핑)
            if len(group) > 10:
                q99 = group["전년동월대비증감률"].quantile(0.99)
                q01 = group["전년동월대비증감률"].quantile(0.01)
                group["전년동월대비증감률"] = np.clip(
                    group["전년동월대비증감률"], q01, q99
                )
            
            # 결측치는 0으로 대체 (증감률이므로 중립값)
            group["전년동월대비증감률"] = group["전년동월대비증감률"].fillna(0)
        
        # 4. 전체적인 검증 및 최종 정리
        numeric_columns = group.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 여전히 inf나 극대값이 있는지 확인
            if np.isinf(group[col]).any():
                print(f"        ⚠️  {col}: inf 값 발견, 0으로 대체")
                group[col] = group[col].replace([np.inf, -np.inf], 0)
            
            # 마지막 결측치 확인
            if group[col].isna().any():
                print(f"        ⚠️  {col}: 잔여 결측치 0으로 대체")
                group[col] = group[col].fillna(0)
        
        return group

    def preserve_and_transform_features(self):
        """
        원본 데이터 보존하면서 다양한 변환 버전 생성
        - 원본 데이터는 절대 변경하지 않음
        - 로그변환, 표준화, 정규화 등 다양한 옵션 제공
        - 검증 및 시각화를 위한 원본 데이터 유지
        """
        print("\n📦 원본 데이터 보존 및 다중 변환 생성 중...")

        # 원본 데이터 백업 컬럼 목록 (절대 변경 안 됨)
        original_columns = [
            "입국자수",
            "입국자수_1개월전",
            "입국자수_3개월전", 
            "입국자수_12개월전",
            "입국자수_3개월평균",
            "입국자수_12개월평균",
        ]

        # 1단계: 원본 데이터 백업 (원본 보존)
        print("  📋 원본 데이터 백업 중...")
        backup_count = 0
        for col in original_columns:
            if col in self.processed_data.columns:
                backup_col = f"{col}_원본"
                self.processed_data[backup_col] = self.processed_data[col].copy()
                backup_count += 1
                print(f"    ✅ {col} → {backup_col}")
        
        print(f"  📦 총 {backup_count}개 컬럼 원본 백업 완료")

        # 2단계: 로그 변환 버전 생성 (기존 동작 유지)
        print("  📈 로그 변환 버전 생성 중...")
        log_transform_count = 0
        for col in original_columns:
            if col in self.processed_data.columns:
                log_col = f"{col}_로그변환"
                self.processed_data[log_col] = np.log1p(self.processed_data[col])
                log_transform_count += 1
                print(f"    📊 {col} → {log_col}")
        
        print(f"  📈 총 {log_transform_count}개 컬럼 로그변환 완료")

        # 3단계: 정규화(MinMax) 버전 생성
        print("  ⚖️ 정규화(MinMax) 버전 생성 중...")
        from sklearn.preprocessing import MinMaxScaler
        
        minmax_count = 0
        for col in original_columns:
            if col in self.processed_data.columns:
                minmax_col = f"{col}_정규화"
                scaler = MinMaxScaler()
                # reshape(-1, 1)로 2D 배열로 변환
                values = self.processed_data[col].values.reshape(-1, 1)
                self.processed_data[minmax_col] = scaler.fit_transform(values).flatten()
                minmax_count += 1
                print(f"    ⚖️ {col} → {minmax_col}")
        
        print(f"  ⚖️ 총 {minmax_count}개 컬럼 정규화 완료")

        # 4단계: 표준화(Z-score) 버전 생성
        print("  📏 표준화(Z-score) 버전 생성 중...")
        from sklearn.preprocessing import StandardScaler
        
        standardized_count = 0
        for col in original_columns:
            if col in self.processed_data.columns:
                std_col = f"{col}_표준화"
                scaler = StandardScaler()
                # reshape(-1, 1)로 2D 배열로 변환
                values = self.processed_data[col].values.reshape(-1, 1)
                self.processed_data[std_col] = scaler.fit_transform(values).flatten()
                standardized_count += 1
                print(f"    📏 {col} → {std_col}")
        
        print(f"  📏 총 {standardized_count}개 컬럼 표준화 완료")

        # 5단계: 소규모 국가 맞춤 변환 (데이터 특성 기반)
        print("  🎯 소규모 국가 맞춤 변환 생성 중...")
        optimized_count = 0  # 변수 초기화
        if hasattr(self, 'data_analysis'):
            for col in original_columns:
                if col in self.processed_data.columns:
                    optimized_col = f"{col}_최적화"
                    self.processed_data[optimized_col] = self.apply_adaptive_transformation(col)
                    optimized_count += 1
                    print(f"    🎯 {col} → {optimized_col}")
            
            print(f"  🎯 총 {optimized_count}개 컬럼 적응형 변환 완료")
        else:
            print("  ⚠️ 데이터 분석 결과가 없어 적응형 변환을 건너뜁니다.")

        # 6단계: 기존 동작 유지를 위해 최적화된 값을 원본 컬럼에 복사
        print("  🔄 기존 모델 호환성을 위한 최적화 변환 값 적용 중...")
        for col in original_columns:
            if col in self.processed_data.columns:
                # 최적화 버전이 있으면 사용, 없으면 로그변환 사용
                optimized_col = f"{col}_최적화"
                log_col = f"{col}_로그변환"
                
                if optimized_col in self.processed_data.columns:
                    self.processed_data[col] = self.processed_data[optimized_col].copy()
                    print(f"    🎯 {optimized_col} → {col} (적응형 변환)")
                elif log_col in self.processed_data.columns:
                    self.processed_data[col] = self.processed_data[log_col].copy()
                    print(f"    🔄 {log_col} → {col} (기본 로그변환)")

        print("\n✅ 원본 보존 및 다중 변환 생성 완료!")
        print(f"📊 생성된 변환 버전:")
        print(f"  - 원본 백업: {backup_count}개 (검증용)")
        print(f"  - 로그 변환: {log_transform_count}개 (시각화용)")  
        print(f"  - 정규화: {minmax_count}개 (MinMax 스케일링)")
        print(f"  - 표준화: {standardized_count}개 (Z-score 정규화)")
        if hasattr(self, 'data_analysis'):
            print(f"  - 적응형변환: {optimized_count}개 (소규모 국가 최적화)")
            print(f"  - 기존 컬럼: 적응형 변환 값 적용 (최적화)")
        else:
            print(f"  - 기존 컬럼: 로그변환 값 적용 (호환성 유지)")

    def normalize_features(self):
        """
        기존 함수명 유지 (호환성) - 새로운 함수로 리다이렉트
        """
        print("\n⚖️ 수치형 특성 정규화 시작...")
        self.preserve_and_transform_features()

    def save_processed_data(self):
        """
        전처리된 데이터 저장 (원본 보존 버전 포함)
        """
        print("\n💾 전처리 데이터 저장 중...")

        output_file = os.path.join(self.output_dir, "외국인입국자_전처리완료_딥러닝용.csv")

        # 기본 컬럼 순서
        base_columns = [
            "국적",
            "목적",
            "연도",
            "월",
            "분기",
            "계절",
            "코로나기간",
            "시계열순서",
        ]

        # 모델링 컬럼 (기존 호환성 유지)
        modeling_columns = [
            "입국자수",
            "입국자수_1개월전",
            "입국자수_3개월전",
            "입국자수_12개월전",
            "입국자수_3개월평균",
            "입국자수_12개월평균",
            "전년동월대비증감률",
        ]

        # 원본 보존 컬럼들
        original_columns = [col for col in self.processed_data.columns if col.endswith("_원본")]
        
        # 로그변환 컬럼들  
        log_columns = [col for col in self.processed_data.columns if col.endswith("_로그변환")]
        
        # 정규화 컬럼들
        normalized_columns = [col for col in self.processed_data.columns if col.endswith("_정규화")]
        
        # 표준화 컬럼들 
        standardized_columns = [col for col in self.processed_data.columns if col.endswith("_표준화")]
        
        # 최적화 컬럼들
        optimized_columns = [col for col in self.processed_data.columns if col.endswith("_최적화")]
        
        # 고급 피처 컬럼들 (향상된 피처 엔지니어링)
        advanced_feature_keywords = [
            "변동성_", "변동계수_", "전월대비변화율", "트렌드_", "트렌드연속성",
            "월별평균대비", "상대적크기", "최근최대값대비", "안정성지수", "코로나전대비"
        ]
        advanced_columns = [col for col in self.processed_data.columns 
                          if any(keyword in col for keyword in advanced_feature_keywords)]

        # 최종 컬럼 순서 (체계적으로 정리)
        column_order = (
            base_columns + 
            modeling_columns + 
            advanced_columns +  # 고급 피처 추가
            original_columns + 
            log_columns + 
            normalized_columns +
            standardized_columns +  # 표준화 추가
            optimized_columns  # 최적화 추가
        )

        # 존재하는 컬럼만 선택하여 저장
        available_columns = [col for col in column_order if col in self.processed_data.columns]
        final_data = self.processed_data[available_columns]

        final_data.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"✅ 저장 완료: {output_file}")
        print(f"📊 최종 데이터 형태: {final_data.shape[0]}행 × {final_data.shape[1]}열")
        
        # 컬럼 분류별 개수 출력
        original_count = len([col for col in available_columns if col.endswith("_원본")])
        log_count = len([col for col in available_columns if col.endswith("_로그변환")])
        normalized_count = len([col for col in available_columns if col.endswith("_정규화")])
        standardized_count = len([col for col in available_columns if col.endswith("_표준화")])
        optimized_count = len([col for col in available_columns if col.endswith("_최적화")])
        advanced_count = len(advanced_columns)
        
        print(f"\n📊 저장된 컬럼 분류:")
        print(f"  - 기본 정보: {len(base_columns)}개")
        print(f"  - 모델링용: {len(modeling_columns)}개 (기존 호환성)")
        print(f"  - 고급 피처: {advanced_count}개 (예측 성능 향상용)")
        print(f"  - 원본 백업: {original_count}개 (검증용)")
        print(f"  - 로그변환: {log_count}개 (시각화용)")  
        print(f"  - 정규화: {normalized_count}개 (MinMax 스케일링)")
        print(f"  - 표준화: {standardized_count}개 (Z-score 정규화)")
        print(f"  - 적응형변환: {optimized_count}개 (소규모 국가 최적화)")

        print(f"\n📋 **최종 전처리 데이터 샘플 (기본 컬럼):**")
        sample_columns = base_columns + modeling_columns[:3]  # 샘플 출력용
        sample_data = final_data[[col for col in sample_columns if col in final_data.columns]]
        print(sample_data.head(5).to_string())

        return output_file

    def save_analysis_report(self):
        """
        데이터 특성 분석 결과를 CSV로 저장
        """
        if not hasattr(self, 'data_analysis') or not hasattr(self, 'problem_cases'):
            print("⚠️ 분석 결과가 없어 리포트 저장을 건너뜁니다.")
            return
        
        print("\n📋 데이터 분석 리포트 저장 중...")
        
        # 분석 결과를 DataFrame으로 변환
        analysis_df = pd.DataFrame.from_dict(self.data_analysis, orient='index')
        analysis_df.index = pd.MultiIndex.from_tuples(analysis_df.index, names=['국적', '목적'])
        analysis_df.reset_index(inplace=True)
        
        # 분석 리포트 저장
        analysis_output = os.path.join(self.output_dir, "데이터품질분석_리포트.csv")
        analysis_df.to_csv(analysis_output, index=False, encoding="utf-8-sig")
        
        # 문제 케이스만 별도 저장
        if self.problem_cases:
            problem_df = pd.DataFrame(self.problem_cases)
            problem_output = os.path.join(self.output_dir, "문제케이스_리포트.csv")
            problem_df.to_csv(problem_output, index=False, encoding="utf-8-sig")
            
            print(f"✅ 분석 리포트 저장 완료:")
            print(f"  - 전체 분석: {analysis_output}")
            print(f"  - 문제 케이스: {problem_output}")
        else:
            print(f"✅ 분석 리포트 저장 완료: {analysis_output}")

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
                (self.analyze_data_characteristics, "데이터 특성 분석 및 문제 케이스 감지"),
                (self.add_advanced_features, "향상된 피처 엔지니어링"),
                (self.normalize_features, "원본 보존 및 다중 변환 생성"),
            ]

            # 데이터 로드 단계는 별도 처리 (반환값 확인 필요)
            if not steps[0][0]():
                return False

            # 나머지 단계들 실행
            for step_func, step_name in steps[1:]:
                step_func()

            # 최종 저장 및 요약
            self.save_processed_data()
            self.save_analysis_report()  # 분석 리포트 저장 추가
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
