# -*- coding: utf-8 -*-
"""
데이터 로드 및 전처리 클래스
Author: Jin
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
import sys
import os

# 프로젝트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.config import *
from scripts.utils import *

logger = logging.getLogger(__name__)


class DataLoader:
    """데이터 로드 및 전처리 클래스"""

    def __init__(self, data_file: str = DATA_FILE):
        """
        초기화

        Args:
            data_file: 데이터 파일 경로
        """
        self.data_file = data_file
        self.raw_data = None
        self.processed_data = None
        self.encoders = {}
        self.scaler = None

        logger.info(f"📂 데이터 로더 초기화: {data_file}")

    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        try:
            self.raw_data = pd.read_csv(self.data_file, encoding="utf-8-sig")
            logger.info(
                f"✅ 데이터 로드 완료: {self.raw_data.shape[0]}행 × {self.raw_data.shape[1]}열"
            )
            return self.raw_data
        except Exception as e:
            logger.error(f"❌ 데이터 로드 실패: {str(e)}")
            raise

    def preprocess_data(self) -> pd.DataFrame:
        """데이터 전처리"""
        if self.raw_data is None:
            self.load_data()

        df = self.raw_data.copy()

        # 결측값 처리
        logger.info("🔄 결측값 처리 중...")
        df = df.fillna(0)

        # 카테고리 인코딩
        logger.info("🔄 카테고리 인코딩 중...")
        df, self.encoders = encode_categorical_features(df)

        # 날짜 컬럼 추가 (필요시)
        if "날짜" not in df.columns:
            df["날짜"] = pd.to_datetime(
                df["연도"].astype(str) + "-" + df["월"].astype(str).str.zfill(2) + "-01"
            )

        # 🔧 무한대 값 및 이상치 처리 추가
        logger.info("🔄 무한대 값 및 이상치 처리 중...")

        # 무한대 값을 0으로 변경
        df = df.replace([np.inf, -np.inf], 0)

        # NaN 값 재확인 및 처리
        df = df.fillna(0)

        # 타겟 컬럼 이상치 처리 (99.9% 분위수 이상 값을 캐핑)
        if TARGET_COLUMN in df.columns:
            upper_limit = df[TARGET_COLUMN].quantile(0.999)
            original_max = df[TARGET_COLUMN].max()
            df[TARGET_COLUMN] = np.clip(df[TARGET_COLUMN], 0, upper_limit)
            logger.info(
                f"   - 타겟 컬럼 이상치 처리: 최대값 {original_max:.0f} → {upper_limit:.0f}"
            )

        # 수치형 컬럼들의 이상치 처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != TARGET_COLUMN:  # 타겟 컬럼은 이미 처리했으므로 제외
                # 극값 제거 (99.9% 분위수 기준)
                upper_limit = df[col].quantile(0.999)
                if upper_limit > 0:
                    df[col] = np.clip(df[col], 0, upper_limit)

        # 모든 수치형 데이터가 유한한지 확인
        for col in numeric_columns:
            if not np.isfinite(df[col]).all():
                logger.warning(f"⚠️ {col} 컬럼에 여전히 무한대 값이 있습니다. 0으로 대체합니다.")
                df[col] = df[col].replace([np.inf, -np.inf], 0)
                df[col] = df[col].fillna(0)

        # 정렬
        df = df.sort_values(["국적", "목적", "연도", "월"]).reset_index(drop=True)

        self.processed_data = df
        logger.info(f"✅ 데이터 전처리 완료: {df.shape[0]}행 × {df.shape[1]}열")

        return df

    def split_data(
        self,
        train_end_year: int = TRAIN_END_YEAR,
        val_end_year: int = VAL_END_YEAR,
        test_start_year: int = TEST_START_YEAR,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터 분할"""
        if self.processed_data is None:
            self.preprocess_data()

        df = self.processed_data.copy()

        # 시간 기반 분할
        train_data = df[df["연도"] <= train_end_year]
        val_data = df[(df["연도"] > train_end_year) & (df["연도"] <= val_end_year)]
        test_data = df[df["연도"] >= test_start_year]

        logger.info(f"📊 데이터 분할 완료:")
        logger.info(
            f"   - 학습 데이터: {len(train_data)}행 ({train_data['연도'].min()}-{train_data['연도'].max()})"
        )
        logger.info(
            f"   - 검증 데이터: {len(val_data)}행 ({val_data['연도'].min()}-{val_data['연도'].max()})"
        )
        logger.info(
            f"   - 테스트 데이터: {len(test_data)}행 ({test_data['연도'].min()}-{test_data['연도'].max()})"
        )

        return train_data, val_data, test_data

    def prepare_lstm_dataset(
        self, sequence_length: int = SEQUENCE_LENGTH
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """LSTM용 데이터셋 준비"""
        train_data, val_data, test_data = self.split_data()

        logger.info("🔄 LSTM 데이터셋 준비 중...")

        # 시퀀스 데이터 생성
        X_train, y_train = prepare_lstm_data(train_data, sequence_length)
        X_val, y_val = prepare_lstm_data(val_data, sequence_length)
        X_test, y_test = prepare_lstm_data(test_data, sequence_length)

        logger.info(f"📊 LSTM 데이터셋 크기:")
        logger.info(f"   - 학습: X{X_train.shape}, y{y_train.shape}")
        logger.info(f"   - 검증: X{X_val.shape}, y{y_val.shape}")
        logger.info(f"   - 테스트: X{X_test.shape}, y{y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_data_summary(self) -> Dict[str, Any]:
        """데이터 요약 정보"""
        if self.processed_data is None:
            self.preprocess_data()

        df = self.processed_data

        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "date_range": f"{df['연도'].min()}-{df['연도'].max()}",
            "unique_countries": df["국적"].nunique(),
            "unique_purposes": df["목적"].nunique(),
            "target_stats": {
                "mean": df[TARGET_COLUMN].mean(),
                "std": df[TARGET_COLUMN].std(),
                "min": df[TARGET_COLUMN].min(),
                "max": df[TARGET_COLUMN].max(),
                "median": df[TARGET_COLUMN].median(),
            },
            "missing_values": df.isnull().sum().to_dict(),
            "covid_period_ratio": df["코로나기간"].mean(),
        }

        return summary

    def print_data_info(self):
        """데이터 정보 출력"""
        summary = self.get_data_summary()

        print(f"\n📈 **데이터 요약 정보**")
        print(f"- 총 행수: {summary['total_rows']:,}")
        print(f"- 총 컬럼수: {summary['total_columns']}")
        print(f"- 날짜 범위: {summary['date_range']}")
        print(f"- 국적 수: {summary['unique_countries']}")
        print(f"- 목적 수: {summary['unique_purposes']}")
        print(f"- 코로나 기간 비율: {summary['covid_period_ratio']:.2%}")

        print(f"\n📊 **타겟 변수 ({TARGET_COLUMN}) 통계**")
        stats = summary["target_stats"]
        print(f"- 평균: {stats['mean']:.2f}")
        print(f"- 표준편차: {stats['std']:.2f}")
        print(f"- 최소값: {stats['min']:.2f}")
        print(f"- 최대값: {stats['max']:.2f}")
        print(f"- 중간값: {stats['median']:.2f}")


# 실행 예시
if __name__ == "__main__":
    # 데이터 로더 생성
    loader = DataLoader()

    # 데이터 로드 및 전처리
    data = loader.preprocess_data()

    # 데이터 정보 출력
    loader.print_data_info()

    # LSTM 데이터셋 준비
    X_train, y_train, X_val, y_val, X_test, y_test = loader.prepare_lstm_dataset()

    logger.info("✅ 데이터 로더 테스트 완료")
