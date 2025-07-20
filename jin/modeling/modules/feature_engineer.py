
import pandas as pd
import numpy as np

class FeatureEngineer:
    """시계열 데이터로부터 모델 학습에 사용할 특성을 생성합니다."""

    def create_features(self, data):
        """데이터로부터 다양한 특성을 생성합니다."""
        processed_data = data.copy()
        processed_data = self._create_cyclical_features(processed_data)
        processed_data = self._create_lag_features(processed_data, [1, 3, 6, 12])
        processed_data = self._create_moving_average_features(processed_data, [3, 6, 12])
        processed_data = self._create_volatility_features(processed_data, [3, 6])
        processed_data = self._create_momentum_features(processed_data, [1, 3, 6])
        processed_data = self._create_interaction_features(processed_data)
        
        core_features = self._get_core_features()
        available_features = [col for col in core_features if col in processed_data.columns]
        features_data = processed_data[available_features].copy()
        
        features_data = features_data.ffill().fillna(0)
        features_data.replace([np.inf, -np.inf], 0, inplace=True)
        features_data["입국자수"] = np.clip(features_data["입국자수"], 0, None)
        
        return features_data

    def _create_cyclical_features(self, df):
        """주기적 특성을 생성합니다."""
        df["월_sin"] = np.sin(2 * np.pi * df["월"] / 12)
        df["월_cos"] = np.cos(2 * np.pi * df["월"] / 12)
        for i in range(1, 5):
            df[f"분기_{i}"] = (df["분기"] == i).astype(int)
            df[f"계절_{i}"] = (df["계절"] == i).astype(int)
        return df

    def _create_lag_features(self, df, lags):
        """지연 특성을 생성합니다."""
        for lag in lags:
            df[f"lag_{lag}"] = df["입국자수"].shift(lag)
        return df

    def _create_moving_average_features(self, df, windows):
        """이동 평균 특성을 생성합니다."""
        for window in windows:
            ma_col = f"ma_{window}"
            df[ma_col] = df["입국자수"].rolling(window, min_periods=1).mean()
            df[f"ma_ratio_{window}"] = df["입국자수"] / df[ma_col]
        return df

    def _create_volatility_features(self, df, windows):
        """변동성 특성을 생성합니다."""
        for window in windows:
            df[f"volatility_{window}"] = df["입국자수"].rolling(window, min_periods=1).std()
            df[f"cv_{window}"] = df[f"volatility_{window}"] / df[f"ma_{window}"]
        return df

    def _create_momentum_features(self, df, periods):
        """모멘텀 특성을 생성합니다."""
        for period in periods:
            df[f"momentum_{period}"] = df["입국자수"].pct_change(period)
            df[f"diff_{period}"] = df["입국자수"].diff(period)
        return df

    def _create_interaction_features(self, df):
        """상호작용 특성을 생성합니다."""
        df["월_x_입국자수"] = df["월"] * df["입국자수"]
        df["계절_x_입국자수"] = df["계절"] * df["입국자수"]
        return df

    def _get_core_features(self):
        """핵심 특성 목록을 반환합니다."""
        return [
            "입국자수", "연도", "월", "분기", "계절", "코로나기간",
            "월_sin", "월_cos", "분기_1", "분기_2", "분기_3", "분기_4",
            "계절_1", "계절_2", "계절_3", "계절_4",
            "lag_1", "lag_3", "lag_6", "ma_3", "ma_6", "ma_12",
            "ma_ratio_3", "ma_ratio_6", "volatility_3", "volatility_6",
            "cv_3", "cv_6", "momentum_1", "momentum_3", "diff_1", "diff_3",
            "월_x_입국자수", "계절_x_입국자수"
        ]
