
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    """데이터 로딩, 전처리, 증강을 담당하는 클래스"""

    def __init__(self, config_module):
        self.config = config_module
        self.data_path = self.config.DATA_PATH
        self.covid_strategy = self.config.DEFAULT_COVID_STRATEGY
        self.data = None
        self.country_mapping = {}
        self.load_data()

    def load_data(self):
        """데이터를 로드하고 기본 전처리를 수행합니다."""
        print("데이터 로드 중...")
        try:
            self.data = pd.read_csv(self.data_path, encoding="utf-8")
        except UnicodeDecodeError:
            self.data = pd.read_csv(self.data_path, encoding="cp949")
        self.data["날짜"] = pd.to_datetime(
            self.data["연도"].astype(str) + "-" + self.data["월"].astype(str).str.zfill(2) + "-01"
        )
        season_map = {"봄": 1, "여름": 2, "가을": 3, "겨울": 4}
        self.data["계절"] = self.data["계절"].map(season_map)
        self._initialize_country_mapping()
        self.config.AVAILABLE_NATIONALITIES = sorted(self.data["국적"].unique().tolist())
        print("데이터 로드 및 기본 전처리 완료")

    def _initialize_country_mapping(self):
        """국가 매핑을 초기화합니다."""
        unique_countries = self.data["국적"].unique()
        self.country_mapping = {country: i for i, country in enumerate(unique_countries, 1)}

    def get_data_for_purpose(self, nationality, purpose, covid_strategy):
        """특정 국적과 목적에 맞는 데이터를 준비합니다."""
        df = self.data[(self.data["국적"] == nationality) & (self.data["목적"] == purpose)].copy()
        df = df.sort_values("날짜").reset_index(drop=True)
        df = self._apply_covid_strategy(df, covid_strategy)
        
        # 극단값(outlier) 문제 완화를 위해 로그 변환 적용
        df["입국자수"] = np.log1p(df["입국자수"])
        
        df = self._clean_data(df)
        return df

    def _apply_covid_strategy(self, df, covid_strategy):
        """코로나 전략에 따라 데이터를 처리합니다."""
        if '코로나기간' not in df.columns:
            return df
        if covid_strategy == "exclude":
            return df[df["코로나기간"] == 0].copy()
        elif covid_strategy == "weighted":
            df["sample_weight"] = np.where(df["코로나기간"] == 1, 0.1, 1.0)
        else: # include
            df["sample_weight"] = 1.0
        return df

    def _clean_data(self, df):
        """데이터의 결측치와 이상치를 처리합니다."""
        if df.empty:
            return df
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].interpolate(method='linear').fillna(0)
        if '전년동월대비증감률' in df.columns:
            df['전년동월대비증감률'] = df['전년동월대비증감률'].clip(-100, 100).fillna(0)
        return df

    def _add_noise(self, df, noise_level):
        """데이터에 노이즈를 추가하여 증강합니다."""
        df_augmented = df.copy()
        noise = np.random.normal(0, noise_level * df_augmented["입국자수"].std(), df_augmented.shape[0])
        df_augmented["입국자수"] = np.maximum(0, df_augmented["입국자수"] + noise)
        return df_augmented

    def _add_trend(self, df, trend_factor):
        """데이터에 트렌드를 추가하여 증강합니다."""
        df_augmented = df.copy()
        trend = np.arange(len(df_augmented)) * trend_factor
        df_augmented["입국자수"] = np.maximum(0, df_augmented["입국자수"] * (1 + trend / 100))
        return df_augmented

    def _add_seasonal_boost(self, df, boost_factor):
        """데이터에 계절성 부스트를 추가하여 증강합니다."""
        df_augmented = df.copy()
        # 월별 계절성을 고려하여 부스트 적용 (예: 여름/겨울에 더 큰 부스트)
        # 여기서는 단순화를 위해 일괄 적용
        df_augmented["입국자수"] = np.maximum(0, df_augmented["입국자수"] * boost_factor)
        return df_augmented

    def augment_data(self, data):
        """데이터가 부족할 경우 증강합니다."""
        if len(data) >= self.config.AUGMENTATION_TARGET_MONTHS:
            return [data]

        print(f"데이터 증강 시작: {len(data)}개월 -> 목표 {self.config.AUGMENTATION_TARGET_MONTHS}개월")
        augmented_datasets = [data.copy()]
        current_length = len(data)

        while current_length < self.config.AUGMENTATION_TARGET_MONTHS:
            new_data = augmented_datasets[-1].copy() # 마지막으로 증강된 데이터셋 사용
            
            # 노이즈 증강
            if self.config.AUGMENTATION_NOISE_LEVELS:
                noise_level = np.random.choice(self.config.AUGMENTATION_NOISE_LEVELS)
                new_data = self._add_noise(new_data, noise_level)
            
            # 트렌드 증강
            if self.config.AUGMENTATION_TREND_FACTORS:
                trend_factor = np.random.choice(self.config.AUGMENTATION_TREND_FACTORS)
                new_data = self._add_trend(new_data, trend_factor)
            
            # 계절성 강화 증강
            if self.config.AUGMENTATION_SEASONAL_BOOSTS:
                seasonal_boost = np.random.choice(self.config.AUGMENTATION_SEASONAL_BOOSTS)
                new_data = self._add_seasonal_boost(new_data, seasonal_boost)
            
            augmented_datasets.append(new_data)
            current_length += len(new_data) # 증강된 데이터셋의 길이를 더함

            if len(augmented_datasets) > 100: # 무한 루프 방지
                print("경고: 데이터 증강이 너무 많이 반복되어 중단합니다. 목표 길이에 도달하지 못했습니다.")
                break

        # 목표 길이에 맞춰 데이터셋을 자르거나 병합
        final_augmented_data = pd.concat(augmented_datasets, ignore_index=True)
        final_augmented_data = final_augmented_data.drop_duplicates(subset=["날짜", "국적", "목적"]).sort_values("날짜").reset_index(drop=True)
        
        if len(final_augmented_data) > self.config.AUGMENTATION_TARGET_MONTHS:
            final_augmented_data = final_augmented_data.tail(self.config.AUGMENTATION_TARGET_MONTHS).reset_index(drop=True)

        print(f"데이터 증강 완료: {len(data)}개월 -> {len(final_augmented_data)}개월")
        return [final_augmented_data]

    def preprocess_for_model(self, features, purpose):
        """모델 학습을 위해 데이터를 스케일링하고 시퀀스를 생성합니다."""
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features.values)
        
        sequence_length = self._get_sequence_length(features, purpose)
        X, y = self._create_sequences(scaled_features, sequence_length)
        
        return X, y, scaler

    def _get_sequence_length(self, features, purpose):
        """목적과 데이터 크기에 따라 적절한 시퀀스 길이를 반환합니다."""
        is_tourism = '관광' in purpose
        is_large_data = len(features) > 200

        if is_tourism:
            return self.config.TOURISM_SEQUENCE_LENGTH
        
        return self.config.LSTM_SEQUENCE_LENGTH_LARGE_DATA if is_large_data else self.config.LSTM_SEQUENCE_LENGTH_SMALL_DATA

    def _create_sequences(self, data, seq_length):
        """시계열 데이터로부터 시퀀스를 생성합니다."""
        xs, ys = [], []
        if len(data) <= seq_length:
            return np.array(xs), np.array(ys)
            
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(data[i + seq_length, 0]) # Target is the first column '입국자수'
        return np.array(xs), np.array(ys)
