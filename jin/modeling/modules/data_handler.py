
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
            # 데이터 특성에 맞는 동적 가중치 계산
            covid_weight = self._calculate_optimal_covid_weight(df)
            df["sample_weight"] = np.where(df["코로나기간"] == 1, covid_weight, 1.0)
        else: # include
            df["sample_weight"] = 1.0
        return df
    
    def _calculate_optimal_covid_weight(self, df):
        """데이터 특성에 맞는 최적 코로나 가중치를 계산합니다."""
        # 전체 데이터 대비 코로나 기간 비율
        covid_ratio = df["코로나기간"].mean()
        
        # 데이터 크기 (작을수록 코로나 데이터를 더 활용)
        data_size = len(df)
        
        # 기본 가중치 계산 (데이터가 적을수록 높은 가중치)
        if data_size < 50:
            base_weight = 0.3  # 데이터 부족 시 코로나 데이터도 활용
        elif data_size < 100:
            base_weight = 0.2  # 중간 크기
        else:
            base_weight = 0.1  # 데이터 충분 시 낮은 가중치
        
        # 코로나 비율에 따른 조정 (코로나 데이터가 많을수록 가중치 감소)
        if covid_ratio > 0.3:
            base_weight *= 0.7  # 코로나 데이터가 30% 이상이면 가중치 감소
        
        return max(0.05, min(0.4, base_weight))  # 5%~40% 범위로 제한

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

    def augment_data(self, data, purpose=None):
        """데이터가 부족할 경우 증강합니다."""
        # 목적별 차별화된 증강 목표 적용
        if purpose and hasattr(self.config, 'PURPOSE_SPECIFIC_AUGMENTATION') and purpose in self.config.PURPOSE_SPECIFIC_AUGMENTATION:
            target_months = self.config.PURPOSE_SPECIFIC_AUGMENTATION[purpose]
            print(f"[{purpose}] 목적별 증강 목표: {target_months}개월")
        else:
            target_months = self.config.AUGMENTATION_TARGET_MONTHS
            
        if len(data) >= target_months:
            return [data]

        print(f"데이터 증강 시작: {len(data)}개월 -> 목표 {target_months}개월")
        augmented_datasets = [data.copy()]
        current_length = len(data)

        while current_length < target_months:
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
        
        if len(final_augmented_data) > target_months:
            final_augmented_data = final_augmented_data.tail(target_months).reset_index(drop=True)

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
        # 목적별 차별화된 시퀀스 길이 적용 (성능 최적화)
        if hasattr(self.config, 'PURPOSE_SPECIFIC_SEQUENCE_LENGTH') and purpose in self.config.PURPOSE_SPECIFIC_SEQUENCE_LENGTH:
            optimal_length = self.config.PURPOSE_SPECIFIC_SEQUENCE_LENGTH[purpose]
            print(f"[{purpose}] 목적별 최적 시퀀스 길이: {optimal_length}")
            return optimal_length
        
        # 기존 로직 (fallback)
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

    # --- 데이터 특성 분석 함수들 (3단계 개선) ---
    
    def analyze_data_characteristics(self, data, purpose):
        """
        데이터의 특성을 분석하여 최적 모델 선택에 필요한 정보를 제공합니다.
        """
        if data.empty or len(data) < 12:
            return self._get_default_characteristics()
        
        print(f"\n📊 데이터 특성 분석 시작 (목적: {purpose})")
        
        # 기본 특성 계산
        data_size = len(data)
        values = np.expm1(data["입국자수"])  # 로그 변환 해제
        
        # 1. 변동성 분석 (변동계수)
        volatility = values.std() / max(values.mean(), 1)
        
        # 2. 계절성 분석 (월별 변동)
        seasonality = self._analyze_seasonality(data)
        
        # 3. 트렌드 안정성 분석
        trend_stability = self._analyze_trend_stability(data)
        
        # 4. 데이터 크기 분류
        if data_size < self.config.DATA_ANALYSIS_THRESHOLDS["small_data"]:
            size_category = "small"
        elif data_size < self.config.DATA_ANALYSIS_THRESHOLDS["medium_data"]:
            size_category = "medium"
        else:
            size_category = "large"
        
        # 5. 특성 기반 분류
        characteristics = self._classify_data_characteristics(
            size_category, volatility, seasonality, trend_stability
        )
        
        # 결과 출력
        print(f"  📏 데이터 크기: {data_size}개월 ({size_category})")
        print(f"  📈 변동성: {volatility:.3f} ({'높음' if volatility > self.config.DATA_ANALYSIS_THRESHOLDS['high_volatility'] else '보통'})")
        print(f"  🌀 계절성: {seasonality:.3f} ({'강함' if seasonality > self.config.DATA_ANALYSIS_THRESHOLDS['strong_seasonality'] else '약함'})")
        print(f"  📊 트렌드 안정성: {trend_stability:.3f}")
        print(f"  🎯 데이터 특성: {characteristics}")
        
        return {
            "size": data_size,
            "size_category": size_category,
            "volatility": volatility,
            "seasonality": seasonality,
            "trend_stability": trend_stability,
            "characteristics": characteristics,
            "purpose": purpose
        }
    
    def _analyze_seasonality(self, data):
        """
        계절성 강도를 분석합니다.
        """
        if len(data) < 24:  # 2년 미만의 데이터
            return 0.1  # 기본값
        
        # 월별 평균 계산
        monthly_means = []
        for month in range(1, 13):
            month_data = data[data["월"] == month]["입국자수"]
            if len(month_data) > 0:
                monthly_means.append(np.expm1(month_data.mean()))
            else:
                monthly_means.append(0)
        
        # 월별 변동계수로 계절성 측정
        if len(monthly_means) > 0 and np.mean(monthly_means) > 0:
            seasonality = np.std(monthly_means) / np.mean(monthly_means)
        else:
            seasonality = 0.1
        
        return min(seasonality, 1.0)  # 최대값 제한
    
    def _analyze_trend_stability(self, data):
        """
        트렌드의 안정성을 분석합니다.
        """
        if len(data) < 12:
            return 0.5  # 기본값
        
        values = np.expm1(data["입국자수"])
        
        # 6개월 단위 이동 평균의 변동성으로 트렌드 안정성 측정
        if len(values) >= 6:
            moving_avg = pd.Series(values).rolling(window=6, center=True).mean().dropna()
            if len(moving_avg) > 1:
                trend_stability = moving_avg.std() / max(moving_avg.mean(), 1)
            else:
                trend_stability = 0.5
        else:
            trend_stability = 0.5
        
        return min(trend_stability, 1.0)  # 최대값 제한
    
    def _classify_data_characteristics(self, size_category, volatility, seasonality, trend_stability):
        """
        데이터 특성을 종합하여 분류합니다.
        """
        thresholds = self.config.DATA_ANALYSIS_THRESHOLDS
        
        # 주요 특성 판별
        is_high_volatile = volatility > thresholds["high_volatility"]
        is_seasonal = seasonality > thresholds["strong_seasonality"]
        is_stable = trend_stability < thresholds["stable_trend"]
        
        # 특성 조합 결정
        if is_high_volatile:
            characteristic = f"{size_category}_high_volatile"
        elif is_seasonal:
            characteristic = f"{size_category}_seasonal"
        elif is_stable:
            characteristic = f"{size_category}_stable"
        else:
            # 기본값: 중간 특성
            characteristic = f"{size_category}_stable"
        
        return characteristic
    
    def _get_default_characteristics(self):
        """
        데이터가 부족할 때 기본 특성을 반환합니다.
        """
        return {
            "size": 50,
            "size_category": "small",
            "volatility": 0.2,
            "seasonality": 0.1,
            "trend_stability": 0.3,
            "characteristics": "small_stable",
            "purpose": "기타"
        }
    
    def get_optimal_models_for_characteristics(self, characteristics):
        """
        데이터 특성에 기반한 최적 모델 목록을 반환합니다.
        """
        optimal_models = self.config.OPTIMAL_MODEL_BY_CHARACTERISTICS.get(
            characteristics["characteristics"],
            ["LSTM", "GRU", "DENSE"]  # 기본값
        )
        
        print(f"  🎯 권장 모델: {optimal_models}")
        return optimal_models
