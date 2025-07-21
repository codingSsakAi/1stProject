
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model # Functional API를 위해 Model 임포트

# XGBoost 임포트 (2단계 개선)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost 사용 가능: 트리 기반 모델 활성화")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost가 설치되어 있지 않습니다.")
    print("   pip install xgboost 명령으로 설치하면 성능이 더욱 향상됩니다!")
    print("   현재는 딥러닝 모델만 사용하여 앙상블을 구성합니다.")

class ModelBuilder:
    """데이터 특성에 맞춰 동적으로 LSTM 모델 구조를 생성합니다."""

    def __init__(self, config_module):
        self.config = config_module

    def build(self, input_shape, data_size, purpose):
        """데이터 크기와 목적에 따라 적응형 모델을 구축합니다."""
        if purpose == "관광":
            return self._build_tourism_lstm_attention_model(input_shape, data_size)
        else:
            return self._build_standard_model(input_shape, data_size)

    def _build_standard_model(self, input_shape, data_size):
        """표준 목적을 위한 모델을 구축합니다."""
        if data_size < 100:
            model = self._build_simple_dense(input_shape)
        elif data_size < 150:
            model = self._build_single_gru(input_shape)  # GRU 모델 추가
        elif data_size < 200:
            model = self._build_single_lstm(input_shape)
        else:
            model = self._build_multi_lstm(input_shape)
        
        learning_rate = 0.005 if data_size < 200 else 0.001
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def _build_tourism_lstm_attention_model(self, input_shape, data_size):
        """관광 목적을 위한 LSTM-Attention 모델을 구축합니다."""
        # LSTM으로 시퀀스 특징을 학습하고, Attention으로 중요한 시간 단계에 집중합니다.
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(64, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", dropout=0.3)(inputs)
        batch_norm_lstm = BatchNormalization()(lstm_out)
        dropout_lstm = Dropout(0.3)(batch_norm_lstm)

        # Attention Mechanism
        attention_output = Attention()([dropout_lstm, dropout_lstm])
        
        # Attention 출력 후 GlobalAveragePooling1D를 추가하여 시퀀스 차원을 제거
        pooled_attention_output = GlobalAveragePooling1D()(attention_output)
        
        dense_out = Dense(32, activation="relu")(pooled_attention_output)
        predictions = Dense(1, activation="linear", dtype="float32")(dense_out)

        model = Model(inputs=inputs, outputs=predictions)

        if data_size < 100:
            learning_rate = self.config.TOURISM_LEARNING_RATE_SMALL_DATA
        elif data_size < 200:
            learning_rate = self.config.TOURISM_LEARNING_RATE_MEDIUM_DATA
        else:
            learning_rate = self.config.TOURISM_LEARNING_RATE_LARGE_DATA

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        print(f"모델 구축: 관광 목적 특화 LSTM-Attention (학습률: {learning_rate})")
        return model

    def _build_simple_dense(self, input_shape):
        """소규모 데이터셋을 위한 간단한 Dense 네트워크"""
        # input_shape 안전 처리 (오류 방지)
        if isinstance(input_shape, tuple):
            if len(input_shape) == 1:
                input_dim = input_shape[0]
            else:
                # 다차원 튜플인 경우 첫 번째 차원 사용
                input_dim = input_shape[0]
        else:
            # 정수인 경우
            input_dim = input_shape
        
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="linear", dtype="float32"),
        ])
        print(f"모델 구축: Dense 네트워크 (입력 차원: {input_dim})")
        return model

    def _build_single_lstm(self, input_shape):
        """단일 LSTM 레이어 모델"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(32, activation="tanh", recurrent_activation="sigmoid", dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="linear", dtype="float32"),
        ])
        print(f"모델 구축: 단일 LSTM")
        return model

    def _build_multi_lstm(self, input_shape):
        """다층 LSTM 레이어 모델"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32, activation="tanh", recurrent_activation="sigmoid", dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            Dense(24, activation="relu"),
            Dense(1, activation="linear", dtype="float32"),
        ])
        print(f"모델 구축: 다층 LSTM")
        return model

    def _build_single_gru(self, input_shape):
        """단일 GRU 레이어 모델 (빠른 학습과 적은 메모리 사용)"""
        model = Sequential([
            Input(shape=input_shape),
            GRU(32, activation="tanh", recurrent_activation="sigmoid", dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="linear", dtype="float32"),
        ])
        print(f"모델 구축: 단일 GRU (빠른 학습)")
        return model

    def _build_multi_gru(self, input_shape):
        """다층 GRU 레이어 모델 (고성능 GRU)"""
        model = Sequential([
            Input(shape=input_shape),
            GRU(64, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            GRU(32, activation="tanh", recurrent_activation="sigmoid", dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            Dense(24, activation="relu"),
            Dense(1, activation="linear", dtype="float32"),
        ])
        print(f"모델 구축: 다층 GRU (고성능)")
        return model

    # --- XGBoost 모델 빌더 (2단계 개선) ---
    
    def build_xgboost(self, purpose, feature_count):
        """
        XGBoost 회귀 모델을 구축합니다.
        시계열 특성을 고려한 트리 기반 모델로, 딥러닝과 다른 관점의 패턴을 학습합니다.
        """
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost가 설치되어 있지 않아 XGBoost 모델을 생성할 수 없습니다.")
            return None
            
        # 목적별 하이퍼파라미터 가져오기
        params = self.config.XGBOOST_PARAMS.get(purpose, self.config.XGBOOST_PARAMS["관광"])
        
        # XGBoost 회귀 모델 생성 (경고 수정: early_stopping_rounds 생성자 추가)
        model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"], 
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=params["random_state"],
            n_jobs=params["n_jobs"],
            # 시계열 특성에 맞는 추가 설정
            objective='reg:squarederror',  # 회귀 문제
            eval_metric='rmse',            # 평가 지표
            early_stopping_rounds=20,     # 조기 종료 (생성자에서 설정)
            verbosity=0                    # 출력 최소화
        )
        
        print(f"모델 구축: XGBoost (목적: {purpose}, 특성 수: {feature_count})")
        print(f"  - 트리 개수: {params['n_estimators']}")
        print(f"  - 최대 깊이: {params['max_depth']}")
        print(f"  - 학습률: {params['learning_rate']}")
        
        return model
        
    def is_xgboost_model(self, model):
        """
        주어진 모델이 XGBoost 모델인지 확인합니다.
        """
        return XGBOOST_AVAILABLE and isinstance(model, xgb.XGBRegressor)

    # --- 스마트 모델 선택 로직 (3단계 개선) ---
    
    def build_smart_model(self, input_shape, data_characteristics, data_handler):
        """
        데이터 특성에 기반하여 최적의 모델을 자동으로 선택하고 구축합니다.
        """
        if not self.config.ENABLE_SMART_OPTIMIZATION:
            # 스마트 최적화가 비활성화된 경우 기존 방식 사용
            return self.build(input_shape, data_characteristics["size"], data_characteristics["purpose"])
        
        print(f"\n🧠 스마트 모델 선택 시작...")
        
        # 데이터 특성에 기반한 최적 모델 목록 가져오기
        optimal_models = data_handler.get_optimal_models_for_characteristics(data_characteristics)
        
        # 첫 번째 권장 모델을 주 모델로 선택
        primary_model_type = optimal_models[0] if optimal_models else "LSTM"
        
        print(f"  🎯 선택된 주 모델: {primary_model_type}")
        
        # 모델 구축
        model = self._build_model_by_type(primary_model_type, input_shape, data_characteristics)
        
        if model is None:
            print(f"  ⚠️ {primary_model_type} 모델 구축 실패, LSTM으로 폴백")
            model = self._build_single_lstm(input_shape)
            primary_model_type = "LSTM"
        
        # 모델 컴파일 (오류 수정: XGBoost 제외)
        if not self.is_xgboost_model(model):
            # TensorFlow 모델만 컴파일 필요
            data_size = data_characteristics["size"]
            
            # 데이터 크기에 따른 학습률 결정
            if data_size < 100:
                learning_rate = 0.01
            elif data_size < 200:
                learning_rate = 0.005
            else:
                learning_rate = 0.001
            
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
            print(f"  ✅ 모델 컴파일 완료 (학습률: {learning_rate})")
        else:
            print(f"  ✅ XGBoost 모델 (컴파일 불필요)")
        
        return model
    
    def _build_model_by_type(self, model_type, input_shape, data_characteristics):
        """
        모델 타입에 따라 적절한 모델을 구축합니다.
        """
        purpose = data_characteristics["purpose"]
        data_size = data_characteristics["size"]
        
        try:
            if model_type == "LSTM_ATTENTION":
                return self._build_tourism_lstm_attention_model(input_shape, data_size)
            elif model_type == "LSTM":
                # 데이터 크기에 따라 단일/다층 LSTM 선택
                if data_size > 200:
                    return self._build_multi_lstm(input_shape)
                else:
                    return self._build_single_lstm(input_shape)
            elif model_type == "GRU":
                # 데이터 크기에 따라 단일/다층 GRU 선택
                if data_size > 200:
                    return self._build_multi_gru(input_shape)
                else:
                    return self._build_single_gru(input_shape)
            elif model_type == "DENSE":
                return self._build_simple_dense(input_shape)
            elif model_type == "XGBOOST":
                # XGBoost는 특성 수 필요
                feature_count = input_shape[0] if len(input_shape) > 0 else 10
                return self.build_xgboost(purpose, feature_count)
            else:
                print(f"  ⚠️ 알 수 없는 모델 타입: {model_type}")
                return None
                
        except Exception as e:
            print(f"  ❌ {model_type} 모델 구축 중 오류: {str(e)}")
            return None
    
    def get_optimized_ensemble_weights(self, data_characteristics):
        """
        데이터 특성에 기반하여 최적화된 앙상블 가중치를 계산합니다.
        """
        if not self.config.ENABLE_SMART_OPTIMIZATION:
            # 기본 앙상블 가중치 반환
            purpose = data_characteristics["purpose"]
            return self.config.ENSEMBLE_MODELS.get(purpose, self.config.ENSEMBLE_MODELS["관광"])
        
        print(f"\n⚖️ 앙상블 가중치 최적화 중...")
        
        # 데이터 특성 기반 가중치 조정
        base_weights = self.config.ENSEMBLE_MODELS.get(
            data_characteristics["purpose"], 
            self.config.ENSEMBLE_MODELS["관광"]
        ).copy()
        
        # 변동성이 높은 경우 XGBoost 비중 증가
        if data_characteristics["volatility"] > self.config.DATA_ANALYSIS_THRESHOLDS["high_volatility"]:
            if "XGBOOST" in base_weights:
                base_weights["XGBOOST"] = min(base_weights["XGBOOST"] * 1.3, 0.5)
                self._normalize_weights(base_weights)
        
        # 계절성이 강한 경우 LSTM_ATTENTION 비중 증가
        if data_characteristics["seasonality"] > self.config.DATA_ANALYSIS_THRESHOLDS["strong_seasonality"]:
            if "LSTM_ATTENTION" in base_weights:
                base_weights["LSTM_ATTENTION"] = min(base_weights["LSTM_ATTENTION"] * 1.2, 0.5)
                self._normalize_weights(base_weights)
        
        # 데이터가 작은 경우 복잡한 모델 비중 감소
        if data_characteristics["size_category"] == "small":
            for model in ["LSTM_ATTENTION", "LSTM"]:
                if model in base_weights:
                    base_weights[model] *= 0.8
            # DENSE와 GRU 비중 증가
            for model in ["DENSE", "GRU"]:
                if model in base_weights:
                    base_weights[model] *= 1.2
            self._normalize_weights(base_weights)
        
        print(f"  📊 최적화된 가중치: {base_weights}")
        return base_weights
    
    def _normalize_weights(self, weights):
        """
        가중치를 정규화하여 합이 1이 되도록 합니다.
        """
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] = weights[key] / total
