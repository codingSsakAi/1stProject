
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    """데이터 특성에 맞춰 동적으로 LSTM 모델 구조를 생성합니다."""

    def __init__(self, config_module):
        self.config = config_module

    def build(self, input_shape, data_size, purpose):
        """데이터 크기와 목적에 따라 적응형 모델을 구축합니다."""
        if purpose == "관광":
            return self._build_tourism_model(input_shape, data_size)
        else:
            return self._build_standard_model(input_shape, data_size)

    def _build_standard_model(self, input_shape, data_size):
        """표준 목적을 위한 모델을 구축합니다."""
        if data_size < 100:
            model = self._build_simple_dense(input_shape)
        elif data_size < 200:
            model = self._build_single_lstm(input_shape)
        else:
            model = self._build_multi_lstm(input_shape)
        
        learning_rate = 0.005 if data_size < 200 else 0.001
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def _build_tourism_model(self, input_shape, data_size):
        """관광 목적을 위한 특화 모델을 구축합니다."""
        # 관광 모델은 더 복잡한 구조를 가질 수 있습니다.
        # 여기서는 표준 모델과 동일한 구조를 사용하지만, 
        # 필요에 따라 더 복잡한 모델로 확장할 수 있습니다.
        return self._build_standard_model(input_shape, data_size)

    def _build_simple_dense(self, input_shape):
        """소규모 데이터셋을 위한 간단한 Dense 네트워크"""
        model = Sequential([
            Input(shape=(input_shape[1],)),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="linear", dtype="float32"),
        ])
        print(f"모델 구축: Dense 네트워크")
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
