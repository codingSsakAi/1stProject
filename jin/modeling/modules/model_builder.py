
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model # Functional API를 위해 Model 임포트

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
        model = Sequential([
            Input(shape=(input_shape[1],)),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="linear", dtype="float32"),
        ])
        print(f"��델 구축: Dense 네트워크")
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
