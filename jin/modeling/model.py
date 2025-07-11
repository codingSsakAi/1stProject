import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(input_shape, output_size, hidden_size=16):
    """
    LSTM 기반 목적별 입국자수 예측 모델 생성 함수 (TensorFlow/Keras)
    Args:
        input_shape (tuple): (시퀀스길이, 특성수)
        output_size (int): 출력 차원 (예: 목적 개수)
        hidden_size (int): LSTM 은닉 상태 크기
    Returns:
        keras.Model: 생성된 모델
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(hidden_size)(inputs)
    outputs = layers.Dense(output_size)(x)
    model = keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    # 임의의 입력 데이터 (배치=2, 시퀀스길이=6, 특성수=4)
    import numpy as np

    batch_size = 2
    seq_len = 6
    input_size = 4
    output_size = 3
    x = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
    model = build_lstm_model((seq_len, input_size), output_size)
    y_pred = model(x)
    print("입력 shape:", x.shape)
    print("예측 결과 shape:", y_pred.shape)
    print("예측 결과:")
    print(y_pred.numpy())
