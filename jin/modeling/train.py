import numpy as np
from model import build_lstm_model
import tensorflow as tf
from tensorflow import keras


def train_model(train_X, train_y, input_shape, output_size, hidden_size=16, epochs=10, lr=0.001):
    """
    TensorFlow(Keras) 기반 LSTM 모델을 학습하는 함수
    Args:
        train_X (ndarray): 입력 시퀀스 (배치, 시퀀스길이, 특성수)
        train_y (ndarray): 타깃값 (배치, 출력차원)
        input_shape (tuple): (시퀀스길이, 특성수)
        output_size (int): 출력 차원
        hidden_size (int): LSTM 은닉 크기
        epochs (int): 학습 에폭 수
        lr (float): 학습률
    Returns:
        model: 학습된 모델
    """
    # 모델 생성
    model = build_lstm_model(input_shape, output_size, hidden_size)
    # 모델 컴파일 (회귀 문제이므로 MSE 손실 사용)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    # 모델 학습
    model.fit(train_X, train_y, epochs=epochs, batch_size=16, verbose=1)
    return model


if __name__ == "__main__":
    # 임의의 학습 데이터 생성 (배치=32, 시퀀스길이=6, 특성수=4, 출력차원=3)
    num_samples = 32
    seq_len = 6
    input_size = 4
    output_size = 3
    X = np.random.randn(num_samples, seq_len, input_size).astype(np.float32)
    y = np.random.randn(num_samples, output_size).astype(np.float32)
    # 모델 학습
    model = train_model(X, y, (seq_len, input_size), output_size, hidden_size=16, epochs=5)
    # 학습된 모델 저장 (HDF5 포맷)
    model.save("lstm_visitor_model.h5")
    print("모델이 'lstm_visitor_model.h5'로 저장되었습니다.")
