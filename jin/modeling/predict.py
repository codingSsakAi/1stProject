import numpy as np
from tensorflow import keras


def predict(model, input_seq):
    """
    학습된 모델로 입력 시퀀스에 대해 예측값을 반환하는 함수 (TensorFlow/Keras)
    Args:
        model: 학습된 keras.Model
        input_seq (ndarray): (배치, 시퀀스길이, 특성수) 입력
    Returns:
        ndarray: 예측 결과 (배치, 출력차원)
    """
    y_pred = model.predict(input_seq)
    return y_pred


if __name__ == "__main__":
    # 임의의 입력 데이터 생성 (배치=2, 시퀀스길이=6, 특성수=4)
    batch_size = 2
    seq_len = 6
    input_size = 4
    output_size = 3
    x = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
    # 저장된 모델 불러오기
    try:
        model = keras.models.load_model("lstm_visitor_model.h5")
        print("저장된 모델을 불러왔습니다.")
    except Exception as e:
        print("모델을 불러오지 못했습니다. (임의 파라미터 사용)", e)
        from model import build_lstm_model

        model = build_lstm_model((seq_len, input_size), output_size)
    # 예측
    y_pred = predict(model, x)
    print("입력 shape:", x.shape)
    print("예측 결과 shape:", y_pred.shape)
    print("예측 결과:")
    print(y_pred)
