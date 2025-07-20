
import numpy as np
import pandas as pd
import tensorflow as tf

class Predictor:
    """학습된 모델을 사용하여 미래 값을 예측합니다."""

    def __init__(self, model, scaler, features_data, combo_data, config_module):
        self.model = model
        self.scaler = scaler
        self.features_data = features_data
        self.combo_data = combo_data
        self.config = config_module

    def predict_future(self, target_months, purpose):
        """미래 월별 입국자 수를 예측합니다."""
        sequence_length = self._get_sequence_length(len(self.features_data))
        current_sequence = self.features_data.tail(sequence_length).values
        current_sequence = self.scaler.transform(current_sequence)

        predictions_with_log = []
        # 마지막 실제 값은 feature 데이터에서 가져와야 하며, 이는 이미 로그 변환된 값입니다.
        last_actual_log_value = self.features_data["입국자수"].iloc[-1]

        for i, target_month in enumerate(target_months):
            pred_scaled = self.model.predict(current_sequence.reshape(1, sequence_length, -1), verbose=2)[0, 0]
            
            # 역스케일링하여 로그 스케일의 예측값 획득
            pred_log_value = self._inverse_scale_single(pred_scaled)
            
            # 연속성 적용 (로그 스케일에서)
            if i == 0:
                pred_log_value = (pred_log_value * 0.5) + (last_actual_log_value * 0.5)
            else:
                # 이전 예측의 로그 값을 사용
                prev_log_value = np.log1p(predictions_with_log[-1]["value"])
                pred_log_value = (pred_log_value * 0.85) + (prev_log_value * 0.15)

            # 원래 스케일로 변환
            pred_value = np.expm1(pred_log_value)
            pred_value *= (1 + np.random.normal(0, 0.05)) # 변동성 추가
            pred_value = max(1, int(pred_value))

            # 리포팅을 위해 실제 값과 로그 값 모두 저장
            predictions_with_log.append({"month": target_month, "value": pred_value, "log_value": pred_log_value, "type": "predicted"})

            # 시퀀스 업데이트 (로그 스케일 값 사용)
            new_row = self._create_new_feature_row(pred_log_value, target_month)
            new_row_scaled = self.scaler.transform(new_row.reshape(1, -1))
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row_scaled[0]

        # 최종 결과에서는 log_value 필드 제거하여 기존 형식 유지
        final_predictions = [{"month": p["month"], "value": p["value"], "type": p["type"]} for p in predictions_with_log]
        return final_predictions

    def _get_sequence_length(self, data_size):
        """데이터 크기에 따라 시퀀스 길이를 반환합니다."""
        if data_size < 100:
            return self.config.LSTM_SEQUENCE_LENGTH_SMALL_DATA
        return self.config.LSTM_SEQUENCE_LENGTH_LARGE_DATA

    def _inverse_scale_single(self, scaled_value):
        """단일 값을 역스케일링합니다."""
        # MinMaxScaler의 inverse_transform은 2D 배열을 기대합니다.
        # 스케일러가 학습된 feature의 개수를 알아야 합니다.
        n_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 1
        dummy_array = np.zeros((1, n_features))
        dummy_array[0, 0] = scaled_value
        return self.scaler.inverse_transform(dummy_array)[0, 0]

    def _create_new_feature_row(self, pred_log_value, target_month):
        """예측을 위한 새로운 특성 행을 생성합니다."""
        # 이 부분은 실제 특성 생성 로직에 따라 더 정교하게 구현되어야 합니다.
        # 여기서는 간단한 예시로 마지막 행을 복사하여 사용합니다.
        new_row = self.features_data.iloc[-1].copy()
        new_row["입국자수"] = pred_log_value # 이미 역스케일링된 로그 값입니다.
        target_date = pd.to_datetime(target_month + "-01")
        new_row["연도"] = target_date.year
        new_row["월"] = target_date.month
        # ... 기타 특성 업데이트 ...
        return new_row.values
