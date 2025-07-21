
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class Trainer:
    """모델 학습, 평가, 로깅을 담당합니다."""

    def __init__(self, model, scaler, config_module):
        self.model = model
        self.scaler = scaler
        self.config = config_module
        self.X_val = None
        self.y_val = None

    def train(self, X, y, purpose):
        """데이터를 분할하고 모델을 학습합니다."""
        # 시계열 데이터는 순서가 중요하므로 shuffle=False 옵션으로 분할
        X_train, self.X_val, y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        if len(X_train) == 0:
            print("학습 데이터가 없어 훈련을 건너뜁니다.")
            return None

        epochs, batch_size = self._get_training_params(len(X_train), purpose)
        self.callbacks = self._get_callbacks()

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val) if len(self.X_val) > 0 else None,
            callbacks=self.callbacks,
            verbose=1
        )
        return history

    def evaluate(self, purpose):
        """학습된 모델을 평가합니다."""
        if self.X_val is None or len(self.X_val) == 0:
            return {}

        y_pred_scaled = self.model.predict(self.X_val, verbose=2).flatten()
        y_true_rescaled, y_pred_rescaled = self._inverse_transform(self.y_val, y_pred_scaled)

        thresholds = self._get_performance_thresholds(len(self.X_val))
        
        metrics = self._calculate_metrics(y_true_rescaled, y_pred_rescaled, purpose, thresholds)
        metrics['training_samples'] = len(self.X_val) + len(self.y_val) # A rough estimation
        metrics['validation_samples'] = len(self.X_val)
        return metrics

    def _get_training_params(self, data_size, purpose):
        """데이터 크기와 목적에 따라 학습 파라미터를 반환합니다."""
        if purpose == "관광":
            epochs = self.config.TOURISM_LSTM_EPOCHS_LARGE_DATA if data_size >= 200 else self.config.TOURISM_LSTM_EPOCHS_SMALL_DATA
            batch_size = self.config.TOURISM_LSTM_BATCH_SIZE_LARGE_DATA if data_size >= 200 else self.config.TOURISM_LSTM_BATCH_SIZE_SMALL_DATA
        else:
            epochs = self.config.LSTM_EPOCHS_LARGE_DATA if data_size >= 200 else self.config.LSTM_EPOCHS_SMALL_DATA
            batch_size = self.config.LSTM_BATCH_SIZE
        return epochs, batch_size

    def _get_callbacks(self):
        """학습에 사용할 콜백을 반환합니다."""
        return [
            EarlyStopping(
                monitor=self.config.EARLY_STOPPING_MONITOR, 
                patience=self.config.EARLY_STOPPING_PATIENCE, 
                restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor=self.config.EARLY_STOPPING_MONITOR, 
                factor=self.config.REDUCE_LR_FACTOR, 
                patience=self.config.REDUCE_LR_PATIENCE, 
                min_lr=self.config.REDUCE_LR_MIN_LR, verbose=1
            )
        ]

    def _inverse_transform(self, y_true_scaled, y_pred_scaled):
        """스케일링된 값을 원래 값으로 변환합니다."""
        n_features = self.scaler.n_features_in_
        dummy_true = np.zeros((len(y_true_scaled), n_features))
        dummy_true[:, 0] = y_true_scaled
        y_true_rescaled = self.scaler.inverse_transform(dummy_true)[:, 0]

        dummy_pred = np.zeros((len(y_pred_scaled), n_features))
        dummy_pred[:, 0] = y_pred_scaled
        y_pred_rescaled = self.scaler.inverse_transform(dummy_pred)[:, 0]

        # 로그 변환된 값을 원래 스케일로 복원
        y_true_actual = np.expm1(y_true_rescaled)
        y_pred_actual = np.expm1(y_pred_rescaled)

        return np.maximum(y_true_actual, 0), np.maximum(y_pred_actual, 0)

    def _get_performance_thresholds(self, data_size):
        """데이터 크기에 따라 성능 기준을 반환합니다."""
        if data_size < 100:
            return {k: v * 3 for k, v in self.config.BASE_PERFORMANCE_THRESHOLDS.items()}
        elif data_size < 200:
            return {k: v * 2 for k, v in self.config.BASE_PERFORMANCE_THRESHOLDS.items()}
        else:
            return self.config.BASE_PERFORMANCE_THRESHOLDS

    def _calculate_metrics(self, y_true, y_pred, purpose, thresholds):
        """성능 메트릭을 계산합니다."""
        metrics = {
            "purpose": purpose,
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2_score": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100,
            "f1_score": 0.0, # F1_score는 현재 계산되지 않으므로 임시값 설정
            **thresholds
        }

        # 등급 계산 (기준값과 비교)
        metrics["mae_등급"] = "양호" if metrics["mae"] <= metrics["mae_기준값"] else "미흡"
        metrics["rmse_등급"] = "양호" if metrics["rmse"] <= metrics["rmse_기준값"] else "미흡"
        metrics["r2_score_등급"] = "양호" if metrics["r2_score"] >= metrics["r2_score_기준값"] else "미흡"
        metrics["mape_등급"] = "양호" if metrics["mape"] <= metrics["mape_기준값"] else "미흡"
        metrics["f1_score_등급"] = "양호" if metrics["f1_score"] >= metrics["f1_score_기준값"] else "미흡" # F1_score는 현재 0.0으로 설정되어 있으므로 실제 계산 필요

        return metrics
