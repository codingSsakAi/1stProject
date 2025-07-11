# -*- coding: utf-8 -*-
"""
LSTM 모델 클래스
Author: Jin
Created: 2025-01-15
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import sys
import os
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt

# 프로젝트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.config import *
from scripts.utils import *
from scripts.data_loader import DataLoader

logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM 모델 클래스"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        초기화

        Args:
            config: 모델 설정 딕셔너리
        """
        self.config = config or LSTM_CONFIG
        self.model = None
        self.scaler = None
        self.history = None
        self.is_trained = False

        # TensorFlow 설정
        tf.random.set_seed(RANDOM_SEED)

        logger.info("🤖 LSTM 모델 초기화")
        logger.info(f"📋 설정: {self.config}")

    def build_model(self) -> Model:
        """LSTM 모델 구조 생성"""
        logger.info("🏗️ LSTM 모델 구조 생성 중...")

        # 입력 레이어
        inputs = Input(shape=(self.config["sequence_length"], self.config["n_features"]))

        # LSTM 레이어들
        x = inputs

        # 첫 번째 LSTM 레이어
        x = LSTM(
            self.config["lstm_units"][0],
            return_sequences=True,
            dropout=self.config["dropout_rate"],
            recurrent_dropout=self.config["dropout_rate"],
        )(x)
        x = BatchNormalization()(x)

        # 두 번째 LSTM 레이어
        if len(self.config["lstm_units"]) > 1:
            x = LSTM(
                self.config["lstm_units"][1],
                return_sequences=False,
                dropout=self.config["dropout_rate"],
                recurrent_dropout=self.config["dropout_rate"],
            )(x)
            x = BatchNormalization()(x)

        # Dense 레이어들
        for units in self.config["dense_units"]:
            x = Dense(units, activation="relu")(x)
            x = Dropout(self.config["dropout_rate"])(x)
            x = BatchNormalization()(x)

        # 출력 레이어
        outputs = Dense(1, activation="linear")(x)

        # 모델 생성
        self.model = Model(inputs=inputs, outputs=outputs)

        # 모델 컴파일
        self.model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss="mse",
            metrics=["mae", "mape"],
        )

        logger.info("✅ LSTM 모델 구조 생성 완료")
        self.model.summary()

        return self.model

    def prepare_data(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """데이터 준비 및 스케일링"""
        logger.info("📊 데이터 준비 중...")

        # LSTM 데이터셋 준비
        X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_lstm_dataset(
            sequence_length=self.config["sequence_length"]
        )

        # 특성 스케일링
        X_train_scaled, self.scaler, X_val_scaled, X_test_scaled = scale_features(
            X_train, X_val, X_test
        )

        logger.info(f"✅ 데이터 준비 완료:")
        logger.info(f"   - 학습 데이터: {X_train_scaled.shape}")
        logger.info(f"   - 검증 데이터: {X_val_scaled.shape}")
        logger.info(f"   - 테스트 데이터: {X_test_scaled.shape}")

        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict[str, Any]:
        """모델 학습"""
        logger.info("🚀 LSTM 모델 학습 시작")

        if self.model is None:
            self.build_model()

        # 콜백 설정
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config["patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(LSTM_MODEL_DIR / "best_model.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ]

        # 모델 학습
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        self.is_trained = True
        logger.info("✅ LSTM 모델 학습 완료")

        # 학습 결과 저장
        self.save_training_history()

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행"""
        if not self.is_trained or self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train()을 실행하세요.")

        logger.info("🔮 예측 수행 중...")
        predictions = self.model.predict(X, verbose=0)

        return predictions.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """모델 평가"""
        if not self.is_trained or self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train()을 실행하세요.")

        logger.info("📊 모델 평가 중...")

        # 예측 수행
        y_pred = self.predict(X)

        # 평가 지표 계산
        metrics = calculate_metrics(y, y_pred)

        # 결과 출력
        print_metrics(metrics, "LSTM 모델 평가 결과")

        return metrics

    def save_model(self, filepath: str = None):
        """모델 저장"""
        if self.model is None:
            logger.warning("저장할 모델이 없습니다.")
            return

        if filepath is None:
            filepath = LSTM_MODEL_DIR / "lstm_model.h5"

        # 모델 저장
        self.model.save(filepath)

        # 스케일러 저장
        if self.scaler is not None:
            scaler_path = str(filepath).replace(".h5", "_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)

        # 설정 저장
        config_path = str(filepath).replace(".h5", "_config.pkl")
        joblib.dump(self.config, config_path)

        logger.info(f"✅ 모델 저장 완료: {filepath}")

    def load_model(self, filepath: str = None):
        """모델 로드"""
        if filepath is None:
            filepath = LSTM_MODEL_DIR / "lstm_model.h5"

        try:
            # 모델 로드
            self.model = tf.keras.models.load_model(filepath)

            # 스케일러 로드
            scaler_path = str(filepath).replace(".h5", "_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            # 설정 로드
            config_path = str(filepath).replace(".h5", "_config.pkl")
            if os.path.exists(config_path):
                self.config = joblib.load(config_path)

            self.is_trained = True
            logger.info(f"✅ 모델 로드 완료: {filepath}")

        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {str(e)}")
            raise

    def save_training_history(self):
        """학습 히스토리 저장 및 시각화"""
        if self.history is None:
            return

        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(RESULTS_DIR / "lstm_training_history.csv", index=False)

        # 학습 곡선 시각화
        self.plot_training_history()

        logger.info("✅ 학습 히스토리 저장 완료")

    def plot_training_history(self):
        """학습 곡선 시각화"""
        if self.history is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss 곡선
        axes[0, 0].plot(self.history.history["loss"], label="Train Loss")
        axes[0, 0].plot(self.history.history["val_loss"], label="Val Loss")
        axes[0, 0].set_title("Model Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MAE 곡선
        axes[0, 1].plot(self.history.history["mae"], label="Train MAE")
        axes[0, 1].plot(self.history.history["val_mae"], label="Val MAE")
        axes[0, 1].set_title("Model MAE")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # MAPE 곡선
        axes[1, 0].plot(self.history.history["mape"], label="Train MAPE")
        axes[1, 0].plot(self.history.history["val_mape"], label="Val MAPE")
        axes[1, 0].set_title("Model MAPE")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("MAPE")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 학습률 곡선 (있을 경우)
        if "lr" in self.history.history:
            axes[1, 1].plot(self.history.history["lr"], label="Learning Rate")
            axes[1, 1].set_title("Learning Rate")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No LR data",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=12,
            )
            axes[1, 1].set_title("Learning Rate")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "lstm_training_history.png", dpi=300, bbox_inches="tight")
        plt.show()

    def predict_future(self, data_loader: DataLoader, n_periods: int = 6) -> np.ndarray:
        """미래 예측"""
        if not self.is_trained or self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train()을 실행하세요.")

        logger.info(f"🔮 향후 {n_periods}개월 예측 중...")

        # 최근 데이터 가져오기
        data = data_loader.processed_data

        # 국적-목적별 마지막 시퀀스 생성
        future_predictions = []

        for (country, purpose), group in data.groupby(["국적", "목적"]):
            group_sorted = group.sort_values(["연도", "월"])

            if len(group_sorted) >= self.config["sequence_length"]:
                # 마지막 시퀀스 추출
                last_sequence = (
                    group_sorted[FEATURE_COLUMNS].tail(self.config["sequence_length"]).values
                )

                # 스케일링
                if self.scaler is not None:
                    last_sequence = self.scaler.transform(
                        last_sequence.reshape(-1, last_sequence.shape[-1])
                    )
                    last_sequence = last_sequence.reshape(1, self.config["sequence_length"], -1)

                # 예측
                pred = self.model.predict(last_sequence, verbose=0)[0, 0]

                future_predictions.append({"국적": country, "목적": purpose, "예측값": pred})

        predictions_df = pd.DataFrame(future_predictions)

        # 예측 결과 저장
        predictions_df.to_csv(
            PREDICTIONS_DIR / "lstm_future_predictions.csv", index=False, encoding="utf-8-sig"
        )

        logger.info(f"✅ 미래 예측 완료: {len(predictions_df)}개 예측")

        return predictions_df

    def get_model_summary(self) -> Dict[str, Any]:
        """모델 요약 정보"""
        if self.model is None:
            return {"error": "모델이 생성되지 않았습니다."}

        summary = {
            "model_type": "LSTM",
            "total_params": self.model.count_params(),
            "trainable_params": sum(
                [tf.keras.backend.count_params(w) for w in self.model.trainable_weights]
            ),
            "config": self.config,
            "is_trained": self.is_trained,
        }

        if self.history is not None:
            summary["best_val_loss"] = min(self.history.history["val_loss"])
            summary["training_epochs"] = len(self.history.history["loss"])

        return summary


# 실행 예시
if __name__ == "__main__":
    # 데이터 로더 생성
    data_loader = DataLoader()

    # LSTM 모델 생성
    lstm_model = LSTMModel()

    # 모델 구조 생성
    lstm_model.build_model()

    # 데이터 준비
    X_train, y_train, X_val, y_val, X_test, y_test = lstm_model.prepare_data(data_loader)

    # 모델 학습
    history = lstm_model.train(X_train, y_train, X_val, y_val)

    # 모델 평가
    test_metrics = lstm_model.evaluate(X_test, y_test)

    # 모델 저장
    lstm_model.save_model()

    # 미래 예측
    future_predictions = lstm_model.predict_future(data_loader)

    # 모델 요약
    summary = lstm_model.get_model_summary()
    print("\n📋 모델 요약:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    logger.info("✅ LSTM 모델 테스트 완료")
