# -*- coding: utf-8 -*-
"""
예측 실행기 클래스
Author: Jin
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime, timedelta
import json
from pathlib import Path
import joblib

# 프로젝트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.config import *
from scripts.utils import *
from scripts.data_loader import DataLoader
from scripts.lstm_model import LSTMModel

# 한글 폰트 설정
setup_plotting()

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class Predictor:
    """예측 실행기 클래스"""

    def __init__(self, data_loader: DataLoader = None):
        """
        초기화

        Args:
            data_loader: 데이터 로더 인스턴스
        """
        self.data_loader = data_loader or DataLoader()
        self.models = {}
        self.predictions = {}
        self.ensemble_predictions = {}
        self.feature_scalers = {}
        self.target_scalers = {}

        logger.info("🔮 예측 실행기 초기화")

    def load_model(self, model_name: str, model_path: str, model_class):
        """
        모델 로드

        Args:
            model_name: 모델 이름
            model_path: 모델 파일 경로
            model_class: 모델 클래스
        """
        try:
            model_instance = model_class()
            model_instance.load_model(model_path)

            self.models[model_name] = model_instance
            logger.info(f"✅ {model_name} 모델 로드 완료")

            # 스케일러 로드 시도
            scaler_path = Path(model_path).parent / f"{model_name.lower()}_scalers.pkl"
            if scaler_path.exists():
                scalers = joblib.load(scaler_path)
                self.feature_scalers[model_name] = scalers.get("feature_scaler")
                self.target_scalers[model_name] = scalers.get("target_scaler")
                logger.info(f"✅ {model_name} 스케일러 로드 완료")

        except Exception as e:
            logger.error(f"❌ {model_name} 모델 로드 실패: {str(e)}")
            raise

    def prepare_prediction_data(
        self, data: pd.DataFrame, model_name: str, sequence_length: int = 12
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        예측용 데이터 준비

        Args:
            data: 입력 데이터
            model_name: 모델 이름
            sequence_length: 시퀀스 길이

        Returns:
            준비된 예측용 데이터와 메타데이터
        """
        logger.info(f"📊 {model_name} 예측용 데이터 준비 중...")

        # 데이터 복사
        df = data.copy()

        # 특성 컬럼 정의
        feature_columns = [
            col for col in df.columns if col not in ["날짜", "국적", "목적", "입국자수"]
        ]

        # 특성 데이터 추출
        X = df[feature_columns].values

        # 스케일링 (있을 경우)
        if model_name in self.feature_scalers and self.feature_scalers[model_name] is not None:
            X = self.feature_scalers[model_name].transform(X)

        # 시퀀스 데이터 생성 (LSTM용)
        if len(X) >= sequence_length:
            X_seq = []
            metadata = []

            for i in range(sequence_length, len(X) + 1):
                X_seq.append(X[i - sequence_length : i])
                metadata.append(
                    {
                        "index": i - 1,
                        "date": df.iloc[i - 1]["날짜"] if "날짜" in df.columns else None,
                        "nationality": df.iloc[i - 1]["국적"] if "국적" in df.columns else None,
                        "purpose": df.iloc[i - 1]["목적"] if "목적" in df.columns else None,
                    }
                )

            X_seq = np.array(X_seq)
            metadata_df = pd.DataFrame(metadata)

            logger.info(f"✅ 시퀀스 데이터 생성 완료: {X_seq.shape}")

            return X_seq, metadata_df
        else:
            logger.warning(f"⚠️ 데이터 길이 부족: {len(X)} < {sequence_length}")
            return None, None

    def predict_single_model(
        self, model_name: str, X: np.ndarray, return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        단일 모델 예측

        Args:
            model_name: 모델 이름
            X: 입력 데이터
            return_confidence: 신뢰구간 반환 여부

        Returns:
            예측 결과 딕셔너리
        """
        if model_name not in self.models:
            raise ValueError(f"로드되지 않은 모델: {model_name}")

        logger.info(f"🔮 {model_name} 예측 수행 중...")

        model = self.models[model_name]

        # 예측 수행
        predictions = model.predict(X)

        # 역스케일링 (있을 경우)
        if model_name in self.target_scalers and self.target_scalers[model_name] is not None:
            predictions = (
                self.target_scalers[model_name]
                .inverse_transform(predictions.reshape(-1, 1))
                .flatten()
            )

        result = {
            "model_name": model_name,
            "predictions": predictions,
            "num_predictions": len(predictions),
        }

        # 신뢰구간 계산 (몬테카를로 드롭아웃)
        if return_confidence and hasattr(model, "predict_with_uncertainty"):
            try:
                mean_pred, std_pred = model.predict_with_uncertainty(X)

                # 역스케일링
                if (
                    model_name in self.target_scalers
                    and self.target_scalers[model_name] is not None
                ):
                    mean_pred = (
                        self.target_scalers[model_name]
                        .inverse_transform(mean_pred.reshape(-1, 1))
                        .flatten()
                    )
                    # 표준편차는 스케일만 조정
                    scale = self.target_scalers[model_name].scale_[0]
                    std_pred = std_pred * scale

                result["confidence"] = {
                    "mean": mean_pred,
                    "std": std_pred,
                    "upper_95": mean_pred + 1.96 * std_pred,
                    "lower_95": mean_pred - 1.96 * std_pred,
                    "upper_68": mean_pred + std_pred,
                    "lower_68": mean_pred - std_pred,
                }
            except Exception as e:
                logger.warning(f"신뢰구간 계산 실패: {str(e)}")

        self.predictions[model_name] = result

        logger.info(f"✅ {model_name} 예측 완료: {len(predictions)}개")

        return result

    def predict_ensemble(
        self,
        model_names: List[str],
        X: np.ndarray,
        weights: List[float] = None,
        method: str = "weighted_average",
    ) -> Dict[str, Any]:
        """
        앙상블 예측

        Args:
            model_names: 사용할 모델 이름 리스트
            X: 입력 데이터
            weights: 모델별 가중치 (None이면 균등 가중치)
            method: 앙상블 방법 ('weighted_average', 'median', 'best_performance')

        Returns:
            앙상블 예측 결과
        """
        logger.info(f"🎯 앙상블 예측 수행: {model_names}")

        # 개별 모델 예측
        individual_predictions = {}
        for model_name in model_names:
            if model_name in self.models:
                pred_result = self.predict_single_model(model_name, X)
                individual_predictions[model_name] = pred_result["predictions"]

        if not individual_predictions:
            raise ValueError("유효한 모델이 없습니다.")

        # 가중치 설정
        if weights is None:
            weights = [1.0 / len(individual_predictions)] * len(individual_predictions)
        elif len(weights) != len(individual_predictions):
            raise ValueError("가중치 개수가 모델 개수와 일치하지 않습니다.")

        # 앙상블 예측 수행
        predictions_array = np.array(list(individual_predictions.values()))

        if method == "weighted_average":
            ensemble_pred = np.average(predictions_array, axis=0, weights=weights)
        elif method == "median":
            ensemble_pred = np.median(predictions_array, axis=0)
        elif method == "best_performance":
            # 최고 성능 모델 선택 (가중치 기반)
            best_model_idx = np.argmax(weights)
            ensemble_pred = predictions_array[best_model_idx]
        else:
            raise ValueError(f"지원하지 않는 앙상블 방법: {method}")

        result = {
            "ensemble_method": method,
            "models_used": model_names,
            "weights": weights,
            "predictions": ensemble_pred,
            "individual_predictions": individual_predictions,
            "num_predictions": len(ensemble_pred),
        }

        # 예측 분산 계산
        pred_variance = np.var(predictions_array, axis=0)
        pred_std = np.std(predictions_array, axis=0)

        result["uncertainty"] = {
            "variance": pred_variance,
            "std": pred_std,
            "upper_95": ensemble_pred + 1.96 * pred_std,
            "lower_95": ensemble_pred - 1.96 * pred_std,
        }

        self.ensemble_predictions[f"ensemble_{method}"] = result

        logger.info(f"✅ 앙상블 예측 완료: {len(ensemble_pred)}개")

        return result

    def predict_future(
        self, model_name: str, steps: int = 12, use_recursive: bool = True
    ) -> Dict[str, Any]:
        """
        미래 예측

        Args:
            model_name: 사용할 모델 이름
            steps: 예측할 미래 단계 수
            use_recursive: 재귀적 예측 사용 여부

        Returns:
            미래 예측 결과
        """
        logger.info(f"🔮 {model_name} 미래 예측 수행: {steps}단계")

        if model_name not in self.models:
            raise ValueError(f"로드되지 않은 모델: {model_name}")

        # 최신 데이터 로드
        df = self.data_loader.load_data()

        # 예측용 데이터 준비
        X_seq, metadata = self.prepare_prediction_data(df, model_name)

        if X_seq is None:
            raise ValueError("예측용 데이터 준비 실패")

        model = self.models[model_name]

        # 마지막 시퀀스 가져오기
        last_sequence = X_seq[-1:].copy()

        future_predictions = []
        prediction_dates = []

        # 기준 날짜 설정
        last_date = pd.to_datetime(df["날짜"].iloc[-1])

        for step in range(steps):
            # 예측 수행
            pred = model.predict(last_sequence)

            # 역스케일링
            if model_name in self.target_scalers and self.target_scalers[model_name] is not None:
                pred_scaled = (
                    self.target_scalers[model_name].inverse_transform(pred.reshape(-1, 1)).flatten()
                )
            else:
                pred_scaled = pred.flatten()

            future_predictions.append(pred_scaled[0])

            # 다음 달 계산
            next_date = last_date + timedelta(days=32)
            next_date = next_date.replace(day=1)  # 월 첫날로 설정
            prediction_dates.append(next_date)
            last_date = next_date

            # 재귀적 예측을 위한 시퀀스 업데이트
            if use_recursive and step < steps - 1:
                # 새로운 특성 벡터 생성 (간단한 예시)
                new_features = last_sequence[0, -1, :].copy()

                # 입국자수 관련 특성들 업데이트
                if len(new_features) > 0:
                    new_features[0] = pred[0]  # 예측값으로 업데이트

                # 시퀀스 업데이트
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, :] = new_features

        result = {
            "model_name": model_name,
            "prediction_steps": steps,
            "recursive_prediction": use_recursive,
            "predictions": np.array(future_predictions),
            "prediction_dates": prediction_dates,
            "base_date": df["날짜"].iloc[-1],
        }

        logger.info(f"✅ {model_name} 미래 예측 완료: {steps}단계")

        return result

    def plot_predictions(
        self, prediction_name: str, historical_data: pd.DataFrame = None, save_plots: bool = True
    ):
        """
        예측 결과 시각화

        Args:
            prediction_name: 예측 결과 이름
            historical_data: 과거 데이터 (비교용)
            save_plots: 그래프 저장 여부
        """
        # 예측 결과 찾기
        pred_result = None
        if prediction_name in self.predictions:
            pred_result = self.predictions[prediction_name]
        elif prediction_name in self.ensemble_predictions:
            pred_result = self.ensemble_predictions[prediction_name]

        if pred_result is None:
            logger.warning(f"예측 결과를 찾을 수 없습니다: {prediction_name}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 예측 결과 시계열
        predictions = pred_result["predictions"]

        if "prediction_dates" in pred_result:
            # 미래 예측
            dates = pred_result["prediction_dates"]
            axes[0, 0].plot(dates, predictions, "b-", marker="o", label="예측값")
            axes[0, 0].set_title(f"{prediction_name} - 미래 예측")

            # 과거 데이터 추가 (있을 경우)
            if historical_data is not None:
                hist_dates = pd.to_datetime(historical_data["날짜"])
                hist_values = historical_data["입국자수"]
                axes[0, 0].plot(
                    hist_dates.tail(24), hist_values.tail(24), "g-", alpha=0.7, label="과거 실제값"
                )
        else:
            # 일반 예측
            axes[0, 0].plot(predictions, "b-", marker="o", label="예측값")
            axes[0, 0].set_title(f"{prediction_name} - 예측 결과")

        axes[0, 0].set_xlabel("날짜")
        axes[0, 0].set_ylabel("입국자수")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 신뢰구간 (있을 경우)
        if "confidence" in pred_result:
            conf = pred_result["confidence"]
            x_axis = range(len(predictions))

            axes[0, 1].plot(x_axis, conf["mean"], "b-", label="예측 평균")
            axes[0, 1].fill_between(
                x_axis, conf["lower_95"], conf["upper_95"], alpha=0.3, label="95% 신뢰구간"
            )
            axes[0, 1].fill_between(
                x_axis, conf["lower_68"], conf["upper_68"], alpha=0.5, label="68% 신뢰구간"
            )
            axes[0, 1].set_title(f"{prediction_name} - 신뢰구간")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        elif "uncertainty" in pred_result:
            # 앙상블 불확실성
            unc = pred_result["uncertainty"]
            x_axis = range(len(predictions))

            axes[0, 1].plot(x_axis, predictions, "b-", label="앙상블 예측")
            axes[0, 1].fill_between(
                x_axis, unc["lower_95"], unc["upper_95"], alpha=0.3, label="95% 불확실성"
            )
            axes[0, 1].set_title(f"{prediction_name} - 불확실성")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 예측값 분포
        axes[1, 0].hist(predictions, bins=30, alpha=0.7, density=True)
        axes[1, 0].axvline(
            np.mean(predictions),
            color="red",
            linestyle="--",
            label=f"평균: {np.mean(predictions):.0f}",
        )
        axes[1, 0].set_title(f"{prediction_name} - 예측값 분포")
        axes[1, 0].set_xlabel("예측값")
        axes[1, 0].set_ylabel("밀도")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 개별 모델 비교 (앙상블인 경우)
        if "individual_predictions" in pred_result:
            individual_preds = pred_result["individual_predictions"]
            x_axis = range(len(predictions))

            for model_name, model_preds in individual_preds.items():
                axes[1, 1].plot(x_axis, model_preds, alpha=0.7, label=model_name)

            axes[1, 1].plot(x_axis, predictions, "k-", linewidth=2, label="앙상블")
            axes[1, 1].set_title(f"{prediction_name} - 개별 모델 비교")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # 예측값 통계
            stats_text = f"""
예측 통계:
평균: {np.mean(predictions):.0f}
표준편차: {np.std(predictions):.0f}
최소값: {np.min(predictions):.0f}
최대값: {np.max(predictions):.0f}
중앙값: {np.median(predictions):.0f}
"""
            axes[1, 1].text(
                0.1,
                0.5,
                stats_text,
                transform=axes[1, 1].transAxes,
                fontsize=12,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
            )
            axes[1, 1].set_title(f"{prediction_name} - 예측 통계")
            axes[1, 1].axis("off")

        plt.tight_layout()

        if save_plots:
            safe_name = prediction_name.replace(" ", "_").lower()
            save_path = PLOTS_DIR / f"{safe_name}_predictions.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"📊 예측 결과 그래프 저장: {save_path}")

        plt.show()

    def save_predictions(self, prediction_name: str = None):
        """
        예측 결과 저장

        Args:
            prediction_name: 저장할 예측 이름 (None이면 전체 저장)
        """
        logger.info("💾 예측 결과 저장 중...")

        if prediction_name:
            # 특정 예측 결과만 저장
            if prediction_name in self.predictions:
                self._save_single_prediction(prediction_name, self.predictions[prediction_name])
            elif prediction_name in self.ensemble_predictions:
                self._save_single_prediction(
                    prediction_name, self.ensemble_predictions[prediction_name]
                )
            else:
                logger.warning(f"예측 결과를 찾을 수 없습니다: {prediction_name}")
        else:
            # 모든 예측 결과 저장
            for name, result in self.predictions.items():
                self._save_single_prediction(name, result)

            for name, result in self.ensemble_predictions.items():
                self._save_single_prediction(name, result)

        logger.info("✅ 예측 결과 저장 완료")

    def _save_single_prediction(self, name: str, result: Dict[str, Any]):
        """단일 예측 결과 저장"""
        # CSV 저장
        pred_df = pd.DataFrame({"predictions": result["predictions"]})

        if "prediction_dates" in result:
            pred_df["prediction_date"] = result["prediction_dates"]

        if "confidence" in result:
            conf = result["confidence"]
            pred_df["confidence_mean"] = conf["mean"]
            pred_df["confidence_lower_95"] = conf["lower_95"]
            pred_df["confidence_upper_95"] = conf["upper_95"]

        safe_name = name.replace(" ", "_").lower()
        pred_df.to_csv(PREDICTIONS_DIR / f"{safe_name}.csv", index=False, encoding="utf-8-sig")

        # JSON 저장 (메타데이터 포함)
        result_copy = result.copy()

        # numpy 배열을 리스트로 변환
        for key, value in result_copy.items():
            if isinstance(value, np.ndarray):
                result_copy[key] = value.tolist()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        result_copy[key][sub_key] = sub_value.tolist()

        with open(PREDICTIONS_DIR / f"{safe_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(result_copy, f, indent=2, ensure_ascii=False, default=str)

    def generate_prediction_report(self, prediction_names: List[str] = None) -> str:
        """
        예측 보고서 생성

        Args:
            prediction_names: 보고서에 포함할 예측 이름 리스트

        Returns:
            보고서 텍스트
        """
        logger.info("📋 예측 보고서 생성 중...")

        if prediction_names is None:
            prediction_names = list(self.predictions.keys()) + list(
                self.ensemble_predictions.keys()
            )

        report = f"""
# 예측 보고서

## 📊 예측 개요
- 보고서 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 예측 수행 건수: {len(prediction_names)}
- 사용된 모델: {list(self.models.keys())}

## 🔮 예측 결과 요약
"""

        for pred_name in prediction_names:
            pred_result = None
            if pred_name in self.predictions:
                pred_result = self.predictions[pred_name]
            elif pred_name in self.ensemble_predictions:
                pred_result = self.ensemble_predictions[pred_name]

            if pred_result:
                predictions = pred_result["predictions"]

                report += f"""
### {pred_name}
- **예측 개수**: {len(predictions)}개
- **예측 평균**: {np.mean(predictions):.0f}명
- **예측 표준편차**: {np.std(predictions):.0f}명
- **예측 범위**: {np.min(predictions):.0f}명 ~ {np.max(predictions):.0f}명
"""

                if "prediction_dates" in pred_result:
                    start_date = min(pred_result["prediction_dates"])
                    end_date = max(pred_result["prediction_dates"])
                    report += f"- **예측 기간**: {start_date.strftime('%Y-%m')} ~ {end_date.strftime('%Y-%m')}\n"

                if "confidence" in pred_result:
                    conf = pred_result["confidence"]
                    report += f"- **신뢰구간 평균 폭**: {np.mean(conf['upper_95'] - conf['lower_95']):.0f}명\n"

        report += f"""

## 📈 주요 인사이트
1. 예측 결과의 전반적인 트렌드 분석
2. 계절성 및 주기성 패턴 확인
3. 모델별 예측 성향 차이 분석
4. 불확실성 수준 평가

## 💡 권장사항
1. 정기적인 모델 재학습 수행
2. 예측 결과의 지속적인 모니터링
3. 외부 요인 변화 시 모델 업데이트 고려
4. 예측 정확도 향상을 위한 특성 추가 검토

---
생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # 보고서 저장
        with open(RESULTS_DIR / "prediction_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("✅ 예측 보고서 생성 완료")

        return report


# 실행 예시
if __name__ == "__main__":
    # 데이터 로더 생성
    data_loader = DataLoader()

    # 예측 실행기 생성
    predictor = Predictor(data_loader)

    # LSTM 모델 로드 (예시)
    lstm_model_path = LSTM_MODEL_DIR / "lstm_model.h5"
    if lstm_model_path.exists():
        predictor.load_model("LSTM", lstm_model_path, LSTMModel)

    # 데이터 준비
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_lstm_dataset()

    # 단일 모델 예측
    if "LSTM" in predictor.models:
        lstm_prediction = predictor.predict_single_model("LSTM", X_test, return_confidence=True)

        # 미래 예측
        future_prediction = predictor.predict_future("LSTM", steps=12, use_recursive=True)

        # 시각화
        predictor.plot_predictions("LSTM")

        # 결과 저장
        predictor.save_predictions()

        # 보고서 생성
        report = predictor.generate_prediction_report()
        print(report)

    logger.info("✅ 예측 실행기 테스트 완료")
