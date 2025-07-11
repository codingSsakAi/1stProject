# -*- coding: utf-8 -*-
"""
모델 평가기 클래스
Author: Jin
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime
import json

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


class ModelEvaluator:
    """모델 평가기 클래스"""

    def __init__(self, data_loader: DataLoader = None):
        """
        초기화

        Args:
            data_loader: 데이터 로더 인스턴스
        """
        self.data_loader = data_loader or DataLoader()
        self.models = {}
        self.evaluation_results = {}
        self.predictions = {}

        logger.info("📊 모델 평가기 초기화")

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

        except Exception as e:
            logger.error(f"❌ {model_name} 모델 로드 실패: {str(e)}")
            raise

    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """
        모델 평가

        Args:
            model_name: 모델 이름
            X_test, y_test: 테스트 데이터
            X_val, y_val: 검증 데이터 (선택사항)
            detailed: 상세 평가 여부

        Returns:
            평가 결과 딕셔너리
        """
        if model_name not in self.models:
            raise ValueError(f"로드되지 않은 모델: {model_name}")

        logger.info(f"📊 {model_name} 모델 평가 시작")

        model = self.models[model_name]

        # 예측 수행
        y_pred_test = model.predict(X_test)
        self.predictions[f"{model_name}_test"] = y_pred_test

        # 기본 평가 지표 계산
        test_metrics = calculate_metrics(y_test, y_pred_test)

        result = {
            "model_name": model_name,
            "test_metrics": test_metrics,
            "test_predictions": y_pred_test,
            "test_actual": y_test,
        }

        # 검증 데이터 평가 (있을 경우)
        if X_val is not None and y_val is not None:
            y_pred_val = model.predict(X_val)
            val_metrics = calculate_metrics(y_val, y_pred_val)
            result["val_metrics"] = val_metrics
            result["val_predictions"] = y_pred_val
            result["val_actual"] = y_val
            self.predictions[f"{model_name}_val"] = y_pred_val

        # 상세 평가
        if detailed:
            detailed_analysis = self.detailed_evaluation(y_test, y_pred_test, model_name)
            result["detailed_analysis"] = detailed_analysis

        self.evaluation_results[model_name] = result

        logger.info(f"✅ {model_name} 모델 평가 완료")
        print_metrics(test_metrics, f"{model_name} 테스트 성능")

        return result

    def detailed_evaluation(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> Dict[str, Any]:
        """
        상세 평가 분석

        Args:
            y_true: 실제값
            y_pred: 예측값
            model_name: 모델 이름

        Returns:
            상세 분석 결과
        """
        logger.info(f"🔍 {model_name} 상세 평가 분석 중...")

        # 잔차 계산
        residuals = y_true - y_pred

        # 통계적 분석
        analysis = {
            "residuals": {
                "mean": np.mean(residuals),
                "std": np.std(residuals),
                "min": np.min(residuals),
                "max": np.max(residuals),
                "median": np.median(residuals),
                "q25": np.percentile(residuals, 25),
                "q75": np.percentile(residuals, 75),
            },
            "distribution": {
                "skewness": stats.skew(residuals),
                "kurtosis": stats.kurtosis(residuals),
                "normality_test": stats.jarque_bera(residuals),
            },
            "correlation": {
                "pearson": stats.pearsonr(y_true, y_pred)[0],
                "spearman": stats.spearmanr(y_true, y_pred)[0],
            },
        }

        # 예측 구간별 성능
        percentiles = [0, 25, 50, 75, 100]
        y_true_percentiles = np.percentile(y_true, percentiles)

        performance_by_range = {}
        for i in range(len(percentiles) - 1):
            mask = (y_true >= y_true_percentiles[i]) & (y_true < y_true_percentiles[i + 1])
            if np.any(mask):
                range_metrics = calculate_metrics(y_true[mask], y_pred[mask])
                performance_by_range[f"P{percentiles[i]}-P{percentiles[i+1]}"] = range_metrics

        analysis["performance_by_range"] = performance_by_range

        return analysis

    def plot_predictions(self, model_name: str, save_plots: bool = True):
        """
        예측 결과 시각화

        Args:
            model_name: 모델 이름
            save_plots: 그래프 저장 여부
        """
        if model_name not in self.evaluation_results:
            logger.warning(f"평가되지 않은 모델: {model_name}")
            return

        result = self.evaluation_results[model_name]
        y_true = result["test_actual"]
        y_pred = result["test_predictions"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 시계열 예측 비교
        axes[0, 0].plot(y_true, label="실제값", alpha=0.7)
        axes[0, 0].plot(y_pred, label="예측값", alpha=0.7)
        axes[0, 0].set_title(f"{model_name} - 시계열 예측 비교")
        axes[0, 0].set_xlabel("시간")
        axes[0, 0].set_ylabel("입국자수")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 산점도 (실제값 vs 예측값)
        axes[0, 1].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 1].plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        axes[0, 1].set_title(f"{model_name} - 실제값 vs 예측값")
        axes[0, 1].set_xlabel("실제값")
        axes[0, 1].set_ylabel("예측값")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 잔차 분포
        residuals = y_true - y_pred
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True)
        axes[1, 0].axvline(
            np.mean(residuals), color="red", linestyle="--", label=f"평균: {np.mean(residuals):.2f}"
        )
        axes[1, 0].set_title(f"{model_name} - 잔차 분포")
        axes[1, 0].set_xlabel("잔차")
        axes[1, 0].set_ylabel("밀도")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 잔차 vs 예측값
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color="red", linestyle="--")
        axes[1, 1].set_title(f"{model_name} - 잔차 vs 예측값")
        axes[1, 1].set_xlabel("예측값")
        axes[1, 1].set_ylabel("잔차")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            save_path = PLOTS_DIR / f"{model_name.lower()}_evaluation.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"📊 예측 결과 그래프 저장: {save_path}")

        plt.show()

    def plot_residual_analysis(self, model_name: str, save_plots: bool = True):
        """
        잔차 분석 시각화

        Args:
            model_name: 모델 이름
            save_plots: 그래프 저장 여부
        """
        if model_name not in self.evaluation_results:
            logger.warning(f"평가되지 않은 모델: {model_name}")
            return

        result = self.evaluation_results[model_name]
        y_true = result["test_actual"]
        y_pred = result["test_predictions"]
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 잔차 시계열
        axes[0, 0].plot(residuals, alpha=0.7)
        axes[0, 0].axhline(y=0, color="red", linestyle="--")
        axes[0, 0].set_title(f"{model_name} - 잔차 시계열")
        axes[0, 0].set_xlabel("시간")
        axes[0, 0].set_ylabel("잔차")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. QQ 플롯
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title(f"{model_name} - 잔차 QQ Plot")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 잔차 박스플롯
        axes[1, 0].boxplot(residuals)
        axes[1, 0].set_title(f"{model_name} - 잔차 박스플롯")
        axes[1, 0].set_ylabel("잔차")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 잔차 자기상관
        from pandas.plotting import autocorrelation_plot

        autocorrelation_plot(pd.Series(residuals), ax=axes[1, 1])
        axes[1, 1].set_title(f"{model_name} - 잔차 자기상관")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            save_path = PLOTS_DIR / f"{model_name.lower()}_residual_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"📊 잔차 분석 그래프 저장: {save_path}")

        plt.show()

    def compare_models(self, model_names: List[str] = None, save_plots: bool = True):
        """
        모델 성능 비교

        Args:
            model_names: 비교할 모델 이름 리스트
            save_plots: 그래프 저장 여부
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())

        logger.info(f"📊 모델 성능 비교: {model_names}")

        # 비교 데이터 준비
        comparison_data = []
        for model_name in model_names:
            if model_name in self.evaluation_results:
                result = self.evaluation_results[model_name]
                metrics = result["test_metrics"]
                comparison_data.append(
                    {
                        "Model": model_name,
                        "MAE": metrics["mae"],
                        "RMSE": metrics["rmse"],
                        "MAPE": metrics["mape"],
                        "R²": metrics["r2"],
                    }
                )

        comparison_df = pd.DataFrame(comparison_data)

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        metrics_to_plot = ["MAE", "RMSE", "MAPE", "R²"]

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(comparison_df["Model"], comparison_df[metric])
            ax.set_title(f"모델별 {metric} 비교")
            ax.set_ylabel(metric)

            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                )

            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            save_path = PLOTS_DIR / "model_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"📊 모델 비교 그래프 저장: {save_path}")

        plt.show()

        # 비교 결과 출력
        print("\n📊 모델 성능 비교 결과:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

        # 최고 성능 모델 찾기
        best_models = {}
        for metric in metrics_to_plot:
            if metric == "R²":
                best_idx = comparison_df[metric].idxmax()
            else:
                best_idx = comparison_df[metric].idxmin()
            best_models[metric] = comparison_df.loc[best_idx, "Model"]

        print("\n🏆 지표별 최고 성능 모델:")
        for metric, model in best_models.items():
            print(f"  {metric}: {model}")

        return comparison_df

    def performance_by_category(self, model_name: str, category_data: Dict[str, Any]):
        """
        카테고리별 성능 분석

        Args:
            model_name: 모델 이름
            category_data: 카테고리별 데이터 (예: 국가별, 목적별)
        """
        if model_name not in self.evaluation_results:
            logger.warning(f"평가되지 않은 모델: {model_name}")
            return

        logger.info(f"📊 {model_name} 카테고리별 성능 분석")

        result = self.evaluation_results[model_name]
        y_true = result["test_actual"]
        y_pred = result["test_predictions"]

        category_performance = {}

        for category, indices in category_data.items():
            if len(indices) > 0:
                cat_y_true = y_true[indices]
                cat_y_pred = y_pred[indices]
                cat_metrics = calculate_metrics(cat_y_true, cat_y_pred)
                category_performance[category] = cat_metrics

        # 결과 저장
        category_df = pd.DataFrame(category_performance).T
        category_df.to_csv(
            EVALUATION_DIR / f"{model_name.lower()}_category_performance.csv", encoding="utf-8-sig"
        )

        return category_performance

    def generate_evaluation_report(self, model_names: List[str] = None) -> str:
        """
        평가 보고서 생성

        Args:
            model_names: 보고서에 포함할 모델 이름 리스트

        Returns:
            보고서 텍스트
        """
        logger.info("📋 평가 보고서 생성 중...")

        if model_names is None:
            model_names = list(self.evaluation_results.keys())

        report = f"""
# 모델 평가 보고서

## 📊 평가 개요
- 평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 평가 모델 수: {len(model_names)}
- 평가 지표: MAE, RMSE, MAPE, R²

## 📈 모델별 성능 결과
"""

        for model_name in model_names:
            if model_name in self.evaluation_results:
                result = self.evaluation_results[model_name]
                metrics = result["test_metrics"]

                report += f"""
### {model_name}
- **MAE**: {metrics['mae']:.4f}
- **RMSE**: {metrics['rmse']:.4f}
- **MAPE**: {metrics['mape']:.4f}%
- **R²**: {metrics['r2']:.4f}

"""

                # 상세 분석 (있을 경우)
                if "detailed_analysis" in result:
                    detailed = result["detailed_analysis"]
                    report += f"""
#### 상세 분석
- **잔차 평균**: {detailed['residuals']['mean']:.4f}
- **잔차 표준편차**: {detailed['residuals']['std']:.4f}
- **상관관계 (Pearson)**: {detailed['correlation']['pearson']:.4f}
- **정규성 검정 p-value**: {detailed['distribution']['normality_test'][1]:.4f}

"""

        # 최고 성능 모델 선정
        best_model = None
        best_score = float("inf")

        for model_name in model_names:
            if model_name in self.evaluation_results:
                score = self.evaluation_results[model_name]["test_metrics"]["mape"]
                if score < best_score:
                    best_score = score
                    best_model = model_name

        report += f"""
## 🏆 최고 성능 모델
- **모델명**: {best_model}
- **MAPE 점수**: {best_score:.4f}%

## 📋 권장사항
1. 최고 성능 모델인 {best_model} 사용 권장
2. 잔차 분석 결과 확인 필요
3. 카테고리별 성능 차이 분석 수행
4. 추가 특성 엔지니어링 고려

---
생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # 보고서 저장
        with open(EVALUATION_DIR / "evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("✅ 평가 보고서 생성 완료")

        return report

    def save_evaluation_results(self):
        """평가 결과 저장"""
        logger.info("💾 평가 결과 저장 중...")

        # 평가 결과 요약
        summary = {}
        for model_name, result in self.evaluation_results.items():
            summary[model_name] = {
                "test_metrics": result["test_metrics"],
                "val_metrics": result.get("val_metrics", {}),
                "model_name": model_name,
            }

        # JSON 저장
        with open(EVALUATION_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 예측 결과 저장
        for pred_name, predictions in self.predictions.items():
            pred_df = pd.DataFrame({"predictions": predictions})
            pred_df.to_csv(
                PREDICTIONS_DIR / f"{pred_name}_predictions.csv", index=False, encoding="utf-8-sig"
            )

        logger.info("✅ 평가 결과 저장 완료")


# 실행 예시
if __name__ == "__main__":
    # 데이터 로더 생성
    data_loader = DataLoader()

    # 모델 평가기 생성
    evaluator = ModelEvaluator(data_loader)

    # LSTM 모델 로드 (예시)
    lstm_model_path = LSTM_MODEL_DIR / "lstm_model.h5"
    if lstm_model_path.exists():
        evaluator.load_model("LSTM", lstm_model_path, LSTMModel)

    # 테스트 데이터 준비
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_lstm_dataset()

    # 모델 평가
    if "LSTM" in evaluator.models:
        lstm_result = evaluator.evaluate_model("LSTM", X_test, y_test, X_val, y_val)

        # 시각화
        evaluator.plot_predictions("LSTM")
        evaluator.plot_residual_analysis("LSTM")

        # 보고서 생성
        report = evaluator.generate_evaluation_report(["LSTM"])
        print(report)

        # 결과 저장
        evaluator.save_evaluation_results()

    logger.info("✅ 모델 평가기 테스트 완료")
