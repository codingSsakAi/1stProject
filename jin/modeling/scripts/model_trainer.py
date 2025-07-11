# -*- coding: utf-8 -*-
"""
모델 학습 관리자
Author: Jin
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings

# 프로젝트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.config import *
from scripts.utils import *
from scripts.data_loader import DataLoader
from scripts.lstm_model import LSTMModel

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """모델 학습 관리자 클래스"""

    def __init__(self, data_loader: DataLoader = None):
        """
        초기화

        Args:
            data_loader: 데이터 로더 인스턴스
        """
        self.data_loader = data_loader or DataLoader()
        self.models = {}
        self.training_results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float("inf")

        # 학습 시작 시간
        self.start_time = None

        logger.info("🎯 모델 학습 관리자 초기화")

    def register_model(self, name: str, model_class, config: Dict[str, Any] = None):
        """
        모델 등록

        Args:
            name: 모델 이름
            model_class: 모델 클래스
            config: 모델 설정
        """
        self.models[name] = {
            "class": model_class,
            "config": config,
            "instance": None,
            "trained": False,
            "metrics": None,
        }

        logger.info(f"📝 모델 등록: {name}")

    def train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        단일 모델 학습

        Args:
            model_name: 모델 이름
            X_train, y_train: 학습 데이터
            X_val, y_val: 검증 데이터
            X_test, y_test: 테스트 데이터

        Returns:
            학습 결과 딕셔너리
        """
        if model_name not in self.models:
            raise ValueError(f"등록되지 않은 모델: {model_name}")

        logger.info(f"🚀 {model_name} 모델 학습 시작")
        start_time = time.time()

        model_info = self.models[model_name]

        try:
            # 모델 인스턴스 생성
            if model_info["config"]:
                model_instance = model_info["class"](model_info["config"])
            else:
                model_instance = model_info["class"]()

            # 모델 학습
            if hasattr(model_instance, "prepare_data"):
                # 데이터 준비가 필요한 경우 (예: LSTM)
                X_train_prep, y_train_prep, X_val_prep, y_val_prep, X_test_prep, y_test_prep = (
                    model_instance.prepare_data(self.data_loader)
                )

                # 학습 실행
                training_history = model_instance.train(
                    X_train_prep, y_train_prep, X_val_prep, y_val_prep
                )

                # 평가 실행
                test_metrics = model_instance.evaluate(X_test_prep, y_test_prep)
                val_metrics = model_instance.evaluate(X_val_prep, y_val_prep)

            else:
                # 일반적인 경우
                training_history = model_instance.train(X_train, y_train, X_val, y_val)
                test_metrics = model_instance.evaluate(X_test, y_test)
                val_metrics = model_instance.evaluate(X_val, y_val)

            # 학습 시간 계산
            training_time = time.time() - start_time

            # 결과 저장
            result = {
                "model_name": model_name,
                "training_time": training_time,
                "training_history": training_history,
                "test_metrics": test_metrics,
                "val_metrics": val_metrics,
                "model_instance": model_instance,
                "success": True,
                "error": None,
            }

            # 모델 정보 업데이트
            self.models[model_name]["instance"] = model_instance
            self.models[model_name]["trained"] = True
            self.models[model_name]["metrics"] = test_metrics

            # 최적 모델 업데이트
            current_score = test_metrics.get(MODEL_SELECTION_METRIC, float("inf"))
            if current_score < self.best_score:
                self.best_score = current_score
                self.best_model = model_instance
                self.best_model_name = model_name

                logger.info(
                    f"🏆 새로운 최적 모델: {model_name} ({MODEL_SELECTION_METRIC}: {current_score:.4f})"
                )

            logger.info(f"✅ {model_name} 모델 학습 완료 ({training_time:.2f}초)")

            return result

        except Exception as e:
            logger.error(f"❌ {model_name} 모델 학습 실패: {str(e)}")

            result = {
                "model_name": model_name,
                "training_time": time.time() - start_time,
                "training_history": None,
                "test_metrics": None,
                "val_metrics": None,
                "model_instance": None,
                "success": False,
                "error": str(e),
            }

            return result

    def train_all_models(self, save_models: bool = True) -> Dict[str, Any]:
        """
        모든 등록된 모델 학습

        Args:
            save_models: 모델 저장 여부

        Returns:
            전체 학습 결과
        """
        logger.info("🎯 전체 모델 학습 시작")
        self.start_time = time.time()

        # 데이터 준비
        logger.info("📊 데이터 준비 중...")
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.prepare_lstm_dataset()

        # 각 모델 학습
        for model_name in self.models.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"🎯 {model_name} 모델 학습 시작")
            logger.info(f"{'='*50}")

            result = self.train_single_model(
                model_name, X_train, y_train, X_val, y_val, X_test, y_test
            )

            self.training_results[model_name] = result

            # 모델 저장
            if save_models and result["success"] and result["model_instance"]:
                try:
                    model_instance = result["model_instance"]
                    if hasattr(model_instance, "save_model"):
                        save_path = MODEL_DIR / model_name.lower() / f"{model_name.lower()}_model"
                        model_instance.save_model(save_path)
                        logger.info(f"💾 {model_name} 모델 저장 완료")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} 모델 저장 실패: {str(e)}")

        # 전체 학습 시간
        total_time = time.time() - self.start_time

        logger.info(f"\n{'='*50}")
        logger.info(f"✅ 전체 모델 학습 완료 ({total_time:.2f}초)")
        logger.info(
            f"🏆 최적 모델: {self.best_model_name} ({MODEL_SELECTION_METRIC}: {self.best_score:.4f})"
        )
        logger.info(f"{'='*50}")

        # 결과 저장
        self.save_training_results()

        return self.training_results

    def save_training_results(self):
        """학습 결과 저장"""
        logger.info("💾 학습 결과 저장 중...")

        # 결과 요약 생성
        summary = self.generate_training_summary()

        # JSON 저장
        results_for_json = {}
        for model_name, result in self.training_results.items():
            results_for_json[model_name] = {
                "model_name": result["model_name"],
                "training_time": result["training_time"],
                "test_metrics": result["test_metrics"],
                "val_metrics": result["val_metrics"],
                "success": result["success"],
                "error": result["error"],
            }

        # 전체 결과 저장
        final_results = {
            "training_summary": summary,
            "model_results": results_for_json,
            "best_model": self.best_model_name,
            "best_score": self.best_score,
            "training_date": datetime.now().isoformat(),
        }

        # JSON 파일로 저장
        with open(RESULTS_DIR / "training_results.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        # CSV 파일로 저장
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(RESULTS_DIR / "training_summary.csv", encoding="utf-8-sig")

        logger.info("✅ 학습 결과 저장 완료")

    def generate_training_summary(self) -> Dict[str, Dict[str, Any]]:
        """학습 결과 요약 생성"""
        summary = {}

        for model_name, result in self.training_results.items():
            if result["success"]:
                summary[model_name] = {
                    "training_time": result["training_time"],
                    "test_mae": result["test_metrics"]["mae"],
                    "test_mse": result["test_metrics"]["mse"],
                    "test_rmse": result["test_metrics"]["rmse"],
                    "test_mape": result["test_metrics"]["mape"],
                    "test_r2": result["test_metrics"]["r2"],
                    "val_mae": result["val_metrics"]["mae"],
                    "val_mse": result["val_metrics"]["mse"],
                    "val_rmse": result["val_metrics"]["rmse"],
                    "val_mape": result["val_metrics"]["mape"],
                    "val_r2": result["val_metrics"]["r2"],
                    "status": "success",
                }
            else:
                summary[model_name] = {
                    "training_time": result["training_time"],
                    "test_mae": None,
                    "test_mse": None,
                    "test_rmse": None,
                    "test_mape": None,
                    "test_r2": None,
                    "val_mae": None,
                    "val_mse": None,
                    "val_rmse": None,
                    "val_mape": None,
                    "val_r2": None,
                    "status": "failed",
                    "error": result["error"],
                }

        return summary

    def compare_models(self) -> pd.DataFrame:
        """모델 성능 비교"""
        logger.info("📊 모델 성능 비교 중...")

        comparison_data = []

        for model_name, result in self.training_results.items():
            if result["success"]:
                comparison_data.append(
                    {
                        "Model": model_name,
                        "Training Time (s)": result["training_time"],
                        "Test MAE": result["test_metrics"]["mae"],
                        "Test RMSE": result["test_metrics"]["rmse"],
                        "Test MAPE": result["test_metrics"]["mape"],
                        "Test R²": result["test_metrics"]["r2"],
                        "Val MAE": result["val_metrics"]["mae"],
                        "Val RMSE": result["val_metrics"]["rmse"],
                        "Val MAPE": result["val_metrics"]["mape"],
                        "Val R²": result["val_metrics"]["r2"],
                        "Status": "Success",
                    }
                )
            else:
                comparison_data.append(
                    {
                        "Model": model_name,
                        "Training Time (s)": result["training_time"],
                        "Test MAE": None,
                        "Test RMSE": None,
                        "Test MAPE": None,
                        "Test R²": None,
                        "Val MAE": None,
                        "Val RMSE": None,
                        "Val MAPE": None,
                        "Val R²": None,
                        "Status": "Failed",
                    }
                )

        comparison_df = pd.DataFrame(comparison_data)

        # 결과 저장
        comparison_df.to_csv(
            RESULTS_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig"
        )

        # 결과 출력
        print("\n📊 모델 성능 비교 결과:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

        return comparison_df

    def get_best_model(self) -> Tuple[str, Any]:
        """최적 모델 반환"""
        if self.best_model is None:
            logger.warning("학습된 모델이 없습니다.")
            return None, None

        return self.best_model_name, self.best_model

    def save_best_model(self):
        """최적 모델 저장"""
        if self.best_model is None:
            logger.warning("저장할 최적 모델이 없습니다.")
            return

        # 최적 모델 저장
        if hasattr(self.best_model, "save_model"):
            save_path = BEST_MODEL_DIR / f"best_{self.best_model_name.lower()}_model"
            self.best_model.save_model(save_path)

            # 최적 모델 정보 저장
            best_model_info = {
                "model_name": self.best_model_name,
                "score": self.best_score,
                "metric": MODEL_SELECTION_METRIC,
                "metrics": self.training_results[self.best_model_name]["test_metrics"],
                "save_path": str(save_path),
                "save_date": datetime.now().isoformat(),
            }

            with open(BEST_MODEL_DIR / "best_model_info.json", "w", encoding="utf-8") as f:
                json.dump(best_model_info, f, indent=2, ensure_ascii=False)

            logger.info(f"🏆 최적 모델 저장 완료: {self.best_model_name}")

    def generate_report(self) -> str:
        """학습 결과 보고서 생성"""
        logger.info("📋 학습 결과 보고서 생성 중...")

        total_time = time.time() - self.start_time if self.start_time else 0

        report = f"""
# 모델 학습 결과 보고서

## 📊 학습 개요
- 학습 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 총 학습 시간: {total_time:.2f}초
- 학습 모델 수: {len(self.models)}
- 성공 모델 수: {sum(1 for r in self.training_results.values() if r['success'])}

## 🏆 최적 모델
- 모델명: {self.best_model_name}
- 평가 지표: {MODEL_SELECTION_METRIC}
- 점수: {self.best_score:.4f}

## 📈 모델별 성능 비교
"""

        for model_name, result in self.training_results.items():
            if result["success"]:
                report += f"""
### {model_name}
- 학습 시간: {result['training_time']:.2f}초
- Test MAE: {result['test_metrics']['mae']:.4f}
- Test RMSE: {result['test_metrics']['rmse']:.4f}
- Test MAPE: {result['test_metrics']['mape']:.4f}%
- Test R²: {result['test_metrics']['r2']:.4f}
"""
            else:
                report += f"""
### {model_name}
- 상태: 학습 실패
- 오류: {result['error']}
"""

        report += f"""
## 🎯 결론 및 권장사항
1. 최적 모델: {self.best_model_name}
2. 주요 성능 지표: {MODEL_SELECTION_METRIC} = {self.best_score:.4f}
3. 다음 단계: 최적 모델을 사용한 예측 수행

---
생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # 보고서 저장
        with open(RESULTS_DIR / "training_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("✅ 학습 결과 보고서 생성 완료")

        return report


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="모델 학습 관리자")
    parser.add_argument(
        "--models", nargs="+", default=["lstm"], help="학습할 모델 리스트 (lstm, prophet, etc.)"
    )
    parser.add_argument("--save-models", action="store_true", default=True, help="모델 저장 여부")
    parser.add_argument("--compare", action="store_true", default=True, help="모델 비교 수행 여부")

    args = parser.parse_args()

    # 데이터 로더 생성
    data_loader = DataLoader()

    # 모델 학습 관리자 생성
    trainer = ModelTrainer(data_loader)

    # 모델 등록
    if "lstm" in args.models:
        trainer.register_model("LSTM", LSTMModel, LSTM_CONFIG)

    # 추가 모델들은 여기에 등록
    # if 'prophet' in args.models:
    #     trainer.register_model('Prophet', ProphetModel, PROPHET_CONFIG)

    # 전체 모델 학습
    results = trainer.train_all_models(save_models=args.save_models)

    # 모델 비교
    if args.compare:
        comparison_df = trainer.compare_models()

    # 최적 모델 저장
    trainer.save_best_model()

    # 보고서 생성
    report = trainer.generate_report()
    print(report)

    logger.info("🎉 모델 학습 관리자 실행 완료")


if __name__ == "__main__":
    main()
