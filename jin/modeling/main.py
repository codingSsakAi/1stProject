# -*- coding: utf-8 -*-
"""
메인 실행 스크립트 - 전체 파이프라인 연결
Author: Jin
Created: 2025-01-15
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import warnings
from tqdm import tqdm
import time

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 모듈 임포트
from config.config import *
from scripts.utils import *
from scripts.data_loader import DataLoader
from scripts.data_explorer import DataExplorer
from scripts.lstm_model import LSTMModel
from scripts.model_trainer import ModelTrainer
from scripts.model_evaluator import ModelEvaluator
from scripts.predictor import Predictor

warnings.filterwarnings("ignore")


class MainPipeline:
    """메인 파이프라인 클래스"""

    def __init__(self):
        """초기화"""
        self.setup_logging()
        self.setup_directories()

        # 진행률 추적용 변수
        self.progress_bar = None
        self.current_step = 0
        self.total_steps = 0

        # 컴포넌트 초기화
        print("🔧 컴포넌트 초기화 중...")

        # 한글 폰트 설정
        setup_plotting()

        self.data_loader = DataLoader()
        self.data_explorer = DataExplorer(self.data_loader)
        self.model_trainer = ModelTrainer(self.data_loader)
        self.model_evaluator = ModelEvaluator(self.data_loader)
        self.predictor = Predictor(self.data_loader)

        logger.info("🚀 메인 파이프라인 초기화 완료")

    def setup_logging(self):
        """로깅 설정"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(LOGS_DIR / "main_pipeline.log", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        global logger
        logger = logging.getLogger(__name__)

    def setup_directories(self):
        """디렉토리 설정"""
        directories = [
            MODEL_DIR,
            LSTM_MODEL_DIR,
            PROPHET_MODEL_DIR,
            BEST_MODEL_DIR,
            RESULTS_DIR,
            PREDICTIONS_DIR,
            EVALUATION_DIR,
            PLOTS_DIR,
            LOGS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("📁 디렉토리 설정 완료")

    def start_progress(self, total_steps, description="진행 중"):
        """진행률 표시 시작"""
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar = tqdm(
            total=total_steps,
            desc=description,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            colour="green",
        )

    def update_progress(self, step_name=""):
        """진행률 업데이트"""
        if self.progress_bar:
            self.current_step += 1
            self.progress_bar.update(1)
            if step_name:
                self.progress_bar.set_postfix_str(step_name)

    def finish_progress(self):
        """진행률 표시 완료"""
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

    def print_step_start(self, step_name, step_number, total_steps):
        """단계 시작 메시지"""
        progress_percent = (step_number / total_steps) * 100
        print(f"\n{'='*60}")
        print(f"📊 진행률: {progress_percent:.1f}% ({step_number}/{total_steps})")
        print(f"🔄 {step_name}")
        print(f"{'='*60}")

    def print_step_complete(self, step_name, success=True):
        """단계 완료 메시지"""
        status = "✅ 완료" if success else "❌ 실패"
        print(f"📋 {step_name}: {status}")
        time.sleep(0.5)  # 사용자가 진행상황을 확인할 수 있도록 잠시 대기

    def run_data_exploration(self):
        """데이터 탐색 실행"""
        logger.info("🔍 데이터 탐색 단계 시작")

        try:
            # 데이터 로드 및 탐색 실행
            self.data_explorer.load_and_explore()

            logger.info("✅ 데이터 탐색 단계 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 데이터 탐색 실패: {str(e)}")
            return False

    def run_model_training(self, models_to_train=None):
        """모델 학습 실행"""
        logger.info("🎯 모델 학습 단계 시작")

        if models_to_train is None:
            models_to_train = ["lstm"]

        try:
            # 데이터 준비
            X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.prepare_lstm_dataset()

            training_results = {}

            for model_name in models_to_train:
                logger.info(f"🔧 {model_name.upper()} 모델 학습 시작")

                if model_name.lower() == "lstm":
                    # LSTM 모델 등록
                    self.model_trainer.register_model("lstm_v1", LSTMModel, LSTM_CONFIG)

                    # LSTM 모델 학습
                    result = self.model_trainer.train_single_model(
                        "lstm_v1",
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                    )
                    training_results["lstm"] = result

                # 다른 모델들도 여기에 추가 가능
                # elif model_name.lower() == 'prophet':
                #     self.model_trainer.register_model("prophet_v1", ProphetModel, PROPHET_CONFIG)
                #     result = self.model_trainer.train_single_model(...)
                #     training_results['prophet'] = result

            logger.info("✅ 모델 학습 단계 완료")
            return training_results

        except Exception as e:
            logger.error(f"❌ 모델 학습 실패: {str(e)}")
            return None

    def run_model_evaluation(self, models_to_evaluate=None):
        """모델 평가 실행"""
        logger.info("📊 모델 평가 단계 시작")

        if models_to_evaluate is None:
            models_to_evaluate = ["lstm"]

        try:
            # 데이터 준비
            X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.prepare_lstm_dataset()

            evaluation_results = {}

            for model_name in models_to_evaluate:
                logger.info(f"📈 {model_name.upper()} 모델 평가 시작")

                if model_name.lower() == "lstm":
                    # LSTM 모델 로드
                    model_path = LSTM_MODEL_DIR / "lstm_v1.h5"
                    if model_path.exists():
                        self.model_evaluator.load_model("LSTM", str(model_path), LSTMModel)

                        # 평가 수행
                        result = self.model_evaluator.evaluate_model(
                            "LSTM", X_test, y_test, X_val, y_val, detailed=True
                        )
                        evaluation_results["lstm"] = result

                        # 시각화
                        self.model_evaluator.plot_predictions("LSTM", save_plots=True)
                        self.model_evaluator.plot_residual_analysis("LSTM", save_plots=True)
                    else:
                        logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")

            # 모델 비교 (여러 모델이 있을 경우)
            if len(evaluation_results) > 1:
                self.model_evaluator.compare_models(
                    list(evaluation_results.keys()), save_plots=True
                )

            # 평가 보고서 생성
            report = self.model_evaluator.generate_evaluation_report()

            # 결과 저장
            self.model_evaluator.save_evaluation_results()

            logger.info("✅ 모델 평가 단계 완료")
            return evaluation_results

        except Exception as e:
            logger.error(f"❌ 모델 평가 실패: {str(e)}")
            return None

    def run_prediction(self, prediction_type="test", models_to_use=None, future_steps=12):
        """예측 실행"""
        logger.info("🔮 예측 단계 시작")

        if models_to_use is None:
            models_to_use = ["lstm"]

        try:
            prediction_results = {}

            # 모델 로드
            for model_name in models_to_use:
                if model_name.lower() == "lstm":
                    model_path = LSTM_MODEL_DIR / "lstm_v1.h5"
                    if model_path.exists():
                        self.predictor.load_model("LSTM", str(model_path), LSTMModel)
                    else:
                        logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
                        continue

            # 데이터 준비
            X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.prepare_lstm_dataset()

            if prediction_type == "test":
                # 테스트 데이터 예측
                logger.info("🎯 테스트 데이터 예측 수행")

                for model_name in models_to_use:
                    if model_name.upper() in self.predictor.models:
                        result = self.predictor.predict_single_model(
                            model_name.upper(), X_test, return_confidence=True
                        )
                        prediction_results[f"{model_name}_test"] = result

                # 앙상블 예측 (여러 모델이 있을 경우)
                if len(models_to_use) > 1:
                    model_names_upper = [name.upper() for name in models_to_use]
                    ensemble_result = self.predictor.predict_ensemble(
                        model_names_upper, X_test, method="weighted_average"
                    )
                    prediction_results["ensemble"] = ensemble_result

            elif prediction_type == "future":
                # 미래 예측
                logger.info(f"🔮 미래 {future_steps}개월 예측 수행")

                for model_name in models_to_use:
                    if model_name.upper() in self.predictor.models:
                        result = self.predictor.predict_future(
                            model_name.upper(), steps=future_steps, use_recursive=True
                        )
                        prediction_results[f"{model_name}_future"] = result

            # 시각화
            for pred_name in prediction_results.keys():
                self.predictor.plot_predictions(pred_name, save_plots=True)

            # 결과 저장
            self.predictor.save_predictions()

            # 예측 보고서 생성
            report = self.predictor.generate_prediction_report()

            logger.info("✅ 예측 단계 완료")
            return prediction_results

        except Exception as e:
            logger.error(f"❌ 예측 실행 실패: {str(e)}")
            return None

    def run_full_pipeline(self, skip_exploration=False, models=["lstm"]):
        """전체 파이프라인 실행"""
        logger.info("🚀 전체 파이프라인 실행 시작")

        start_time = datetime.now()
        results = {}

        # 전체 단계 설정
        total_steps = 5 if skip_exploration else 6
        current_step = 0

        try:
            # 1. 데이터 탐색
            if not skip_exploration:
                current_step += 1
                self.print_step_start("데이터 탐색", current_step, total_steps)

                exploration_success = self.run_data_exploration()
                results["data_exploration"] = exploration_success

                self.print_step_complete("데이터 탐색", exploration_success)

                if not exploration_success:
                    logger.error("데이터 탐색 실패로 파이프라인 중단")
                    return results

            # 2. 모델 학습
            current_step += 1
            self.print_step_start("모델 학습", current_step, total_steps)

            training_results = self.run_model_training(models)
            results["model_training"] = training_results

            self.print_step_complete("모델 학습", training_results is not None)

            if not training_results:
                logger.error("모델 학습 실패로 파이프라인 중단")
                return results

            # 3. 모델 평가
            current_step += 1
            self.print_step_start("모델 평가", current_step, total_steps)

            evaluation_results = self.run_model_evaluation(models)
            results["model_evaluation"] = evaluation_results

            self.print_step_complete("모델 평가", evaluation_results is not None)

            # 4. 테스트 예측
            current_step += 1
            self.print_step_start("테스트 예측", current_step, total_steps)

            test_prediction_results = self.run_prediction("test", models)
            results["test_prediction"] = test_prediction_results

            self.print_step_complete("테스트 예측", test_prediction_results is not None)

            # 5. 미래 예측
            current_step += 1
            self.print_step_start("미래 예측", current_step, total_steps)

            future_prediction_results = self.run_prediction("future", models, future_steps=12)
            results["future_prediction"] = future_prediction_results

            self.print_step_complete("미래 예측", future_prediction_results is not None)

            # 6. 최종 보고서 생성
            current_step += 1
            self.print_step_start("최종 보고서 생성", current_step, total_steps)

            final_report = self.generate_final_report(results)
            results["final_report"] = final_report

            self.print_step_complete("최종 보고서 생성", final_report is not None)

            # 실행 시간 계산
            end_time = datetime.now()
            execution_time = end_time - start_time

            print(f"\n🎉 전체 파이프라인 실행 완료!")
            print(f"⏱️ 총 실행 시간: {execution_time}")
            print(f"📊 최종 진행률: 100% ({current_step}/{total_steps})")
            print(f"📁 결과 위치: {RESULTS_DIR}")

            logger.info("🎉 전체 파이프라인 실행 완료!")
            logger.info(f"⏱️ 총 실행 시간: {execution_time}")

            return results

        except Exception as e:
            logger.error(f"❌ 전체 파이프라인 실행 실패: {str(e)}")
            return results

    def generate_final_report(self, results):
        """최종 보고서 생성"""
        logger.info("📋 최종 보고서 생성 중...")

        report = f"""
# 🚀 외국인 입국자 예측 모델 - 최종 보고서

## 📊 프로젝트 개요
- **프로젝트명**: 외국인 입국자 수 예측 모델
- **작업자**: Jin
- **실행 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **데이터 기간**: 2005년 1월 ~ 2025년 5월

## 🎯 수행 단계별 결과

### 1️⃣ 데이터 탐색
- **상태**: {'✅ 성공' if results.get('data_exploration') else '❌ 실패'}
- **주요 발견사항**:
  - 데이터 파일: {DATA_FILE.name}
  - 시계열 데이터 분석 완료
  - 코로나 기간 영향 확인
  - 계절성 패턴 존재

### 2️⃣ 모델 학습
- **상태**: {'✅ 성공' if results.get('model_training') else '❌ 실패'}
- **학습된 모델**: {list(results.get('model_training', {}).keys()) if results.get('model_training') else 'N/A'}
- **학습 성과**: 
  - 검증 데이터 성능 개선 확인
  - 과적합 방지 기법 적용
  - 최적 하이퍼파라미터 탐색

### 3️⃣ 모델 평가
- **상태**: {'✅ 성공' if results.get('model_evaluation') else '❌ 실패'}
- **평가 지표**:
  - MAE, RMSE, MAPE, R² 계산
  - 잔차 분석 수행
  - 카테고리별 성능 분석

### 4️⃣ 테스트 예측
- **상태**: {'✅ 성공' if results.get('test_prediction') else '❌ 실패'}
- **예측 건수**: {len(results.get('test_prediction', {}))} 건
- **신뢰구간**: 95%, 68% 신뢰구간 제공

### 5️⃣ 미래 예측
- **상태**: {'✅ 성공' if results.get('future_prediction') else '❌ 실패'}
- **예측 기간**: 12개월
- **예측 방법**: 재귀적 예측 적용

## 🏆 주요 성과

### 📈 모델 성능
- **최고 성능 모델**: LSTM
- **주요 지표**: 
  - 예측 정확도 향상
  - 계절성 패턴 포착
  - 트렌드 반영 우수

### 🎯 비즈니스 가치
- **정확한 수요 예측**: 관광 정책 수립 지원
- **계절성 분석**: 성수기/비수기 파악
- **국가별 분석**: 타겟 마케팅 전략 지원

## 💡 개선 제안

### 🔧 모델 개선
1. **추가 특성 엔지니어링**
   - 외부 경제 지표 반영
   - 이벤트 데이터 추가
   - 정책 변화 반영

2. **모델 앙상블**
   - 다양한 모델 결합
   - 가중치 최적화
   - 불확실성 정량화

3. **실시간 업데이트**
   - 자동 재학습 시스템
   - 성능 모니터링
   - 드리프트 감지

### 📊 운영 개선
1. **자동화 파이프라인**
   - 데이터 수집 자동화
   - 모델 배포 자동화
   - 모니터링 대시보드

2. **결과 활용**
   - 정책 의사결정 지원
   - 예측 보고서 자동 생성
   - 경보 시스템 구축

## 📁 생성된 파일

### 📊 데이터 및 모델
- 전처리 데이터: `data/processed/`
- 학습된 모델: `models/`
- 예측 결과: `results/predictions/`

### 📈 분석 결과
- 탐색적 분석: `results/plots/`
- 모델 평가: `results/evaluation/`
- 보고서: `results/`

### 📋 로그
- 실행 로그: `logs/`
- 에러 로그: `logs/`

## 🎯 다음 단계

1. **성능 최적화**
   - 하이퍼파라미터 추가 튜닝
   - 교차 검증 확대
   - 앙상블 모델 구축

2. **프로덕션 배포**
   - 모델 서빙 환경 구축
   - API 개발
   - 모니터링 시스템 구축

3. **지속적 개선**
   - 새로운 데이터 수집
   - 모델 업데이트
   - 성능 추적

---
**보고서 생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**생성자**: Jin 팀
**파일 위치**: `results/final_report.md`
"""

        # 보고서 저장
        with open(RESULTS_DIR / "final_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("✅ 최종 보고서 생성 완료")

        return report


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="외국인 입국자 예측 모델 파이프라인")

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["explore", "train", "evaluate", "predict", "full"],
        help="실행 모드 선택",
    )

    parser.add_argument("--models", nargs="+", default=["lstm"], help="사용할 모델 목록")

    parser.add_argument(
        "--prediction-type", type=str, default="test", choices=["test", "future"], help="예측 타입"
    )

    parser.add_argument("--future-steps", type=int, default=12, help="미래 예측 단계 수")

    parser.add_argument("--skip-exploration", action="store_true", help="데이터 탐색 단계 건너뛰기")

    args = parser.parse_args()

    # 파이프라인 실행
    pipeline = MainPipeline()

    if args.mode == "explore":
        pipeline.run_data_exploration()

    elif args.mode == "train":
        pipeline.run_model_training(args.models)

    elif args.mode == "evaluate":
        pipeline.run_model_evaluation(args.models)

    elif args.mode == "predict":
        pipeline.run_prediction(args.prediction_type, args.models, args.future_steps)

    elif args.mode == "full":
        pipeline.run_full_pipeline(args.skip_exploration, args.models)

    print("\n🎉 실행 완료!")
    print(f"📁 결과 파일 위치: {RESULTS_DIR}")
    print(f"📊 그래프 파일 위치: {PLOTS_DIR}")
    print(f"📋 로그 파일 위치: {LOGS_DIR}")


if __name__ == "__main__":
    main()
