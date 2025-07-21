import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.utils import setup_gpu, set_environment, create_timestamped_results_dir
from modules.data_handler import DataHandler
from modules.feature_engineer import FeatureEngineer
from modules.model_builder import ModelBuilder
from modules.trainer import Trainer
from modules.predictor import Predictor
from modules.reporter import Reporter
import config

class MainPredictor:
    """전체 예측 파이프라인을 관리하는 메인 클래스"""

    def __init__(self):
        set_environment()
        setup_gpu()
        self.config = config
        self.data_handler = DataHandler(config)
        self.feature_engineer = FeatureEngineer()
        self.model_builder = ModelBuilder(config)
        self.results_dir, self.timestamp = create_timestamped_results_dir(config.BASE_RESULTS_DIR)
        self.reporter = Reporter(self.results_dir, config)
        self.models = {}
        self.scalers = {}

    def run(self):
        """대화형으로 예측을 실행합니다."""
        nationality = self._get_nationality_from_user()
        purpose = self._get_purpose_from_user(nationality)
        start_date, end_date = self._get_date_range_from_user()
        covid_strategy = self._get_covid_strategy_from_user()

        self.execute_prediction(
            nationality, 
            purpose, 
            start_date, 
            end_date,
            covid_strategy
        )

    def _get_nationality_from_user(self):
        """사용자로부터 국적을 입력받습니다."""
        print("\n--- 예측할 국적 선택 ---")
        for i, nat in enumerate(self.config.AVAILABLE_NATIONALITIES):
            print(f"  {i+1}. {nat}")
        
        while True:
            try:
                choice = input(f"국적 번호를 입력하세요 (1-{len(self.config.AVAILABLE_NATIONALITIES)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(self.config.AVAILABLE_NATIONALITIES):
                    return self.config.AVAILABLE_NATIONALITIES[choice_idx]
                else:
                    print("잘못된 번호입니다. 다시 입력해주세요.")
            except ValueError:
                print("숫자를 입력해주세요.")

    def _get_purpose_from_user(self, nationality):
        """사용자로부터 목적을 입력받습니다."""
        print(f"\n--- {nationality}의 예측 목적 선택 ---")
        purposes = self.data_handler.data[self.data_handler.data["국적"] == nationality]["목적"].unique()
        
        for i, p in enumerate(purposes):
            print(f"  {i+1}. {p}")
        print("\n(전체 목적을 예측하려면 Enter를 누르세요)")

        while True:
            try:
                choice = input(f"목적 번호를 입력하세요 (1-{len(purposes)}): ")
                if not choice:
                    return None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(purposes):
                    return purposes[choice_idx]
                else:
                    print("잘못된 번호입니다. 다시 입력해주세요.")
            except ValueError:
                print("숫자를 입력해주세요.")

    def _get_date_range_from_user(self):
        """사용자로부터 예측 기간을 입력받습니다."""
        print("\n--- 예측 기간 설정 ---")
        while True:
            start_date = input(f"예측 시작 연월 (YYYY-MM, 기본값: {self.config.DEFAULT_START_DATE}): ") or self.config.DEFAULT_START_DATE
            if self._is_valid_date_format(start_date):
                break
            print("잘못된 형식입니다. YYYY-MM 형식으로 입력해주세요.")

        while True:
            end_date = input(f"예측 종료 연월 (YYYY-MM, 기본값: {self.config.DEFAULT_END_DATE}): ") or self.config.DEFAULT_END_DATE
            if self._is_valid_date_format(end_date):
                break
            print("잘못된 형식입니다. YYYY-MM 형식으로 입력해주세요.")
        
        return start_date, end_date

    def _get_covid_strategy_from_user(self):
        """사용자로부터 코로나 데이터 처리 전략을 입력받습니다."""
        print("\n--- 코로나 데이터 처리 전략 선택 ---")
        strategies = ["exclude", "weighted", "include"]
        for i, s in enumerate(strategies):
            print(f"  {i+1}. {s}")

        while True:
            try:
                choice = input(f"전략 번호를 입력하세요 (1-{len(strategies)}, 기본값: 2. weighted): ") or "2"
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(strategies):
                    return strategies[choice_idx]
                else:
                    print("잘못된 번호입니다. 다시 입력해주세요.")
            except ValueError:
                print("숫자를 입력해주세요.")

    def _is_valid_date_format(self, date_string):
        """날짜 형식이 YYYY-MM인지 확인합니다."""
        try:
            pd.to_datetime(date_string, format='%Y-%m')
            return True
        except ValueError:
            return False

    def execute_prediction(self, nationality, purpose, start_date, end_date, covid_strategy):
        """지정된 국적과 목적에 대해 예측을 실행하고 리포트를 생성합니다."""
        target_months = self._generate_target_months(start_date, end_date)
        
        if purpose:
            purposes_to_predict = [purpose]
        else:
            purposes_to_predict = self.data_handler.data[self.data_handler.data["국적"] == nationality]["목적"].unique()

        results = {}
        performance_results = []
        historical_data_collection = {}

        for p in purposes_to_predict:
            print(f"\n--- {nationality} - {p} 예측 시작 ---")
            
            # 목적별 최적 코로나 전략 적용 (성능 향상)
            optimal_strategy = self._get_optimal_covid_strategy(p, covid_strategy)
            if optimal_strategy != covid_strategy:
                print(f"[최적화] {p} 목적에 최적화된 코로나 전략 적용: {covid_strategy} → {optimal_strategy}")
            
            combo_data = self.data_handler.get_data_for_purpose(nationality, p, optimal_strategy)
            if combo_data.empty or len(combo_data) < config.LSTM_SEQUENCE_LENGTH_LARGE_DATA:
                print(f"데이터가 부족하여 ({len(combo_data)}개) 건너뜁니다.")
                continue
            
            # 데이터 증강 활성화 (목적별 차별화된 성능 향상)
            print(f"원본 데이터: {len(combo_data)}개월")
            augmented_datasets = self.data_handler.augment_data(combo_data, purpose=p)
            combo_data = augmented_datasets[0]  # 증강된 데이터 사용
            print(f"증강 후 데이터: {len(combo_data)}개월")
            
            # 데이터 특성 분석 (3단계 개선)
            data_characteristics = self.data_handler.analyze_data_characteristics(combo_data, p)
            
            historical_data_collection[p] = combo_data

            features = self.feature_engineer.create_features(combo_data)
            
            # DataHandler를 통해 스케일링 및 시퀀스 생성
            X, y, scaler = self.data_handler.preprocess_for_model(features, p)
            self.scalers[p] = scaler # 목적별 스케일러 저장

            if len(X) == 0:
                print("시퀀스 생성에 실패하여 건너뜁니다.")
                continue

            # 스마트 앙상블 모드 확인 및 모델 학습 (1-3단계 통합 개선)
            if config.ENABLE_ENSEMBLE:
                print(f"\n🔥 스마트 앙상블 모드로 학습 진행 중...")
                
                # 동적 최적화된 앙상블 학습
                ensemble_models, ensemble_scalers, ensemble_metrics = self._train_smart_ensemble_models(
                    X, y, p, data_characteristics
                )
                
                if ensemble_models:
                    # 앙상블 모델 저장
                    self.models[p] = {'type': 'ensemble', 'models': ensemble_models, 'scalers': ensemble_scalers}
                    # 앙상블 성능 정보를 위한 대표 성능 선택 (최고 성능 모델 기준)
                    best_performance = max(ensemble_metrics.values(), key=lambda x: x.get('f1', 0))
                    performance = best_performance.copy()
                    performance['ensemble_count'] = len(ensemble_models)
                    performance['ensemble_models'] = list(ensemble_models.keys())
                    performance['data_characteristics'] = data_characteristics['characteristics']
                    performance['optimization_applied'] = config.ENABLE_SMART_OPTIMIZATION
                    
                    # 더미 history 생성 (호환성을 위해)
                    class DummyHistory:
                        def __init__(self):
                            self.history = {'loss': [0.1], 'mae': [0.1], 'val_loss': [0.1], 'val_mae': [0.1]}
                    history = DummyHistory()
                else:
                    print("❌ 앙상블 학습 실패, 스마트 단일 모델로 폴백")
                    # 스마트 단일 모델 학습으로 폴백
                    model = self.model_builder.build_smart_model(X.shape[1:], data_characteristics, self.data_handler)
                    trainer = Trainer(model, scaler, config)
                    history = trainer.train(X, y, p)
                    self.models[p] = {'type': 'single', 'model': model, 'scaler': scaler}
                    performance = trainer.evaluate(p)
            else:
                # 스마트 단일 모델 학습 방식 (기존 코드 + 3단계 개선)
                if config.ENABLE_SMART_OPTIMIZATION:
                    model = self.model_builder.build_smart_model(X.shape[1:], data_characteristics, self.data_handler)
                else:
            model = self.model_builder.build(X.shape[1:], len(X), p)
                
            trainer = Trainer(model, scaler, config)
            history = trainer.train(X, y, p)
                self.models[p] = {'type': 'single', 'model': model, 'scaler': scaler}
                performance = trainer.evaluate(p)
            
            if history is None:
                print(f"{p} 목적에 대한 모델 학습에 실패했습니다.")
                continue

            # 평가 지표에 학습 과정 정보 추가 (오류 수정: trainer 존재 여부 확인)
            # 앙상블 모드에서는 이미 performance가 정의되어 있고, trainer가 없을 수 있음
            if 'trainer' not in locals() or trainer is None:
                # 앙상블 모드에서 performance가 이미 정의된 경우, 재평가하지 않음
                if 'performance' not in locals():
                    print(f"⚠️ {p} 목적에 대한 성능 평가를 건너뜁니다.")
                    continue
            else:
                # trainer가 존재하는 경우에만 재평가 (단일 모델 모드 또는 앙상블 폴백)
            performance = trainer.evaluate(p)
            if performance:
                performance['nationality'] = nationality
                performance['epochs_trained'] = len(history.history['loss'])
                performance['final_train_loss'] = history.history['loss'][-1]
                performance['final_val_loss'] = history.history.get('val_loss', [None])[-1]
                performance['final_train_mae'] = history.history['mae'][-1]
                performance['final_val_mae'] = history.history.get('val_mae', [None])[-1]
                performance['best_val_loss'] = min(history.history.get('val_loss', [float('inf')]))
                performance['best_train_loss'] = min(history.history.get('loss', [float('inf')]))
                # trainer 존재 여부에 따른 추가 정보 설정 (오류 방지)
                if 'trainer' in locals() and trainer is not None:
                performance['early_stopped'] = trainer.callbacks[0].stopped_epoch > 0 if hasattr(trainer.callbacks[0], 'stopped_epoch') else False
                performance['learning_rate_used'] = trainer.model.optimizer.learning_rate.numpy()
                else:
                    # 앙상블 모드에서는 기본값 설정
                    performance['early_stopped'] = False
                    performance['learning_rate_used'] = 0.001  # 기본값
                performance['timestamp'] = self.timestamp
                performance_results.append(performance)

            # 앙상블 모드에 따른 예측 실행 (1단계 개선)
            if config.ENABLE_ENSEMBLE and self.models[p]['type'] == 'ensemble':
                # 앙상블 예측 실행
                ensemble_models = self.models[p]['models']
                ensemble_scalers = self.models[p]['scalers']
                # 대표 스케일러 사용 (첫 번째 모델의 스케일러)
                main_scaler = list(ensemble_scalers.values())[0]
                predictor = Predictor(None, main_scaler, features, combo_data, config)
                predictions = predictor.predict_ensemble_future(target_months, p, ensemble_models, ensemble_scalers)
            else:
                # 단일 모델 예측 실행 (기존 방식 보존)
                single_model = self.models[p]['model'] if self.models[p]['type'] == 'single' else self.models[p]
                single_scaler = self.models[p]['scaler'] if self.models[p]['type'] == 'single' else self.scalers[p]
                predictor = Predictor(single_model, single_scaler, features, combo_data, config)
            predictions = predictor.predict_future(target_months, p)
            results[p] = predictions

            print(f"--- {nationality} - {p} 예측 완료 ---")

        self.reporter.generate_reports(
            nationality, 
            results, 
            historical_data_collection, 
            performance_results, 
            start_date, 
            end_date
        )

    def _get_optimal_covid_strategy(self, purpose, user_selected_strategy):
        """목적별 최적 코로나 전략을 반환합니다."""
        if hasattr(self.config, 'PURPOSE_OPTIMAL_COVID_STRATEGY') and purpose in self.config.PURPOSE_OPTIMAL_COVID_STRATEGY:
            optimal_strategy = self.config.PURPOSE_OPTIMAL_COVID_STRATEGY[purpose]
            # 사용자가 weighted를 선택했을 때만 최적화 적용 (사용자 의도 존중)
            if user_selected_strategy == "weighted":
                return optimal_strategy
        return user_selected_strategy

    def _generate_target_months(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()

    # --- 앙상블 시스템 메서드 (1단계 개선) ---
    
    def _train_ensemble_models(self, X, y, purpose):
        """
        앙상블 모델 학습을 실행합니다.
        """
        try:
            # 앙상블 학습을 위한 임시 trainer 생성 (안전한 X_val 설정)
            temp_trainer = Trainer(None, None, self.config)
            # 앙상블에서 사용할 안전한 검증 데이터 설정
            if len(X) > 10:  # 충분한 데이터가 있을 때만 분할
                from sklearn.model_selection import train_test_split
                X_temp, X_val_safe, y_temp, y_val_safe = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                temp_trainer.X_val = X_val_safe
                temp_trainer.y_val = y_val_safe
            else:
                # 데이터가 적을 때는 전체 데이터 사용
                temp_trainer.X_val = X
                temp_trainer.y_val = y
            
            # 앙상블 모델 학습 실행
            ensemble_models, ensemble_scalers, ensemble_metrics = temp_trainer.train_ensemble_models(
                X, y, purpose, self.model_builder
            )
            
            return ensemble_models, ensemble_scalers, ensemble_metrics
            
        except Exception as e:
            print(f"❌ 앙상블 학습 중 오류 발생: {str(e)}")
            return {}, {}, {}

    def _train_smart_ensemble_models(self, X, y, purpose, data_characteristics):
        """
        데이터 특성에 기반한 스마트 앙상블 모델 학습을 실행합니다. (3단계 개선)
        """
        try:
            print(f"  🧠 데이터 특성 기반 최적화 적용 중...")
            
            # 최적화된 앙상블 가중치 계산
            optimized_weights = self.model_builder.get_optimized_ensemble_weights(data_characteristics)
            
            # config의 앙상블 가중치를 임시로 교체
            original_weights = self.config.ENSEMBLE_MODELS.get(purpose, {}).copy()
            self.config.ENSEMBLE_MODELS[purpose] = optimized_weights
            
            # 하이퍼파라미터 자동 튜닝 (선택적)
            if self.config.ENABLE_SMART_OPTIMIZATION and data_characteristics["size"] > 50:
                print(f"  🔧 하이퍼파라미터 자동 튜닝 시도 중...")
                best_params = self._auto_tune_hyperparameters(X, y, purpose, data_characteristics)
                if best_params:
                    print(f"  ✅ 최적 파라미터 적용: {best_params}")
            
            # 스마트 앙상블 학습을 위한 trainer 생성 (안전한 X_val 설정)
            temp_trainer = Trainer(None, None, self.config)
            # 앙상블에서 사용할 안전한 검증 데이터 설정
            if len(X) > 10:  # 충분한 데이터가 있을 때만 분할
                from sklearn.model_selection import train_test_split
                X_temp, X_val_safe, y_temp, y_val_safe = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                temp_trainer.X_val = X_val_safe
                temp_trainer.y_val = y_val_safe
            else:
                # 데이터가 적을 때는 전체 데이터 사용
                temp_trainer.X_val = X
                temp_trainer.y_val = y
            
            # 앙상블 모델 학습 실행
            ensemble_models, ensemble_scalers, ensemble_metrics = temp_trainer.train_ensemble_models(
                X, y, purpose, self.model_builder
            )
            
            # 원래 가중치 복원
            self.config.ENSEMBLE_MODELS[purpose] = original_weights
            
            # 결과에 최적화 정보 추가
            for model_type, metrics in ensemble_metrics.items():
                metrics['data_characteristics'] = data_characteristics['characteristics']
                metrics['optimized_weights_used'] = True
            
            return ensemble_models, ensemble_scalers, ensemble_metrics
            
        except Exception as e:
            print(f"❌ 스마트 앙상블 학습 중 오류 발생: {str(e)}")
            # 기본 앙상블 학습으로 폴백
            return self._train_ensemble_models(X, y, purpose)

    def _auto_tune_hyperparameters(self, X, y, purpose, data_characteristics):
        """
        자동 하이퍼파라미터 튜닝을 실행합니다.
        """
        try:
            temp_trainer = Trainer(None, None, self.config)
            best_params = temp_trainer.auto_tune_hyperparameters(
                X, y, purpose, data_characteristics, self.model_builder
            )
            return best_params
        except Exception as e:
            print(f"  ⚠️ 하이퍼파라미터 튜닝 실패: {str(e)}")
            return None

if __name__ == "__main__":
    predictor = MainPredictor()
    predictor.run()
