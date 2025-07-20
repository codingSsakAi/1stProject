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
            combo_data = self.data_handler.get_data_for_purpose(nationality, p, covid_strategy)
            if combo_data.empty or len(combo_data) < config.LSTM_SEQUENCE_LENGTH_LARGE_DATA:
                print(f"데이터가 부족하여 ({len(combo_data)}개) 건너뜁니다.")
                continue
            
            historical_data_collection[p] = combo_data

            features = self.feature_engineer.create_features(combo_data)
            
            # DataHandler를 통해 스케일링 및 시퀀스 생성
            X, y, scaler = self.data_handler.preprocess_for_model(features, p)
            self.scalers[p] = scaler # 목적별 스케일러 저장

            if len(X) == 0:
                print("시퀀스 생성에 실패하여 건너뜁니다.")
                continue

            model = self.model_builder.build(X.shape[1:], len(X), p)
            trainer = Trainer(model, scaler, config)
            history = trainer.train(X, y, p)
            self.models[p] = model # 목적별 모델 저장
            
            if history is None:
                print(f"{p} 목적에 대한 모델 학습에 실패했습니다.")
                continue

            # 평가 지표에 학습 과정 정보 추가
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
                performance['early_stopped'] = trainer.callbacks[0].stopped_epoch > 0 if hasattr(trainer.callbacks[0], 'stopped_epoch') else False
                performance['learning_rate_used'] = trainer.model.optimizer.learning_rate.numpy()
                performance['timestamp'] = self.timestamp
                performance_results.append(performance)

            predictor = Predictor(model, scaler, features, combo_data, config)
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

    def _generate_target_months(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()

if __name__ == "__main__":
    predictor = MainPredictor()
    predictor.run()
