
import numpy as np
import pandas as pd
import tensorflow as tf

# XGBoost 임포트 (2단계 개선)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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
        sequence_length = self._get_sequence_length(len(self.features_data), purpose)
        current_sequence = self.features_data.tail(sequence_length).values
        
        # 역사적 변동성 계산 (자연스러운 변동을 위해)
        historical_values = np.expm1(self.features_data["입국자수"].tail(24))  # 최근 24개월
        historical_volatility = historical_values.std() / max(historical_values.mean(), 1)
        volatility_factor = min(max(historical_volatility * 0.3, 0.08), 0.25)  # 8%~25% 범위
        
        # 월별 계절성 패턴 계산 (자연스러운 월별 변동을 위해)
        seasonal_factors = {}
        if len(self.combo_data) >= 24:  # 최소 2년 데이터가 있을 때만
            for month in range(1, 13):
                month_data = self.combo_data[self.combo_data["월"] == month]["입국자수"]
                if len(month_data) > 0:
                    # 해당 월의 평균 vs 전체 평균 비율
                    month_avg = np.expm1(month_data.mean())
                    overall_avg = np.expm1(self.combo_data["입국자수"].mean())
                    seasonal_factors[month] = month_avg / max(overall_avg, 1)
                else:
                    seasonal_factors[month] = 1.0
        
        # current_sequence의 유효성 검사 (행 개수)
        if current_sequence.shape[0] < sequence_length:
            print(f"오류: 예측을 위한 시퀀스 길이가 부족합니다. 필요한 길이: {sequence_length}, 현재 길이: {current_sequence.shape[0]}")
            return [] # 예측을 수행할 수 없으므로 빈 리스트 반환

        print(f"[DEBUG] Scaling 전 current_sequence shape: {current_sequence.shape}")
        current_sequence = self.scaler.transform(current_sequence)

        # 새로운 유효성 검사: 특성(feature)의 개수 확인
        if current_sequence.shape[1] == 0:
            print(f"오류: 예측을 위한 특성(feature)의 개수가 0입니다. 모델 입력에 문제가 발생할 수 있습니다.")
            return [] # 예측을 수행할 수 없으므로 빈 리스트 반환

        print(f"[DEBUG] Scaling 후 current_sequence shape: {current_sequence.shape}")
        print(f"[DEBUG] sequence_length: {sequence_length}")

        predictions_with_log = []
        # 마지막 실제 값은 feature 데이터에서 가져와야 하며, 이는 이미 로그 변환된 값입니다.
        last_actual_log_value = self.features_data["입국자수"].iloc[-1]

        for i, target_month in enumerate(target_months):
            # TensorFlow predict 호출: 진행사항 표시
            import tensorflow as tf
            input_tensor = tf.convert_to_tensor(current_sequence.reshape(1, sequence_length, -1), dtype=tf.float32)
            pred_scaled = self.model.predict(input_tensor, verbose=2, batch_size=1)[0, 0]
            
            # 역스케일링하여 로그 스케일의 예측값 획득
            pred_log_value = self._inverse_scale_single(pred_scaled)
            
            # 연속성 적용 (로그 스케일에서) - 변동성 개선을 위해 혼합 비율 조정
            if i == 0:
                pred_log_value = (pred_log_value * 0.6) + (last_actual_log_value * 0.4)
            else:
                # 이전 예측의 로그 값을 사용
                prev_log_value = np.log1p(predictions_with_log[-1]["value"])
                pred_log_value = (pred_log_value * 0.75) + (prev_log_value * 0.25)

            # 원래 스케일로 변환
            pred_value = np.expm1(pred_log_value)
            pred_value *= (1 + np.random.normal(0, volatility_factor)) # 역사적 변동성 기반 변동 적용
            
            # 월별 계절성 패턴 적용 (자연스러운 월별 변동을 위해)
            target_month_num = int(target_month.split('-')[1])  # "2026-01"에서 1 추출
            if seasonal_factors and target_month_num in seasonal_factors:
                seasonal_factor = seasonal_factors[target_month_num]
                # 계절성 효과를 부드럽게 적용 (너무 강하지 않게)
                pred_value *= (0.7 + 0.3 * seasonal_factor)
            
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

    def _get_sequence_length(self, data_size, purpose=None):
        """데이터 크기와 목적에 따라 시퀀스 길이를 반환합니다."""
        # 목적별 시퀀스 길이가 있으면 우선 사용 (학습 시와 일치)
        if purpose and hasattr(self.config, 'PURPOSE_SPECIFIC_SEQUENCE_LENGTH') and purpose in self.config.PURPOSE_SPECIFIC_SEQUENCE_LENGTH:
            return self.config.PURPOSE_SPECIFIC_SEQUENCE_LENGTH[purpose]
        
        # 기존 로직 (fallback)
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
    
    # --- 앙상블 예측 시스템 메서드들 (1단계 개선) ---
    
    def predict_ensemble_future(self, target_months, purpose, ensemble_models, ensemble_scalers):
        """
        앙상블 모델을 사용하여 미래 값을 예측합니다.
        기존 단일 모델 예측 로직은 그대로 유지하고, 새로운 앙상블 기능을 추가합니다.
        """
        print(f"\n🎯 앙상블 예측 시작 (목적: {purpose}, 예측 기간: {len(target_months)}개월)")
        
        if not ensemble_models:
            print("❌ 앙상블 모델이 없습니다. 단일 모델 예측을 사용하세요.")
            return []
        
        # 앙상블 설정 가져오기
        ensemble_config = self.config.ENSEMBLE_MODELS.get(purpose, self.config.ENSEMBLE_MODELS["관광"])
        smoothing_weight = self.config.ENSEMBLE_SMOOTHING_WEIGHTS.get(purpose, 0.75)
        
        # 시퀀스 길이 결정
        sequence_length = self._get_sequence_length(len(self.features_data), purpose)
        current_sequence = self.features_data.tail(sequence_length).values
        
        # 시퀀스 유효성 검사
        if current_sequence.shape[0] < sequence_length:
            print(f"❌ 시퀀스 길이 부족: 필요 {sequence_length}, 현재 {current_sequence.shape[0]}")
            return []
        
        # 역사적 변동성 및 계절성 패턴 계산 (기존 로직 재사용)
        historical_values = np.expm1(self.features_data["입국자수"].tail(24))
        historical_volatility = historical_values.std() / max(historical_values.mean(), 1)
        volatility_factor = min(max(historical_volatility * 0.3, 0.08), 0.25)
        
        seasonal_factors = self._calculate_seasonal_factors()
        
        # 앙상블 예측 수행
        ensemble_predictions = []
        
        for target_month in target_months:
            monthly_predictions = []
            
            # 각 모델별로 예측 수행
            for model_type, weight in ensemble_config.items():
                if model_type in ensemble_models and weight > 0:
                    try:
                        # 개별 모델 예측
                        model = ensemble_models[model_type]
                        scaler = ensemble_scalers[model_type]
                        
                        single_prediction = self._predict_single_model(
                            model, scaler, current_sequence, target_month,
                            volatility_factor, seasonal_factors
                        )
                        
                        # 가중치 적용
                        weighted_prediction = single_prediction * weight
                        monthly_predictions.append(weighted_prediction)
                        
                    except Exception as e:
                        print(f"⚠️ {model_type} 모델 예측 실패: {str(e)}")
                        continue
            
            if monthly_predictions:
                # 가중 평균으로 최종 예측값 계산
                ensemble_prediction = sum(monthly_predictions)
                
                # 앙상블 후처리 적용 (스무딩)
                if len(ensemble_predictions) > 0:
                    prev_prediction = ensemble_predictions[-1]
                    ensemble_prediction = (smoothing_weight * ensemble_prediction + 
                                         (1 - smoothing_weight) * prev_prediction)
                
                ensemble_predictions.append(ensemble_prediction)
                
                # 다음 예측을 위해 시퀀스 업데이트
                current_sequence = self._update_sequence_for_next_prediction(
                    current_sequence, ensemble_prediction, target_month
                )
            else:
                print(f"❌ {target_month} 예측 실패: 사용 가능한 모델이 없습니다.")
                ensemble_predictions.append(0)
        
        # 최종 결과 변환 (로그 스케일에서 원래 스케일로)
        final_predictions = [max(0, np.expm1(pred)) for pred in ensemble_predictions]
        
        print(f"✅ 앙상블 예측 완료: {len(final_predictions)}개월")
        return final_predictions
    
    def _predict_single_model(self, model, scaler, current_sequence, target_month, 
                            volatility_factor, seasonal_factors):
        """
        개별 모델로 단일 월 예측을 수행합니다.
        XGBoost와 TensorFlow 모델을 모두 지원합니다. (2단계 개선)
        """
        # 모델 타입별 시퀀스 스케일링 (차원 오류 수정)
        if XGBOOST_AVAILABLE and isinstance(model, xgb.XGBRegressor):
            # XGBoost 모델: 시퀀스 완전 평탄화 후 스케일링
            sequence_flat = current_sequence.reshape(1, -1)  # (24,34) → (1,816)
            X_flattened = scaler.transform(sequence_flat)
            prediction_scaled = model.predict(X_flattened)[0]
            
        else:
            # TensorFlow 모델: 모델 타입별 적절한 스케일링 판단
            # DENSE 모델인지 확인 (Sequential 모델이고 첫 번째 레이어가 Dense인 경우)
            is_dense_model = (hasattr(model, 'layers') and len(model.layers) > 0 and 
                            hasattr(model.layers[0], '__class__') and 
                            'Dense' in str(model.layers[0].__class__))
            
            if is_dense_model:
                # DENSE 모델: 시퀀스 완전 평탄화 후 스케일링  
                sequence_flat = current_sequence.reshape(1, -1)  # (24,34) → (1,816)
                scaled_sequence = scaler.transform(sequence_flat)
            else:
                # LSTM/GRU/LSTM_ATTENTION: 3차원 데이터 스케일링
                if len(current_sequence.shape) == 3:
                    original_shape = current_sequence.shape
                    sequence_reshaped = current_sequence.reshape(-1, current_sequence.shape[-1])
                    scaled_flat = scaler.transform(sequence_reshaped)
                    scaled_sequence = scaled_flat.reshape(original_shape)
                    # 배치 차원 추가
                    scaled_sequence = scaled_sequence.reshape(1, scaled_sequence.shape[0], scaled_sequence.shape[1])
                else:
                    # 2차원 → 3차원으로 변환
                    scaled_sequence = scaler.transform(current_sequence)
                    scaled_sequence = scaled_sequence.reshape(1, scaled_sequence.shape[0], scaled_sequence.shape[1])
            
            # TensorFlow predict 호출: 진행사항 표시
            import tensorflow as tf
            scaled_tensor = tf.convert_to_tensor(scaled_sequence, dtype=tf.float32)
            prediction_scaled = model.predict(scaled_tensor, verbose=2, batch_size=1).flatten()[0]
        
        # 변동성 및 계절성 적용
        target_date = pd.to_datetime(target_month + "-01")
        month = target_date.month
        seasonal_factor = seasonal_factors.get(month, 1.0)
        
        # 자연스러운 노이즈 추가 (XGBoost는 조금 적게)
        noise_factor = volatility_factor * 0.3 if XGBOOST_AVAILABLE and isinstance(model, xgb.XGBRegressor) else volatility_factor * 0.5
        noise = np.random.normal(0, noise_factor)
        
        # 최종 예측값 계산
        final_prediction = prediction_scaled * seasonal_factor + noise
        
        return final_prediction
    
    def _calculate_seasonal_factors(self):
        """
        계절성 요인을 계산합니다. (기존 로직 재사용)
        """
        seasonal_factors = {}
        if len(self.combo_data) >= 24:
            for month in range(1, 13):
                month_data = self.combo_data[self.combo_data["월"] == month]["입국자수"]
                if len(month_data) > 0:
                    month_avg = np.expm1(month_data.mean())
                    overall_avg = np.expm1(self.combo_data["입국자수"].mean())
                    seasonal_factors[month] = month_avg / max(overall_avg, 1)
                else:
                    seasonal_factors[month] = 1.0
        else:
            # 기본 계절성 패턴 (관광 중심)
            seasonal_factors = {1: 0.8, 2: 0.7, 3: 1.1, 4: 1.2, 5: 1.1, 6: 1.0,
                              7: 1.3, 8: 1.4, 9: 1.1, 10: 1.2, 11: 1.0, 12: 0.9}
        
        return seasonal_factors
    
    def _update_sequence_for_next_prediction(self, current_sequence, prediction, target_month):
        """
        다음 예측을 위해 시퀀스를 업데이트합니다.
        """
        # 새로운 특성 행 생성 (기존 로직 재사용)
        new_row = self._create_new_feature_row(prediction, target_month)
        
        # 시퀀스 업데이트 (한 행 앞으로 이동하고 새 행 추가)
        updated_sequence = np.vstack([current_sequence[1:], new_row.reshape(1, -1)])
        
        return updated_sequence
