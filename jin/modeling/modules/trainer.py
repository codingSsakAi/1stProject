
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

        # XGBoost는 predict에서 verbose 미지원 및 데이터 차원 처리 (오류 수정)
        try:
            import xgboost as xgb
            if isinstance(self.model, xgb.XGBRegressor):
                # XGBoost는 2차원 데이터 필요
                if len(self.X_val.shape) == 3:
                    X_val_2d = self.X_val.reshape(self.X_val.shape[0], -1)
                else:
                    X_val_2d = self.X_val
                y_pred_scaled = self.model.predict(X_val_2d).flatten()
            else:
                # TensorFlow retracing 경고 해결: 안정적인 predict 호출 (간소화)
                import tensorflow as tf
                # TensorFlow 모델의 경우 일관된 shape 보장
                X_val_consistent = tf.convert_to_tensor(self.X_val, dtype=tf.float32)
                # 고정된 배치 크기로 예측 (진행사항 표시)
                y_pred_scaled = self.model.predict(X_val_consistent, verbose=2, batch_size=16).flatten()
        except (ImportError, Exception):
        y_pred_scaled = self.model.predict(self.X_val, verbose=2).flatten()
        y_true_rescaled, y_pred_rescaled = self._inverse_transform(self.y_val, y_pred_scaled)

        thresholds = self._get_performance_thresholds(len(self.X_val), purpose)
        
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

    def _get_performance_thresholds(self, data_size, purpose=None):
        """데이터 크기와 목적에 따라 성능 기준을 반환합니다."""
        # 기본 기준값 가져오기
        if data_size < 100:
            base_thresholds = {k: v * 3 for k, v in self.config.BASE_PERFORMANCE_THRESHOLDS.items()}
        elif data_size < 200:
            base_thresholds = {k: v * 2 for k, v in self.config.BASE_PERFORMANCE_THRESHOLDS.items()}
        else:
            base_thresholds = self.config.BASE_PERFORMANCE_THRESHOLDS.copy()
        
        # 목적별 차별화된 기준값 적용 (우선순위)
        if purpose and hasattr(self.config, 'PURPOSE_SPECIFIC_THRESHOLDS') and purpose in self.config.PURPOSE_SPECIFIC_THRESHOLDS:
            purpose_thresholds = self.config.PURPOSE_SPECIFIC_THRESHOLDS[purpose]
            print(f"[{purpose}] 목적별 성능 기준 적용: R²≥{purpose_thresholds.get('r2_score_기준값', 'N/A')}, F1≥{purpose_thresholds.get('f1_score_기준값', 'N/A')}")
            # 목적별 설정으로 업데이트
            base_thresholds.update(purpose_thresholds)
        
        return base_thresholds

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
        
        # F1-score 계산 및 등급 설정 (동적 허용 오차율 적용)
        dynamic_tolerance = self._calculate_dynamic_tolerance(y_true, purpose)
        metrics["f1_score"] = self._calculate_f1_score(y_true, y_pred, dynamic_tolerance)
        metrics["f1_score_등급"] = "양호" if metrics["f1_score"] >= metrics["f1_score_기준값"] else "미흡"

        return metrics

    def _calculate_dynamic_tolerance(self, y_true, purpose):
        """데이터 특성에 맞는 동적 허용 오차율을 계산합니다."""
        # 데이터의 변동성 계산
        cv = np.std(y_true) / max(np.mean(y_true), 1)  # 변동계수
        
        # 기본 허용 오차율
        base_tolerance = self.config.F1_SCORE_TOLERANCE_PERCENTAGE
        
        # 변동성에 따른 조정 (변동성이 클수록 허용 오차율 증가)
        if cv > 0.5:
            tolerance_multiplier = 1.5  # 변동성이 큰 경우
        elif cv > 0.3:
            tolerance_multiplier = 1.2  # 중간 변동성
        else:
            tolerance_multiplier = 1.0  # 낮은 변동성
        
        # 목적별 조정 (목적에 따라 예측 난이도가 다름)
        if purpose == "관광":
            purpose_multiplier = 1.0    # 관광은 패턴이 비교적 명확
        elif purpose == "상용":
            purpose_multiplier = 1.1    # 상용은 약간 복잡
        else:
            purpose_multiplier = 1.2    # 공용, 유학연수는 더 복잡
        
        # 데이터 크기에 따른 조정 (적을수록 관대하게)
        data_size = len(y_true)
        if data_size < 30:
            size_multiplier = 1.3
        elif data_size < 50:
            size_multiplier = 1.1
        else:
            size_multiplier = 1.0
        
        final_tolerance = base_tolerance * tolerance_multiplier * purpose_multiplier * size_multiplier
        return min(max(final_tolerance, 5.0), 30.0)  # 5%~30% 범위로 제한

    def _calculate_f1_score(self, y_true, y_pred, tolerance_percentage):
        """
        회귀 문제에서 F1-score를 계산합니다.
        예측값과 실제값의 상대 오차(백분율)가 tolerance_percentage 이내일 경우 '정답'으로 간주합니다.
        """
        # 0으로 나누는 것을 방지하기 위해 실제값에 작은 값 추가
        y_true_safe = np.maximum(y_true, 1e-6) 
        
        # 상대 오차 계산
        relative_error = np.abs((y_pred - y_true) / y_true_safe)

        # '정답' 여부 판단 (True = 정답, False = 오답)
        is_correct = relative_error <= (tolerance_percentage / 100.0)

        # True Positives, False Positives, False Negatives 계산
        # 여기서는 '정답'을 긍정 클래스로 간주합니다.
        # TP: 실제 정답이고 예측도 정답 (is_correct가 True)
        # FP: 실제 오답인데 예측은 정답 (is_correct가 True인데 실제로는 오차율 초과) - 이 경우 정의가 어려움
        # FN: 실제 정답인데 예측은 오답 (is_correct가 False인데 실제로는 오차율 이내) - 이 경우 정의가 어려움

        # 회귀 문제에서 F1-score를 적용하기 위해 이진 분류 문제로 변환
        # '정답'을 1, '오답'을 0으로 간주
        y_true_binary = np.ones_like(y_true, dtype=int) # 모든 실제값을 '정답'으로 가정
        y_pred_binary = is_correct.astype(int) # 예측이 허용 오차 내이면 '정답'

        # precision, recall, f1_score 계산
        # sklearn.metrics.f1_score를 사용하기 위해 import 필요
        from sklearn.metrics import f1_score, precision_score, recall_score

        # 'pos_label=1'은 '정답'을 긍정 클래스로 간주함을 의미
        # 'zero_division=0'은 0으로 나눌 때 0을 반환하도록 설정
        precision = precision_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
        
        return f1

    # --- 앙상블 시스템 메서드들 (1단계 개선) ---
    
    def train_ensemble_models(self, X, y, purpose, model_builder):
        """
        여러 모델을 학습하여 앙상블 시스템을 구축합니다.
        기존 단일 모델 학습 로직은 그대로 유지하고, 새로운 앙상블 기능을 추가합니다.
        """
        print(f"\n🔥 앙상블 모델 학습 시작 (목적: {purpose})")
        
        # 앙상블에 사용할 모델들과 가중치 가져오기
        ensemble_config = self.config.ENSEMBLE_MODELS.get(purpose, self.config.ENSEMBLE_MODELS["관광"])
        
        # 여러 모델 학습 결과를 저장할 딕셔너리
        ensemble_models = {}
        ensemble_scalers = {}
        ensemble_histories = {}
        ensemble_metrics = {}
        
        # 각 모델별로 학습 진행
        for model_type, weight in ensemble_config.items():
            if weight > 0:  # 가중치가 0보다 큰 모델만 학습
                print(f"\n📊 {model_type} 모델 학습 중... (가중치: {weight:.1%})")
                
                try:
                    # 모델별 개별 빌드 및 학습
                    model, scaler = self._build_and_train_single_model(
                        X, y, purpose, model_type, model_builder
                    )
                    
                    # 개별 모델 평가 (데이터 타입별 안전 처리)
                    individual_trainer = Trainer(model, scaler, self.config)
                    
                    # 모델 타입에 따른 검증 데이터 설정
                    if model_type in ["DENSE", "XGBOOST"]:
                        # DENSE/XGBoost: 2차원 데이터 사용
                        if len(self.X_val.shape) == 3:
                            X_val_2d = self.X_val.reshape(self.X_val.shape[0], -1)
                        else:
                            X_val_2d = self.X_val
                        individual_trainer.X_val = X_val_2d
                    else:
                        # LSTM/GRU/LSTM_ATTENTION: 3차원 데이터 사용
                        individual_trainer.X_val = self.X_val
                    
                    individual_trainer.y_val = self.y_val
                    metrics = individual_trainer.evaluate(purpose)
                    
                    # 결과 저장
                    ensemble_models[model_type] = model
                    ensemble_scalers[model_type] = scaler  
                    ensemble_metrics[model_type] = metrics
                    
                    print(f"✅ {model_type} 완료 - R²: {metrics.get('r2', 0):.3f}, F1: {metrics.get('f1', 0):.3f}")
                    
                except Exception as e:
                    print(f"❌ {model_type} 학습 실패: {str(e)}")
                    continue
        
        # 앙상블 성능 평가
        if len(ensemble_models) > 1:
            ensemble_performance = self._evaluate_ensemble_performance(
                ensemble_models, ensemble_scalers, purpose
            )
            print(f"\n🎯 앙상블 전체 성능 - R²: {ensemble_performance.get('r2', 0):.3f}, F1: {ensemble_performance.get('f1', 0):.3f}")
        
        return ensemble_models, ensemble_scalers, ensemble_metrics
    
    def _build_and_train_single_model(self, X, y, purpose, model_type, model_builder):
        """
        개별 모델을 빌드하고 학습합니다.
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # 모델 타입별 데이터 전처리 (오류 수정)
        if model_type == "DENSE":
            # DENSE 모델은 2차원 데이터 사용 (3차원 → 2차원 변환)
            if len(X.shape) == 3:
                # (samples, timesteps, features) → (samples, timesteps * features)
                X_flattened = X.reshape(X.shape[0], -1)
                print(f"    📊 DENSE용 데이터 변환: {X.shape} → {X_flattened.shape}")
            else:
                X_flattened = X
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_flattened)
            input_shape = (X_scaled.shape[1],)  # DENSE는 1차원 입력
            
        elif model_type == "XGBOOST":
            # XGBoost도 2차원 데이터 사용
            if len(X.shape) == 3:
                X_flattened = X.reshape(X.shape[0], -1)
                print(f"    📊 XGBoost용 데이터 변환: {X.shape} → {X_flattened.shape}")
            else:
                X_flattened = X
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_flattened)
            input_shape = None  # XGBoost는 input_shape 불필요
            
        else:
            # 기존 방식: LSTM, GRU, LSTM_ATTENTION (3차원 데이터 사용 - 오류 수정)
            scaler = MinMaxScaler()
            if len(X.shape) == 3:
                # 3차원 데이터 안전 처리: (samples, timesteps, features)
                print(f"    📊 3차원 데이터 스케일링: {X.shape}")
                original_shape = X.shape
                # (samples, timesteps, features) → (samples*timesteps, features)
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled_flat = scaler.fit_transform(X_reshaped)
                # 원래 3차원으로 복원
                X_scaled = X_scaled_flat.reshape(original_shape)
                input_shape = (X_scaled.shape[1], X_scaled.shape[2])  # (timesteps, features)
            else:
                # 2차원 데이터
                X_scaled = scaler.fit_transform(X)
                input_shape = (X_scaled.shape[1], 1)
        
        data_size = len(X_scaled)
        
        # 모델 타입별로 빌드
        if model_type == "LSTM_ATTENTION":
            model = model_builder._build_tourism_lstm_attention_model(input_shape, data_size)
        elif model_type == "LSTM":
            model = model_builder._build_single_lstm(input_shape)
        elif model_type == "GRU":
            model = model_builder._build_single_gru(input_shape)
        elif model_type == "DENSE":
            model = model_builder._build_simple_dense(input_shape)
        elif model_type == "XGBOOST":
            # XGBoost 모델 빌드 (2단계 개선)
            model = model_builder.build_xgboost(purpose, X_scaled.shape[1])
            if model is None:
                # XGBoost 사용 불가시 LSTM으로 폴백
                print("XGBoost 사용 불가, LSTM으로 폴백")
                model = model_builder._build_single_lstm(input_shape)
        else:
            # 기본값으로 LSTM 사용
            model = model_builder._build_single_lstm(input_shape)
        
        # 개별 모델 학습 (모델 타입별로 다른 학습 방식 적용)
        if model_type == "XGBOOST" and model_builder.is_xgboost_model(model):
            # XGBoost 모델 학습 (2단계 개선)
            X_train, X_val, y_train, y_val = self._split_data_for_xgboost(X_scaled, y)
            
            # XGBoost 학습 (2차원 데이터 사용 - early_stopping_rounds 경고 수정)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if len(X_val) > 0 else None,
                verbose=False
            )
            print(f"📊 XGBoost 학습 완료 - 트리 수: {model.n_estimators}")
            
        else:
            # TensorFlow 모델 학습 (기존 방식 - 컴파일 및 차원 오류 수정)
            # 모델이 컴파일되지 않은 경우 자동 컴파일
            if not hasattr(model, 'optimizer') or model.optimizer is None:
                learning_rate = 0.001
                from tensorflow.keras.optimizers import Adam
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
                print(f"    ✅ 모델 자동 컴파일 완료 (학습률: {learning_rate})")
            
            # DENSE 모델의 경우 직접 학습 (차원 문제 해결)
            if model_type == "DENSE":
                # DENSE 모델은 이미 2차원으로 변환된 X_scaled 사용
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                if len(X_train) > 0:
                    epochs = 50 if len(X_train) < 100 else 100
                    batch_size = 16 if len(X_train) < 100 else 32
                    
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
                        verbose=2
                    )
                    print(f"    ✅ DENSE 모델 직접 학습 완료 (epochs: {len(history.history['loss'])})")
            else:
                # 기존 방식: LSTM, GRU, LSTM_ATTENTION
                temp_trainer = Trainer(model, scaler, self.config)
                temp_trainer.train(X_scaled, y, purpose)
        
        return model, scaler
    
    def _evaluate_ensemble_performance(self, ensemble_models, ensemble_scalers, purpose):
        """
        앙상블 모델의 전체 성능을 평가합니다.
        """
        if self.X_val is None or len(self.X_val) == 0:
            return {}
        
        # 앙상블 예측 수행
        ensemble_config = self.config.ENSEMBLE_MODELS.get(purpose, self.config.ENSEMBLE_MODELS["관광"])
        ensemble_predictions = []
        
        for model_type, weight in ensemble_config.items():
            if model_type in ensemble_models and weight > 0:
                model = ensemble_models[model_type]
                scaler = ensemble_scalers[model_type]
                
                # 개별 모델 예측 (모델 타입별 데이터 형태 맞춤 - 오류 수정)
                if model_type in ["DENSE", "XGBOOST"]:
                    # DENSE/XGBoost: 2차원 데이터로 변환 후 스케일링
                    if len(self.X_val.shape) == 3:
                        X_val_2d = self.X_val.reshape(self.X_val.shape[0], -1)
                    else:
                        X_val_2d = self.X_val
                    X_val_scaled = scaler.transform(X_val_2d)
                else:
                    # LSTM/GRU/LSTM_ATTENTION: 3차원 데이터 스케일링
                    if len(self.X_val.shape) == 3:
                        original_shape = self.X_val.shape
                        X_val_reshaped = self.X_val.reshape(-1, self.X_val.shape[-1])
                        X_val_scaled_flat = scaler.transform(X_val_reshaped)
                        X_val_scaled = X_val_scaled_flat.reshape(original_shape)
                    else:
                        X_val_scaled = scaler.transform(self.X_val)
                
                try:
                    import xgboost as xgb
                    if isinstance(model, xgb.XGBRegressor):
                        pred_scaled = model.predict(X_val_scaled).flatten()
                    else:
                        # TensorFlow retracing 경고 해결: 안정적인 predict 호출
                        import tensorflow as tf
                        X_val_tensor = tf.convert_to_tensor(X_val_scaled, dtype=tf.float32)
                        pred_scaled = model.predict(X_val_tensor, verbose=2, batch_size=16).flatten()
                except (ImportError, Exception):
                    pred_scaled = model.predict(X_val_scaled, verbose=2).flatten()
                
                # 가중치 적용
                ensemble_predictions.append(pred_scaled * weight)
        
        if not ensemble_predictions:
            return {}
        
        # 가중 평균으로 최종 예측값 계산
        final_prediction = np.sum(ensemble_predictions, axis=0)
        
        # 성능 지표 계산
        thresholds = self._get_performance_thresholds(len(self.X_val), purpose)
        y_true_rescaled = np.expm1(self.y_val)
        y_pred_rescaled = np.expm1(final_prediction)
        
        metrics = self._calculate_metrics(y_true_rescaled, y_pred_rescaled, purpose, thresholds)
        
        return metrics
    
    # --- XGBoost 전용 메서드들 (2단계 개선) ---
    
    def _split_data_for_xgboost(self, X_scaled, y):
        """
        XGBoost를 위한 데이터 분할 및 변환
        XGBoost는 2차원 데이터를 사용하므로 시퀀스 데이터를 평탄화합니다.
        """
        from sklearn.model_selection import train_test_split
        
        # 3차원 시퀀스 데이터를 2차원으로 평탄화
        if len(X_scaled.shape) == 3:
            # (samples, timesteps, features) → (samples, timesteps * features)
            X_flattened = X_scaled.reshape(X_scaled.shape[0], -1)
        else:
            X_flattened = X_scaled
        
        # 시계열 데이터는 순서가 중요하므로 shuffle=False로 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_flattened, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        return X_train, X_val, y_train, y_val
    
    def evaluate_xgboost_model(self, model, scaler, purpose):
        """
        XGBoost 모델의 성능을 평가합니다.
        """
        if self.X_val is None or len(self.X_val) == 0:
            return {}
        
        # XGBoost용 데이터 준비
        X_val_flattened = self.X_val.reshape(self.X_val.shape[0], -1) if len(self.X_val.shape) == 3 else self.X_val
        
        # XGBoost 예측
        y_pred_scaled = model.predict(X_val_flattened)
        
        # 역변환 및 성능 계산
        y_true_rescaled = np.expm1(self.y_val)
        y_pred_rescaled = np.expm1(y_pred_scaled)
        
        thresholds = self._get_performance_thresholds(len(self.X_val), purpose)
        metrics = self._calculate_metrics(y_true_rescaled, y_pred_rescaled, purpose, thresholds)
        
        return metrics

    # --- 자동 하이퍼파라미터 튜닝 (3단계 개선) ---
    
    def auto_tune_hyperparameters(self, X, y, purpose, data_characteristics, model_builder):
        """
        데이터 특성에 기반하여 하이퍼파라미터를 자동으로 튜닝합니다.
        """
        if not self.config.ENABLE_SMART_OPTIMIZATION:
            return None
        
        print(f"\n🔧 자동 하이퍼파라미터 튜닝 시작...")
        
        # 데이터 크기에 따른 튜닝 범위 결정
        tuning_ranges = self._get_tuning_ranges_for_data(data_characteristics)
        
        # 간단한 그리드 서치 수행 (최대 3개 조합만 테스트)
        best_params = None
        best_score = float('-inf')
        
        param_combinations = self._generate_param_combinations(tuning_ranges, max_combinations=3)
        
        for i, params in enumerate(param_combinations):
            print(f"  🔍 조합 {i+1}/{len(param_combinations)} 테스트 중: {params}")
            
            try:
                score = self._evaluate_hyperparameters(X, y, params, purpose, model_builder, data_characteristics)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"    ✅ 새로운 최고 성능: {score:.3f}")
                else:
                    print(f"    📊 성능: {score:.3f}")
                    
            except Exception as e:
                print(f"    ❌ 조합 실패: {str(e)}")
                continue
        
        if best_params:
            print(f"  🏆 최적 파라미터: {best_params} (성능: {best_score:.3f})")
            return best_params
        else:
            print(f"  ⚠️ 튜닝 실패, 기본 파라미터 사용")
            return None
    
    def _get_tuning_ranges_for_data(self, data_characteristics):
        """
        데이터 특성에 따른 튜닝 범위를 결정합니다.
        """
        tuning_ranges = self.config.HYPERPARAMETER_TUNING_RANGES.copy()
        
        # 작은 데이터셋의 경우 파라미터 범위 축소
        if data_characteristics["size_category"] == "small":
            tuning_ranges["lstm_units"] = [32, 64]  # 큰 모델 제외
            tuning_ranges["batch_size"] = [16, 32]  # 작은 배치 크기
            tuning_ranges["learning_rate"] = [0.005, 0.01]  # 높은 학습률
        
        # 변동성이 높은 데이터의 경우 정규화 강화
        elif data_characteristics["volatility"] > self.config.DATA_ANALYSIS_THRESHOLDS["high_volatility"]:
            tuning_ranges["dropout"] = [0.3, 0.4, 0.5]  # 높은 드롭아웃
            tuning_ranges["learning_rate"] = [0.001, 0.005]  # 낮은 학습률
        
        return tuning_ranges
    
    def _generate_param_combinations(self, tuning_ranges, max_combinations=3):
        """
        제한된 수의 파라미터 조합을 생성합니다.
        """
        import itertools
        
        # 가장 중요한 파라미터들만 선택
        important_params = {}
        if "learning_rate" in tuning_ranges:
            important_params["learning_rate"] = tuning_ranges["learning_rate"][:2]  # 최대 2개만
        if "dropout" in tuning_ranges:
            important_params["dropout"] = tuning_ranges["dropout"][:2]  # 최대 2개만
        if "lstm_units" in tuning_ranges:
            important_params["lstm_units"] = tuning_ranges["lstm_units"][:2]  # 최대 2개만
        
        # 조합 생성
        param_names = list(important_params.keys())
        param_values = list(important_params.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
            
            if len(combinations) >= max_combinations:
                break
        
        return combinations
    
    def _evaluate_hyperparameters(self, X, y, params, purpose, model_builder, data_characteristics):
        """
        주어진 하이퍼파라미터로 모델을 학습하고 성능을 평가합니다.
        """
        from sklearn.model_selection import train_test_split
        
        # 데이터 분할 (작은 검증 세트로 빠른 평가)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        if len(X_train) < 10:  # 너무 작은 데이터는 건너뛰기
            return 0.0
        
        # 임시 모델 생성 (LSTM 기반)
        input_shape = X_train.shape[1:]
        model = model_builder._build_single_lstm(input_shape)
        
        # 하이퍼파라미터 적용 (컴파일 단계에서)
        learning_rate = params.get("learning_rate", 0.001)
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        # 빠른 학습 (적은 epoch)
        epochs = 5 if data_characteristics["size_category"] == "small" else 10
        batch_size = params.get("batch_size", 32)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            verbose=2  # 진행사항 표시
        )
        
        # 성능 평가 (validation loss 기준)
        if len(X_val) > 0:
            val_loss = min(history.history.get('val_loss', [float('inf')]))
            # 낮은 loss가 좋으므로 음수로 변환
            score = -val_loss
        else:
            # validation 데이터가 없는 경우 train loss 사용
            train_loss = min(history.history.get('loss', [float('inf')]))
            score = -train_loss
        
        return score
