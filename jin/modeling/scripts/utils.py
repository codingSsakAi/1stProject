# -*- coding: utf-8 -*-
"""
유틸리티 함수 모음
Author: Jin
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import logging
from typing import Dict, List, Tuple, Any
import sys
import os

# 프로젝트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.config import *

warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def setup_plotting():
    """시각화 설정 초기화"""
    # 한글 폰트 설정
    setup_korean_fonts()

    plt.style.use(PLOT_CONFIG["style"])
    plt.rcParams["figure.figsize"] = PLOT_CONFIG["figsize"]
    plt.rcParams["figure.dpi"] = PLOT_CONFIG["dpi"]
    sns.set_palette(PLOT_CONFIG["color_palette"])


def setup_korean_fonts():
    """한글 폰트 설정 (맥/윈도우 호환)"""
    import matplotlib.font_manager as fm
    import matplotlib as mpl
    import platform

    # 폰트 캐시 강제 새로고침
    try:
        fm._load_fontmanager(try_read_cache=False)
    except:
        pass

    # 시스템별 한글 폰트 리스트
    system = platform.system()

    if system == "Darwin":  # macOS
        korean_fonts = [
            "Apple SD Gothic Neo",
            "AppleGothic",
            "Arial Unicode MS",
            "Nanum Gothic",
            "Apple LiGothic",
            "Helvetica",
        ]
    elif system == "Windows":  # Windows
        korean_fonts = [
            "Malgun Gothic",
            "맑은 고딕",
            "NanumGothic",
            "Nanum Gothic",
            "Arial Unicode MS",
            "굴림",
            "돋움",
        ]
    else:  # Linux 등
        korean_fonts = [
            "NanumGothic",
            "Nanum Gothic",
            "Arial Unicode MS",
            "DejaVu Sans",
            "Liberation Sans",
        ]

    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 첫 번째로 사용 가능한 한글 폰트 선택
    selected_font = None
    for font in korean_fonts:
        if font in available_fonts:
            selected_font = font
            break

        # 강제 폰트 설정
    if selected_font:
        # 여러 방법으로 폰트 설정
        plt.rcParams["font.family"] = [selected_font]
        current_sans_serif = plt.rcParams.get("font.sans-serif", [])
        if current_sans_serif:
            plt.rcParams["font.sans-serif"] = [selected_font] + list(current_sans_serif)
        else:
            plt.rcParams["font.sans-serif"] = [selected_font]
        logger.info(f"✅ 한글 폰트 설정: {selected_font}")
    else:
        # macOS 기본 폰트 강제 설정
        if system == "Darwin":
            plt.rcParams["font.family"] = ["Apple SD Gothic Neo"]
            logger.info("✅ macOS 기본 한글 폰트 설정: Apple SD Gothic Neo")
        else:
            plt.rcParams["font.family"] = ["DejaVu Sans"]
            logger.warning("⚠️ 기본 폰트 사용")

    # 추가 설정
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16

    # 폰트 적용 확인을 위한 테스트
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, "한글테스트", fontsize=12)
        plt.close(fig)
        logger.info("✅ 한글 폰트 적용 테스트 통과")
    except Exception as e:
        logger.warning(f"⚠️ 한글 폰트 테스트 실패: {e}")


def ensure_korean_font():
    """개별 그래프에서 한글 폰트 강제 적용"""
    import platform

    system = platform.system()
    if system == "Darwin":
        plt.rcParams["font.family"] = ["Apple SD Gothic Neo"]
    elif system == "Windows":
        plt.rcParams["font.family"] = ["Malgun Gothic"]

    plt.rcParams["axes.unicode_minus"] = False


def set_random_seed(seed: int = RANDOM_SEED):
    """랜덤 시드 설정"""
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """평가 지표 계산"""
    metrics = {}

    # MAE
    metrics["mae"] = mean_absolute_error(y_true, y_pred)

    # MSE
    metrics["mse"] = mean_squared_error(y_true, y_pred)

    # RMSE
    metrics["rmse"] = np.sqrt(metrics["mse"])

    # MAPE
    mask = y_true != 0  # 0으로 나누기 방지
    if np.any(mask):
        metrics["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics["mape"] = np.inf

    # R²
    metrics["r2"] = r2_score(y_true, y_pred)

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "모델 성능"):
    """평가 지표 출력"""
    print(f"\n📊 {title}")
    print("=" * 40)
    for metric, value in metrics.items():
        if metric == "mape":
            print(f"{metric.upper()}: {value:.2f}%")
        elif metric == "r2":
            print(f"{metric.upper()}: {value:.4f}")
        else:
            print(f"{metric.upper()}: {value:.2f}")
    print("=" * 40)


def prepare_lstm_data(
    df: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH, target_col: str = TARGET_COLUMN
) -> Tuple[np.ndarray, np.ndarray]:
    """LSTM용 시퀀스 데이터 준비"""

    # 국적-목적별 그룹화
    grouped = df.groupby(["국적", "목적"])

    X, y = [], []

    for name, group in grouped:
        group_sorted = group.sort_values(["연도", "월"])

        # 시퀀스 길이보다 긴 그룹만 처리
        if len(group_sorted) > sequence_length:
            values = group_sorted[FEATURE_COLUMNS].values
            targets = group_sorted[target_col].values

            # 시퀀스 생성
            for i in range(len(values) - sequence_length):
                X.append(values[i : i + sequence_length])
                y.append(targets[i + sequence_length])

    return np.array(X), np.array(y)


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """카테고리 특성 인코딩"""
    df_encoded = df.copy()
    encoders = {}

    for col in CATEGORY_COLUMNS:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    return df_encoded, encoders


def scale_features(
    X_train: np.ndarray, X_val: np.ndarray = None, X_test: np.ndarray = None
) -> Tuple[np.ndarray, ...]:
    """특성 스케일링"""
    scaler = StandardScaler()

    # 3D 배열을 2D로 변환하여 스케일링
    if X_train.ndim == 3:
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
    else:
        X_train_scaled = scaler.fit_transform(X_train)

    results = [X_train_scaled, scaler]

    if X_val is not None:
        if X_val.ndim == 3:
            n_samples, n_timesteps, n_features = X_val.shape
            X_val_2d = X_val.reshape(-1, n_features)
            X_val_scaled = scaler.transform(X_val_2d)
            X_val_scaled = X_val_scaled.reshape(n_samples, n_timesteps, n_features)
        else:
            X_val_scaled = scaler.transform(X_val)
        results.append(X_val_scaled)

    if X_test is not None:
        if X_test.ndim == 3:
            n_samples, n_timesteps, n_features = X_test.shape
            X_test_2d = X_test.reshape(-1, n_features)
            X_test_scaled = scaler.transform(X_test_2d)
            X_test_scaled = X_test_scaled.reshape(n_samples, n_timesteps, n_features)
        else:
            X_test_scaled = scaler.transform(X_test)
        results.append(X_test_scaled)

    return tuple(results)


def plot_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "예측 결과", save_path: str = None
):
    """예측 결과 시각화"""
    plt.figure(figsize=PLOT_CONFIG["figsize"])

    # 시계열 플롯
    plt.subplot(2, 1, 1)
    plt.plot(y_true, label="실제값", alpha=0.7)
    plt.plot(y_pred, label="예측값", alpha=0.7)
    plt.title(f"{title} - 시계열 비교")
    plt.xlabel("시간")
    plt.ylabel("입국자수")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 산점도
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.title(f"{title} - 실제값 vs 예측값")
    plt.xlabel("실제값")
    plt.ylabel("예측값")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
        logger.info(f"그래프 저장: {save_path}")

    plt.show()


def save_results(results: Dict[str, Any], filename: str):
    """결과 저장"""
    save_path = RESULTS_DIR / filename

    if filename.endswith(".csv"):
        if isinstance(results, pd.DataFrame):
            results.to_csv(save_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(results).to_csv(save_path, index=False, encoding="utf-8-sig")
    else:
        import pickle

        with open(save_path, "wb") as f:
            pickle.dump(results, f)

    logger.info(f"결과 저장: {save_path}")


def load_results(filename: str) -> Any:
    """결과 불러오기"""
    load_path = RESULTS_DIR / filename

    if filename.endswith(".csv"):
        return pd.read_csv(load_path, encoding="utf-8-sig")
    else:
        import pickle

        with open(load_path, "rb") as f:
            return pickle.load(f)


# 초기 설정 실행
setup_plotting()
set_random_seed()

logger.info("✅ 유틸리티 함수 로드 완료")
