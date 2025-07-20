
import os
import platform
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

def setup_gpu():
    """GPU 설정을 최적화합니다."""
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        if not physical_devices:
            print("[경고] GPU 미탐지, CPU로 실행합니다.")
            return

        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if platform.processor() != "arm" and "Apple" not in str(platform.processor()):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[최적화] Mixed precision 활성화 (학습 속도 향상)")
        else:
            print("[M1/M2 Mac] Mixed precision 비활성화 (안정성 우선)")

    except Exception as e:
        print(f"[GPU 설정 에러] {e}")

def set_environment():
    """TensorFlow 및 Matplotlib 환경을 설정합니다."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
    
    if platform.system() == 'Darwin':
        plt.rcParams["font.family"] = "AppleGothic"
    
    plt.rcParams["axes.unicode_minus"] = False

def create_timestamped_results_dir(base_dir):
    """타임스탬프 기반의 결과 저장 디렉토리를 생성합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[디렉토리] 결과 저장 디렉토리 생성: {results_dir}")
    return results_dir, timestamp

class SmartCountryMapper:
    """지능형 국적 매핑 클래스"""
    def __init__(self, data_nationalities=None):
        self.data_nationalities = data_nationalities or []
        self.basic_mapping = {
            "중국": ["china", "cn", "prc"], "일본": ["japan", "jp", "nippon"],
            "대만": ["taiwan", "tw", "formosa"], "태국": ["thailand", "th", "thai"],
            "베트남": ["vietnam", "vn"], "필리핀": ["philippines", "ph"],
            "말레이시아": ["malaysia", "my"], "싱가포르": ["singapore", "sg"],
            "인도네시아": ["indonesia", "id"], "인도": ["india", "in"],
            "몽골": ["mongolia", "mn"], "네팔": ["nepal", "np"],
            "미국": ["usa", "us", "america", "united states"],
            "영국": ["uk", "gb", "britain", "england"], "독일": ["germany", "de"],
            "프랑스": ["france", "fr"], "이탈리아": ["italy", "it"],
            "스페인": ["spain", "es"], "호주": ["australia", "au"],
            "캐나다": ["canada", "ca"], "러시아": ["russia", "ru"],
            "브라질": ["brazil", "br"], "멕시코": ["mexico", "mx"],
            "터키": ["turkey", "tr"], "이집트": ["egypt", "eg"],
        }

    def find_nationality(self, user_input):
        """사용자 입력으로부터 국적을 찾습니다."""
        user_input_lower = user_input.lower().strip()
        for kor_name, aliases in self.basic_mapping.items():
            if user_input_lower in [kor_name.lower()] + aliases:
                return kor_name
        
        for nat in self.data_nationalities:
            if user_input_lower in nat.lower():
                return nat
        
        return None
