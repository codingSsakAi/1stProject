import os
import zipfile
import pickle
import joblib

# 압축 해제
zip_path = '/mnt/data/model.zip'
extract_dir = '/mnt/data/model_unzip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# .pkl → .joblib 변환
for root, _, files in os.walk(extract_dir):
    for file in files:
        if file.endswith('.pkl'):
            pkl_path = os.path.join(root, file)
            joblib_path = pkl_path.replace('.pkl', '.joblib')
            try:
                with open(pkl_path, 'rb') as f:
                    obj = pickle.load(f)
                joblib.dump(obj, joblib_path)
                print(f'변환 완료: {pkl_path} -> {joblib_path}')
            except Exception as e:
                print(f'변환 실패: {pkl_path}, 에러: {e}')
