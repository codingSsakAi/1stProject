import pandas as pd

# 예측 csv 파일 읽기
df = pd.read_csv('../data/forecast_results_2025-06_2026-12.csv')   # 파일명/경로를 실제 파일명으로 변경

# 월-연 변환용 딕셔너리
month_map = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

def parse_month_yy(s):
    # 이미 yyyy-mm 형태면 그대로 반환
    if '-' in s and len(s) == 7 and s[:4].isdigit():
        return s
    try:
        m, y = s.split('-')
        # 월 약자가 올바르면 변환
        if m in month_map:
            year = '20' + y if len(y) == 2 else y
            return f"{year}-{month_map[m]}"
        else:
            return s  # 예외적 입력은 그대로 반환
    except Exception:
        return s  # split 불가 등 모든 예외는 원본 유지

# 변환 적용
df['연월'] = df['연월'].apply(parse_month_yy)

print(df['연월'].head(10)) 

# 컬럼 순서와 이름을 서비스에 맞게
df_save = df[['국가','목적','연월','예측입국자수','r2','mape','confidence']]
df_save.to_csv('../data/2025-06_2026-12_예측값.csv', index=False, encoding='utf-8-sig')
