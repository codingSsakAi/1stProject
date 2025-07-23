import pandas as pd
import os
from model import run_forecast

input_csv = '../data/외국인입국자_전처리완료_딥러닝용.csv'
df = pd.read_csv(input_csv)
국적목록 = sorted(df['국적'].unique())
목적목록 = sorted(df['목적'].unique())
target_periods = [
    (2025, list(range(6, 13))),   # 2025년 6~12월
    (2026, list(range(1, 13))),   # 2026년 1~12월
]

out_csv = '../data/forecast_results_2025-06_2026-12.csv'
first_write = not os.path.exists(out_csv)

# 1. 이미 저장된 조합 읽기 (중단/재시작용)
already_done = set()
if not first_write:
    prev = pd.read_csv(out_csv)
    already_done = set(zip(prev['국가'], prev['목적'], prev['연월']))
    print(f"▶ 이미 저장된 결과 건수: {len(already_done)}")

# 2. 전체 작업 개수 계산
jobs = []
for country in 국적목록:
    for purpose in 목적목록:
        for year, months in target_periods:
            ym_list = [f"{year}-{str(m).zfill(2)}" for m in months]
            jobs.append((country, purpose, year, months, ym_list))
total_jobs = len(jobs)
print(f"전체 연단위 예측 작업 수: {total_jobs}")

job_idx = 0
for country, purpose, year, months, ym_list in jobs:
    job_idx += 1

    # (1) 이번 연도의 모든 월이 이미 예측됐으면 통째로 skip
    if all((country, purpose, ym) in already_done for ym in ym_list if ym >= '2025-06'):
        print(f"[{job_idx}/{total_jobs}] {country} | {purpose} | {year}년 이미 전체 예측 완료 → 스킵")
        continue

    print(f"[{job_idx}/{total_jobs}] {country} | {purpose} | {year}년 예측 실행 중...")

    try:
        preds = run_forecast(country, purpose, year, months)
    except Exception as e:
        print(f"  [에러] {country}, {purpose}, {year}: {e}")
        continue

    rows = []
    if isinstance(preds, list):
        for rec in preds:
            # (에러 dict 등 예외 rec 스킵)
            if not (isinstance(rec, dict) and 'yms' in rec and 'values' in rec):
                print(f"  [스킵] 결과 없음: {country}, {purpose}, {year}: {rec}")
                continue
            for ym, val in zip(rec['yms'], rec['values']):
                # (2) 2025-06 이후, 중복 미포함만 저장
                if ym >= '2025-06' and (country, purpose, ym) not in already_done:
                    print(f"    > {country}/{purpose}/{ym}: {val}")
                    rows.append({
                        '국가': country,
                        '목적': purpose,
                        '연월': ym,
                        '예측입국자수': val,
                        'r2': rec.get('r2'),
                        'mape': rec.get('mape'),
                        'confidence': rec.get('confidence')
                    })
    else:
        print(f"  [스킵] 결과 없음: {country}, {purpose}, {year}: {preds}")

    # (3) 결과 batch 저장
    if rows:
        df_tmp = pd.DataFrame(rows)
        df_tmp.to_csv(out_csv, index=False, mode='a', header=first_write, encoding='utf-8-sig')
        first_write = False
        # (4) 저장된 데이터는 set에 즉시 추가(재시작 누락 방지)
        for r in rows:
            already_done.add((r['국가'], r['목적'], r['연월']))

print(f"\n✅ 모든 예측값이 {out_csv} 파일로 저장되었습니다.")
