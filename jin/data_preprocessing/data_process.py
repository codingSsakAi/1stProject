import pandas as pd
import numpy as np
import os


class ForeignVisitorDataProcessorSimple:
    KEYWORDS_TO_REMOVE = [
        "소 계", "소계", "합 계", "합계", "계", "교포", "소개", "아시아주",
        "미 주", "구 주", "아프리카주", "오세아니아주", "기타", "미주", "구주",
        "아시아", "아프리카", "오세아니아", "전체", "총계", "총 계", "전 체"
    ]

    SEASON_MAP = {
        12: "겨울", 1: "겨울", 2: "겨울",
        3: "봄", 4: "봄", 5: "봄",
        6: "여름", 7: "여름", 8: "여름",
        9: "가을", 10: "가을", 11: "가을"
    }

    COVID_START = "2020-03-01"
    COVID_END = "2022-06-30"

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.data = None

    def load_and_clean(self):
        for encoding in ['utf-8', 'cp949', 'euc-kr']:
            try:
                df = pd.read_csv(self.input_path, encoding=encoding)
                break
            except:
                continue
        else:
            raise Exception("파일 인코딩 문제")

        df = df.iloc[1:].reset_index(drop=True)
        df.columns.values[0:2] = ['국적', '목적']
        df.dropna(subset=['국적', '목적'], inplace=True)
        df['국적'] = df['국적'].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
        df['목적'] = df['목적'].astype(str).str.strip().str.replace(r"\s+", "", regex=True)

        pattern = "|".join(self.KEYWORDS_TO_REMOVE)
        df = df[~df['국적'].str.contains(pattern)]
        df = df[~df['목적'].str.contains(pattern)]

        for col in df.columns[2:]:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)

        self.data = df

    def reshape_and_add_features(self):
        value_vars = [col for col in self.data.columns if col not in ['국적', '목적'] and '계' not in col]
        df = self.data.melt(id_vars=['국적', '목적'], value_vars=value_vars, var_name='연월', value_name='입국자수')

        df['연월'] = df['연월'].str.replace("년", "-").str.replace("월", "")
        df['날짜'] = pd.to_datetime(df['연월'] + "-01", errors='coerce')
        df.dropna(subset=['날짜'], inplace=True)

        df['연도'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        df['분기'] = df['날짜'].dt.quarter
        df['계절'] = df['월'].map(self.SEASON_MAP)

        covid_start = pd.to_datetime(self.COVID_START)
        covid_end = pd.to_datetime(self.COVID_END)
        df['코로나기간'] = ((df['날짜'] >= covid_start) & (df['날짜'] <= covid_end)).astype(int)

        df.sort_values(['국적', '목적', '날짜'], inplace=True)
        df['시계열순서'] = df.groupby(['국적', '목적']).cumcount() + 1

        df['입국자수'] = df['입국자수'].fillna(0)
        df['입국자수_1개월전'] = df.groupby(['국적', '목적'])['입국자수'].shift(1)
        df['입국자수_3개월전'] = df.groupby(['국적', '목적'])['입국자수'].shift(3)
        df['입국자수_12개월전'] = df.groupby(['국적', '목적'])['입국자수'].shift(12)

        df['입국자수_3개월평균'] = df.groupby(['국적', '목적'])['입국자수'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['입국자수_12개월평균'] = df.groupby(['국적', '목적'])['입국자수'].transform(lambda x: x.rolling(12, min_periods=1).mean())

        df['전년동월대비증감률'] = (
            (df['입국자수'] - df['입국자수_12개월전']) / df['입국자수_12개월전'].replace(0, np.nan) * 100
        ).fillna(0)

        # 결측치 보완: 지연 변수 및 이동평균은 0으로 채움
        lag_cols = ['입국자수_1개월전', '입국자수_3개월전', '입국자수_12개월전',
                    '입국자수_3개월평균', '입국자수_12개월평균']
        df[lag_cols] = df[lag_cols].fillna(0)

        self.data = df[[
            '국적', '목적', '연도', '월', '분기', '계절', '코로나기간', '시계열순서',
            '입국자수', '입국자수_1개월전', '입국자수_3개월전', '입국자수_12개월전',
            '입국자수_3개월평균', '입국자수_12개월평균', '전년동월대비증감률'
        ]]

    def save(self):
        self.data.to_csv(self.output_path, index=False, encoding='utf-8-sig')

    def run(self):
        self.load_and_clean()
        self.reshape_and_add_features()
        self.save()


if __name__ == "__main__":
    processor = ForeignVisitorDataProcessorSimple(
        input_path="./data/1_2_(로우데이터_합본.csv)목적별 국적별 입국_(05년1월~25년5월).csv",
        output_path="./data/processed/외국인입국자_전처리완료_딥러닝용.csv"
    )
    processor.run()
    print("✅ 전처리 완료! 저장됨.")
