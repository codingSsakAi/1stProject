import pandas as pd

def covid_mask(df):
    """2020~2022년 기간을 마스킹"""
    return (df['연도'].isin([2020,2021,2022]))

def fill_covid_with_mean(df, target_col='입국자수'):
    """코로나 구간을 코로나 이전/이후 평균값으로 대체"""
    df2 = df.copy()
    mask = covid_mask(df2)
    mean_val = df2.loc[~mask, target_col].mean()
    df2.loc[mask, target_col] = mean_val
    return df2
