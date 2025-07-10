

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])

    # Set Korean font for matplotlib
    plt.rcParams['font.family'] = 'Malgun Gothic' # For Windows
    plt.rcParams['axes.unicode_minus'] = False

    # 1. Total Monthly Visitors Trend
    monthly_total = df.groupby('YearMonth')['Headcount'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_total, x='YearMonth', y='Headcount')
    plt.title('월별 총 해외 여행객 유입량 추이')
    plt.xlabel('연월')
    plt.ylabel('입국자 수')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Admin\data\monthly_total_visitors.png')
    plt.close()

    # 2. Top 5 Nationalities Trend
    # Get top 5 nationalities excluding '전체'
    top_5_nationalities = df[df['국적'] != '전 체'].groupby('국적')['Headcount'].sum().nlargest(5).index
    df_top_5 = df[df['국적'].isin(top_5_nationalities)]

    monthly_nationality_total = df_top_5.groupby(['YearMonth', '국적'])['Headcount'].sum().unstack().fillna(0).reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_nationality_total.melt(id_vars='YearMonth', var_name='국적', value_name='Headcount'), 
                 x='YearMonth', y='Headcount', hue='국적')
    plt.title('주요 5개 국적별 월별 해외 여행객 유입량 추이')
    plt.xlabel('연월')
    plt.ylabel('입국자 수')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Admin\data\top_5_nationalities_trend.png')
    plt.close()

    # 3. Visitors by Purpose Trend (excluding '전체' and empty purpose)
    purpose_df = df[~df['목적'].isin(['전 체', ''])]
    monthly_purpose_total = purpose_df.groupby(['YearMonth', '목적'])['Headcount'].sum().unstack().fillna(0).reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_purpose_total.melt(id_vars='YearMonth', var_name='목적', value_name='Headcount'), 
                 x='YearMonth', y='Headcount', hue='목적')
    plt.title('목적별 월별 해외 여행객 유입량 추이')
    plt.xlabel('연월')
    plt.ylabel('입국자 수')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Admin\data\purpose_trend.png')
    plt.close()

    print(r"Visualization complete. Plots saved to C:\Users\Admin\data")

if __name__ == '__main__':
    visualize_data(r'C:\Users\Admin\data\preprocessed_entrants.csv')

