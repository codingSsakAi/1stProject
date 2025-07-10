import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(file_path):
    # Load the preprocessed data
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # Convert 'YearMonth' to datetime
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])

    # Set Korean font for matplotlib
    plt.rcParams['font.family'] = 'Malgun Gothic' # For Windows
    plt.rcParams['axes.unicode_minus'] = False

    # --- Summary Statistics ---

    # 1. Total visitors per year
    df['Year'] = df['YearMonth'].dt.year
    yearly_visitors = df.groupby('Year')['Headcount'].sum().reset_index()

    # 2. Top 10 nationalities by total visitors
    top_10_nationalities = df.groupby('국적')['Headcount'].sum().nlargest(10).reset_index()

    # 3. Total visitors by purpose
    purpose_visitors = df.groupby('목적')['Headcount'].sum().reset_index()

    # --- Visualizations ---

    # Bar plot for Yearly Visitors
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Year', y='Headcount', data=yearly_visitors, palette='viridis')
    plt.title('연도별 총 해외 여행객 유입량')
    plt.xlabel('연도')
    plt.ylabel('입국자 수')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Admin\data\yearly_visitors_bar.png')
    plt.close()

    # Save summary statistics to a text file
    with open(r'C:\Users\Admin\data\summary_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("--- Yearly Visitors ---\n")
        f.write(yearly_visitors.to_string())
        f.write("\n\n--- Top 10 Nationalities ---\n")
        f.write(top_10_nationalities.to_string())
        f.write("\n\n--- Visitors by Purpose ---\n")
        f.write(purpose_visitors.to_string())

if __name__ == '__main__':
    explore_data(r'C:\Users\Admin\data\preprocessed_entrants.csv')