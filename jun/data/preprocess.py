import pandas as pd

def preprocess_data(file_path):
    # Load the data with a multi-level header
    df = pd.read_csv(file_path, header=[0, 1], encoding='utf-8-sig')

    # Flatten the multi-level header
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Rename the first two columns
    df.rename(columns={df.columns[0]: '국적', df.columns[1]: '목적'}, inplace=True)

    # Forward fill the first two columns (국적 and 목적)
    df.iloc[:, 0:2] = df.iloc[:, 0:2].ffill()

    # Select only the '인원(명)' columns
    df = df[['국적', '목적'] + [col for col in df.columns if '인원(명)' in col]]

    # Rename columns for melting
    df.columns = [col.replace('_인원(명)', '') for col in df.columns]

    # Remove the last column which is a total
    df = df.iloc[:, :-1]

    # Melt the DataFrame
    df_melted = df.melt(id_vars=['국적', '목적'], var_name='YearMonth', value_name='Headcount')

    # Clean the 'YearMonth' column
    df_melted['YearMonth'] = pd.to_datetime(df_melted['YearMonth'], format='%Y년%m월')

    # Filter out unnecessary rows
    df_melted = df_melted[~df_melted['국적'].str.contains('계', na=False)]
    df_melted = df_melted[~df_melted['목적'].str.contains('계', na=False)]

    # Handle missing values and convert to integer
    df_melted['Headcount'] = pd.to_numeric(df_melted['Headcount'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

    print(df_melted.head())

    return df_melted

if __name__ == '__main__':
    preprocessed_df = preprocess_data(r'C:\Users\Admin\data\목적별 국적별 입국_250708124220.csv')
    preprocessed_df.to_csv(r'C:\Users\Admin\data\preprocessed_entrants.csv', index=False, encoding='utf-8-sig')
    print("Preprocessing complete. Saved to preprocessed_entrants.csv")