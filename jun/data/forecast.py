import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

def forecast_visitors(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])

    # Aggregate to total monthly visitors
    monthly_total = df.groupby('YearMonth')['Headcount'].sum().reset_index()
    monthly_total.columns = ['ds', 'y']

    # Initialize and fit Prophet model
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=False, # Monthly data, so no weekly seasonality
        daily_seasonality=False   # Monthly data, so no daily seasonality
    )
    model.fit(monthly_total)

    # Create future dataframe for predictions (until end of 2026)
    future = model.make_future_dataframe(periods=19, freq='MS') # 7 months for 2025 (June-Dec) + 12 months for 2026
    forecast = model.predict(future)

    # --- Visualization ---
    plt.rcParams['font.family'] = 'Malgun Gothic' # For Windows
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot historical data (Dark Blue)
    ax.plot(monthly_total['ds'], monthly_total['y'], 'o-', color='#1A237E', label='과거 실제 유입량', markersize=4, linewidth=2)

    # Plot predictions (Vibrant Orange)
    ax.plot(forecast['ds'], forecast['yhat'], '-', color='#FF6F00', label='예측 유입량', linewidth=2)

    # Plot confidence interval (Light Orange with transparency)
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='#FFB74D', alpha=0.3, label='예측 신뢰 구간')

    # Highlight the forecasted period (Light Grey background)
    forecast_start_date = pd.to_datetime('2025-06-01')
    ax.axvspan(forecast_start_date, forecast['ds'].max(), color='#E0E0E0', alpha=0.4, label='예측 기간')

    # Add titles and labels for clarity
    ax.set_title('해외 여행객 유입량 예측 및 추세 (2025년 하반기 ~ 2026년)', fontsize=16)
    ax.set_xlabel('연월', fontsize=12)
    ax.set_ylabel('입국자 수', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(r'C:\Users\Admin\data\visitor_forecast_improved.png')
    plt.close()

    # --- Presenting Forecasted Numbers ---
    print("--- 해외 여행객 유입량 예측치 (월별) ---")
    # Filter for 2025 H2 and 2026
    forecast_2025_2026 = forecast[(forecast['ds'].dt.year == 2025) & (forecast['ds'].dt.month >= 6) | (forecast['ds'].dt.year == 2026)]
    print(forecast_2025_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))

if __name__ == '__main__':
    forecast_visitors(r'C:\Users\Admin\data\preprocessed_entrants.csv')