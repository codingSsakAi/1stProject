import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

class Reporter:
    """예측 결과와 성능 평가를 시각화하고 리포트를 생성합니다."""

    def __init__(self, results_dir, config_module):
        self.results_dir = results_dir
        self.config = config_module

    def generate_reports(self, nationality, results, historical_data, performance_results, start_date, end_date):
        """모든 리포트를 생성합니다."""
        self.create_prediction_visualization(nationality, results, historical_data, start_date, end_date)
        self.save_performance_report(performance_results)
        self.save_prediction_csv(nationality, results)

    def create_prediction_visualization(self, nationality, results, historical_data, start_date, end_date):
        """예측 결과와 실제 데이터를 함께 시각화하여 리포트를 생성합니다."""
        all_purposes = sorted(list(set(results.keys()) | set(historical_data.keys())))
        if not all_purposes:
            print("시각화할 데이터가 없습니다.")
            return

        num_purposes = len(all_purposes)
        fig, axes = plt.subplots(num_purposes, 1, figsize=(18, 6 * num_purposes), sharex=True, squeeze=False)
        axes = axes.flatten()

        for i, purpose in enumerate(all_purposes):
            ax = axes[i]
            hist_df = historical_data.get(purpose)
            
            # 실제 데이터 플로팅
            if hist_df is not None and not hist_df.empty:
                hist_df = hist_df.copy()
                hist_df['입국자수'] = np.expm1(hist_df['입국자수'])
                ax.plot(hist_df['날짜'], hist_df['입국자수'], label='실제', color='gray', alpha=0.8, linewidth=2)

            # 1. 코로나 시기 표현 (빨간색 투명 마스킹)
            covid_start_date = pd.to_datetime(self.config.COVID_START_DATE)
            covid_end_date = pd.to_datetime(self.config.COVID_END_DATE)
            ax.axvspan(covid_start_date, covid_end_date, color='red', alpha=0.2, label='코로나 팬데믹 기간')

            # 예측 데이터 플로팅
            if purpose in results and results[purpose]:
                predictions = results[purpose]
                pred_dates = pd.to_datetime([p["month"] for p in predictions])
                pred_values = [p["value"] for p in predictions]
                ax.plot(pred_dates, pred_values, label='예측', color='red', marker='o', linestyle='--', markersize=5)
                
                # 실제-예측 연결선
                if hist_df is not None and not hist_df.empty:
                    last_hist_date = hist_df['날짜'].iloc[-1]
                    last_hist_value = hist_df['입국자수'].iloc[-1]
                    ax.plot([last_hist_date, pred_dates[0]], [last_hist_value, pred_values[0]], color='red', linestyle='--')

            ax.set_title(f"[{nationality}] {purpose} 입국자 수", fontsize=16)
            ax.set_ylabel("입국자 수", fontsize=12)
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='x', rotation=30)
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            # 3. 2005-01 ~ 2025-05 까지 대략 중요한 월만 표현 예측일 잘 보이게 표현
            # 4. 예측 시작일 전 월도 예측 입국자수 표현
            # 5. 그래프 선 등 설명 범례 그래프에서 잘 보이게 위치
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10)) # 최대 10개의 주요 틱
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            
            # 예측 시작 월 강조
            if purpose in results and results[purpose]:
                first_pred_date = pd.to_datetime(results[purpose][0]["month"])
                ax.axvline(first_pred_date, color='blue', linestyle=':', linewidth=2, label='예측 시작')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 범례를 그래프 밖에 위치

        fig.suptitle(f'{nationality} 목적별 입국자 수 예측 ({start_date} ~ {end_date})', fontsize=20, y=0.99)
        plt.tight_layout(rect=[0, 0.03, 0.95, 0.97]) # 범례 공간 확보
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.results_dir, f"{nationality}_목적별_예측_시각화_{timestamp}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"시각화 저장 완료: {plot_path}")

    def save_performance_report(self, performance_results):
        """성능 평가 리포트를 CSV 파일로 저장합니다."""
        if not performance_results:
            print("저장할 성능 데이터가 없습니다.")
            return

        report_data = []
        for metrics in performance_results:
            row = {
                "국적": metrics["nationality"],
                "목적": metrics["purpose"],
                "학습샘플수": metrics["training_samples"],
                "검증샘플수": metrics["validation_samples"],
                "학습에포크": metrics["epochs_trained"],
                "MAE_실제값": f"{int(metrics['mae']):,d}",
                "MAE_기준값": f"{int(metrics['mae_기준값']):,d}",
                "MAE_달성여부": "↓" if metrics['mae'] <= metrics['mae_기준값'] else "↑",
                "MAE_등급": metrics["mae_등급"],
                "RMSE_실제값": f"{int(metrics['rmse']):,d}",
                "RMSE_기준값": f"{int(metrics['rmse_기준값']):,d}",
                "RMSE_달성여부": "↓" if metrics['rmse'] <= metrics['rmse_기준값'] else "↑",
                "RMSE_등급": metrics["rmse_등급"],
                "R2_실제값": f"{metrics['r2_score']:.4f}",
                "R2_기준값": f"{metrics['r2_score_기준값']:.2f}",
                "R2_달성여부": "↑" if metrics['r2_score'] >= metrics['r2_score_기준값'] else "↓",
                "R2_등급": metrics["r2_score_등급"],
                "MAPE_실제값": f"{metrics['mape']:.1f}%",
                "MAPE_기준값": f"{metrics['mape_기준값']:.1f}%",
                "MAPE_달성여부": "↓" if metrics['mape'] <= metrics['mape_기준값'] else "↑",
                "MAPE_등급": metrics["mape_등급"],
                "F1_실제값": f"{metrics['f1_score']:.3f}",
                "F1_기준값": f"{metrics['f1_score_기준값']:.2f}",
                "F1_달성여부": "↑" if metrics['f1_score'] >= metrics['f1_score_기준값'] else "↓",
                "F1_등급": metrics["f1_score_등급"],
                "최종학습손실": f"{metrics['final_train_loss']:.6f}",
                "최종검증손실": f"{metrics['final_val_loss']:.6f}" if metrics['final_val_loss'] is not None else "N/A",
                "최종학습MAE": f"{metrics['final_train_mae']:.6f}",
                "최종검증MAE": f"{metrics['final_val_mae']:.6f}" if metrics['final_val_mae'] is not None else "N/A",
                "최고학습손실": f"{metrics['best_train_loss']:.6f}",
                "최고검증손실": f"{metrics['best_val_loss']:.6f}" if metrics['best_val_loss'] != float('inf') else "N/A",
                "조기종료여부": "예" if metrics['early_stopped'] else "아니오",
                "학습률": f"{metrics['learning_rate_used']:.6f}",
                "생성시간": metrics["timestamp"]
            }
            report_data.append(row)

        performance_df = pd.DataFrame(report_data)
        
        # Get the nationality from the first entry (assuming all entries are for the same nationality in one report)
        nationality = performance_results[0]['nationality'] if performance_results else 'Unknown'
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"{nationality}_리포트_{timestamp}.csv")
        performance_df.to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"성능 리포트 저장 완료: {report_path}")

    def save_prediction_csv(self, nationality, results):
        """예측 결과를 CSV 파일로 저장합니다."""
        if not results:
            return
        
        csv_data = []
        
        # Get all unique months from the results
        all_months = sorted(list(set(p["month"] for preds in results.values() for p in preds)))
        
        for month in all_months:
            row = {"월": month}
            total_prediction = 0
            
            # Initialize purpose predictions to 0
            공용_pred = 0
            상용_pred = 0
            관광_pred = 0
            유학연수_pred = 0
            
            for purpose_key, predictions_list in results.items():
                month_pred = next((p["value"] for p in predictions_list if p["month"] == month), 0)
                
                if purpose_key == "공용":
                    공용_pred = month_pred
                elif purpose_key == "상용":
                    상용_pred = month_pred
                elif purpose_key == "관광":
                    관광_pred = month_pred
                elif purpose_key == "유학연수":
                    유학연수_pred = month_pred
                
                total_prediction += month_pred
            
            row["총합"] = total_prediction
            row["공용"] = 공용_pred
            row["상용"] = 상용_pred
            row["관광"] = 관광_pred
            row["유학연수"] = 유학연수_pred
            
            tourism_ratio = (관광_pred / total_prediction * 100) if total_prediction > 0 else 0
            row["관광_비율"] = f"{tourism_ratio:.1f}%"
            
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.results_dir, f"{nationality}_완전예측리포트_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"예측 결과 CSV 저장 완료: {csv_path}")