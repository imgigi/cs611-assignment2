import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from scipy import stats

class MLMonitor:
    
    def __init__(self, output_path='datamart/gold/monitoring'):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/plots', exist_ok=True)
        
    def calculate_performance_metrics(self, predictions_path, labels_path):
        """Calculate model performance metrics"""
        
        print("Calculating performance metrics...")
        
        # load predictions and actual labels
        pred_df = pd.read_parquet(predictions_path)
        label_df = pd.read_parquet(labels_path)
        
        # drop snapshot_date from label_df to avoid conflicts
        if 'snapshot_date' in label_df.columns:
            label_df = label_df.drop(columns=['snapshot_date'])
        
        # merge
        eval_df = pred_df.merge(label_df, on=['Customer_ID', 'loan_id'], how='inner', suffixes=('_pred', '_actual'))
        
        y_true = eval_df['label']
        y_pred = eval_df['prediction']
        y_pred_proba = eval_df['prediction_proba']
        
        # calculate metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(eval_df),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'default_rate_actual': float(y_true.mean()),
            'default_rate_predicted': float(y_pred.mean())
        }
        
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        print(f"Performance Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics, eval_df
    
    def check_data_drift(self, current_data_path, reference_data_path):
        """Check for data drift using statistical tests"""
        
        print("Checking for data drift...")
        
        current_df = pd.read_parquet(current_data_path)
        reference_df = pd.read_parquet(reference_data_path)
        
        # get numeric columns
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['label', 'prediction', 'prediction_proba']]
        
        drift_results = {}
        
        for col in numeric_cols:
            if col in reference_df.columns:
                # KS test
                ks_stat, p_value = stats.ks_2samp(
                    current_df[col].dropna(),
                    reference_df[col].dropna()
                )
                
                drift_results[col] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drift_detected': p_value < 0.05  # significance level
                }
        
        # count drifted features
        n_drifted = sum(1 for v in drift_results.values() if v['drift_detected'])
        drift_pct = n_drifted / len(drift_results) * 100 if drift_results else 0
        
        print(f"Data Drift Check:")
        print(f"  Features analyzed: {len(drift_results)}")
        print(f"  Features with drift: {n_drifted} ({drift_pct:.1f}%)")
        
        return drift_results
    
    def monitor_prediction_stability(self, predictions_df):
        """Monitor prediction stability over time"""
        
        print("Monitoring prediction stability...")
        
        # group by snapshot_date if available
        if 'snapshot_date' in predictions_df.columns or 'snapshot_date_pred' in predictions_df.columns:
            date_col = 'snapshot_date' if 'snapshot_date' in predictions_df.columns else 'snapshot_date_pred'
            
            stability_metrics = predictions_df.groupby(date_col).agg({
                'prediction': ['mean', 'std'],
                'prediction_proba': ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            stability_metrics.columns = ['date', 'pred_mean', 'pred_std', 
                                        'proba_mean', 'proba_std', 'proba_min', 'proba_max']
        else:
            # overall statistics
            stability_metrics = {
                'prediction_mean': float(predictions_df['prediction'].mean()),
                'prediction_std': float(predictions_df['prediction'].std()),
                'proba_mean': float(predictions_df['prediction_proba'].mean()),
                'proba_std': float(predictions_df['prediction_proba'].std())
            }
        
        return stability_metrics
    
    def create_monitoring_visualizations(self, metrics, eval_df, drift_results):
        """Create monitoring visualizations"""
        
        print("Creating monitoring visualizations...")
        
        # 1. Confusion Matrix
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # 2. Prediction Distribution
        axes[0, 1].hist(eval_df['prediction_proba'], bins=50, edgecolor='black')
        axes[0, 1].set_title('Prediction Probability Distribution')
        axes[0, 1].set_xlabel('Predicted Probability')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Actual vs Predicted Default Rate
        default_rates = pd.DataFrame({
            'Actual': [metrics['default_rate_actual']],
            'Predicted': [metrics['default_rate_predicted']]
        })
        default_rates.T.plot(kind='bar', ax=axes[1, 0], legend=False)
        axes[1, 0].set_title('Default Rate: Actual vs Predicted')
        axes[1, 0].set_ylabel('Default Rate')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
        
        # 4. Performance Metrics
        perf_metrics = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc']
        }
        axes[1, 1].barh(list(perf_metrics.keys()), list(perf_metrics.values()))
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/plots/monitoring_dashboard.png', dpi=150)
        print(f"Dashboard saved to {self.output_path}/plots/monitoring_dashboard.png")
        
        # 5. Data Drift visualization
        if drift_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            features = list(drift_results.keys())[:20]  # top 20
            p_values = [drift_results[f]['p_value'] for f in features]
            colors = ['red' if drift_results[f]['drift_detected'] else 'green' for f in features]
            
            ax.barh(features, p_values, color=colors)
            ax.axvline(x=0.05, color='black', linestyle='--', label='Significance Level (0.05)')
            ax.set_xlabel('P-value (KS Test)')
            ax.set_title('Data Drift Detection - Top 20 Features')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.output_path}/plots/data_drift_analysis.png', dpi=150)
            print(f"Data drift plot saved to {self.output_path}/plots/data_drift_analysis.png")
        
        plt.close('all')
    
    def generate_monitoring_report(self, predictions_path='datamart/gold/predictions.parquet',
                                   labels_path='datamart/gold/label_store.parquet',
                                   reference_data_path='datamart/gold/feature_store.parquet'):
        """Generate complete monitoring report"""
        
        print("\n=== Model Monitoring Report ===\n")
        
        # performance metrics
        metrics, eval_df = self.calculate_performance_metrics(predictions_path, labels_path)
        
        # data drift
        drift_results = self.check_data_drift(predictions_path, reference_data_path)
        
        # prediction stability
        stability = self.monitor_prediction_stability(eval_df)
        
        # create visualizations
        self.create_monitoring_visualizations(metrics, eval_df, drift_results)
        
        # compile full report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'performance_metrics': metrics,
            'data_drift': drift_results,
            'prediction_stability': stability if isinstance(stability, dict) else stability.to_dict('records')
        }
        
        # save report
        report_path = f'{self.output_path}/monitoring_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nMonitoring report saved to {report_path}")
        
        # save to parquet as well for easy querying
        metrics_df = pd.DataFrame([{
            'timestamp': metrics['timestamp'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'n_samples': metrics['n_samples']
        }])
        
        metrics_history_path = f'{self.output_path}/metrics_history.parquet'
        if os.path.exists(metrics_history_path):
            history = pd.read_parquet(metrics_history_path)
            metrics_df = pd.concat([history, metrics_df], ignore_index=True)
        
        metrics_df.to_parquet(metrics_history_path)
        print(f"Metrics history updated: {metrics_history_path}")
        
        return report

