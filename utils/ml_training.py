import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

class MLTrainer:
    
    def __init__(self, model_path='models'):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        
    def prepare_features(self, df):
        """Prepare features for training"""
        
        df = df.copy()
        
        # handle categorical variables
        label_encoders = {}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col != 'label':
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
        
        # fill any remaining nulls
        df = df.fillna(0)
        
        return df, label_encoders
    
    def train_models(self, feature_path='datamart/gold/feature_store.parquet'):
        """Train multiple models and select best one"""
        
        print("Loading feature store...")
        df = pd.read_parquet(feature_path)
        
        # prepare data
        df_clean, label_encoders = self.prepare_features(df)
        
        X = df_clean.drop('label', axis=1)
        y = df_clean['label']
        
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"Label distribution: {y.value_counts().to_dict()}")
        
        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # handle imbalance with SMOTE
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        
        # scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sm)
        X_test_scaled = scaler.transform(X_test)
        
        # train multiple models
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=6)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            model.fit(X_train_scaled, y_train_sm)
            
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba
            }
        
        # select best model based on F1 score
        best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1'])
        best_model = results[best_model_name]['model']
        
        print(f"\n*** Best model: {best_model_name} ***")
        print(f"F1 Score: {results[best_model_name]['metrics']['f1']:.4f}")
        
        # save best model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'{self.model_path}/model_{best_model_name}_{timestamp}.pkl'
        scaler_filename = f'{self.model_path}/scaler_{timestamp}.pkl'
        encoders_filename = f'{self.model_path}/label_encoders_{timestamp}.pkl'
        metadata_filename = f'{self.model_path}/model_metadata_{timestamp}.json'
        
        joblib.dump(best_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(label_encoders, encoders_filename)
        
        # save metadata
        metadata = {
            'model_name': best_model_name,
            'timestamp': timestamp,
            'metrics': results[best_model_name]['metrics'],
            'feature_names': list(X.columns),
            'n_features': len(X.columns),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'all_models_results': {k: v['metrics'] for k, v in results.items()}
        }
        
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModel saved to {model_filename}")
        print(f"Scaler saved to {scaler_filename}")
        print(f"Metadata saved to {metadata_filename}")
        
        return best_model, scaler, label_encoders, metadata

