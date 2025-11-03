import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from glob import glob

class MLInference:
    
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.metadata = None
        
    def load_latest_model(self):
        """Load the most recent model"""
        
        model_files = glob(f'{self.model_path}/model_*.pkl')
        if not model_files:
            raise FileNotFoundError("No trained models found!")
        
        # get latest model
        latest_model = max(model_files, key=os.path.getctime)
        # Extract timestamp: model_random_forest_20251101_170057.pkl -> 20251101_170057
        parts = latest_model.split('_')
        timestamp = '_'.join(parts[-2:]).replace('.pkl', '')
        
        print(f"Loading model: {latest_model}")
        
        self.model = joblib.load(latest_model)
        self.scaler = joblib.load(f'{self.model_path}/scaler_{timestamp}.pkl')
        self.label_encoders = joblib.load(f'{self.model_path}/label_encoders_{timestamp}.pkl')
        
        metadata_file = f'{self.model_path}/model_metadata_{timestamp}.json'
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Model loaded successfully!")
        print(f"Model: {self.metadata['model_name']}")
        print(f"Training date: {self.metadata['timestamp']}")
        
        return self.model, self.scaler, self.label_encoders
    
    def prepare_features(self, df):
        """Prepare features using saved encoders"""
        
        df = df.copy()
        
        # apply label encoders
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                # handle unseen categories
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                df[col] = encoder.transform(df[col])
        
        # fill nulls
        df = df.fillna(0)
        
        # ensure same feature order as training
        feature_names = self.metadata['feature_names']
        missing_cols = set(feature_names) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        
        df = df[feature_names]
        
        return df
    
    def predict(self, feature_path='datamart/gold/feature_store_with_ids.parquet',
                output_path='datamart/gold/predictions.parquet'):
        """Make predictions on new data"""
        
        print("Making predictions...")
        
        if self.model is None:
            self.load_latest_model()
        
        # load data
        df = pd.read_parquet(feature_path)
        
        # keep IDs
        id_cols = ['Customer_ID', 'loan_id', 'snapshot_date']
        ids = df[id_cols].copy()
        
        # prepare features
        df_features = df.drop(columns=id_cols + ['label'], errors='ignore')
        df_clean = self.prepare_features(df_features)
        
        # scale
        X_scaled = self.scaler.transform(df_clean)
        
        # predict
        predictions = self.model.predict(X_scaled)
        predictions_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # create output dataframe
        output_df = ids.copy()
        output_df['prediction'] = predictions
        output_df['prediction_proba'] = predictions_proba
        output_df['prediction_timestamp'] = datetime.now()
        
        # save
        output_df.to_parquet(output_path)
        print(f"Predictions saved to {output_path}")
        print(f"Total predictions: {len(output_df)}")
        print(f"Predicted defaults: {predictions.sum()} ({predictions.mean()*100:.2f}%)")
        
        return output_df

