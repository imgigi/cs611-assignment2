import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataProcessor:
    
    def __init__(self, data_path='data', output_path='datamart'):
        self.data_path = data_path
        self.output_path = output_path
        
    def load_raw_data(self):
        """Load raw CSV files"""
        print("Loading raw data...")
        
        clickstream = pd.read_csv(f'{self.data_path}/feature_clickstream.csv')
        attributes = pd.read_csv(f'{self.data_path}/features_attributes.csv')
        financials = pd.read_csv(f'{self.data_path}/features_financials.csv')
        loans = pd.read_csv(f'{self.data_path}/lms_loan_daily.csv')
        
        return clickstream, attributes, financials, loans
    
    def create_bronze_layer(self):
        """Bronze layer - raw data ingestion"""
        print("Creating Bronze layer...")
        
        clickstream, attributes, financials, loans = self.load_raw_data()
        
        # just save as parquet, no transformation
        os.makedirs(f'{self.output_path}/bronze', exist_ok=True)
        
        clickstream.to_parquet(f'{self.output_path}/bronze/clickstream.parquet')
        attributes.to_parquet(f'{self.output_path}/bronze/attributes.parquet')
        financials.to_parquet(f'{self.output_path}/bronze/financials.parquet')
        loans.to_parquet(f'{self.output_path}/bronze/loans.parquet')
        
        print("Bronze layer created!")
        return clickstream, attributes, financials, loans
    
    def clean_attributes(self, df):
        """Clean attributes data"""
        df = df.copy()
        
        # handle missing values - convert to numeric first
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Age'] = df['Age'].apply(lambda x: np.nan if pd.isna(x) or x < 0 or x > 120 else x)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # standardize SSN format - remove invalid ones
        df['SSN'] = df['SSN'].apply(lambda x: x if isinstance(x, str) and len(x) > 5 else np.nan)
        
        # handle occupation nulls
        df['Occupation'] = df['Occupation'].fillna('Unknown')
        
        return df
    
    def clean_financials(self, df):
        """Clean financial data"""
        df = df.copy()
        
        # convert ALL numeric columns to numeric type
        numeric_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                       'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                       'Delay_from_due_date', 'Num_of_Delayed_Payment', 
                       'Num_Credit_Inquiries', 'Outstanding_Debt',
                       'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                       'Amount_invested_monthly', 'Monthly_Balance', 'Changed_Credit_Limit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # fix negative values
        for col in ['Num_of_Loan', 'Num_Bank_Accounts', 'Num_Credit_Card']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
        
        # cap outliers at 97th percentile
        for col in ['Num_of_Loan', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                    'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment']:
            if col in df.columns:
                q97 = df[col].quantile(0.97)
                df[col] = df[col].apply(lambda x: q97 if not pd.isna(x) and x > q97 else x)
        
        # parse Credit_History_Age
        def parse_credit_history(text):
            if pd.isna(text):
                return 0
            years = 0
            months = 0
            try:
                if 'Year' in str(text):
                    years = int(str(text).split('Year')[0].strip().split()[-1])
                if 'Month' in str(text):
                    months = int(str(text).split('Month')[0].strip().split()[-1])
            except:
                pass
            return years * 12 + months
        
        df['credit_history_months'] = df['Credit_History_Age'].apply(parse_credit_history)
        
        # handle categorical
        df['Payment_Behaviour'] = df['Payment_Behaviour'].fillna('Unknown')
        df['Credit_Mix'] = df['Credit_Mix'].fillna('Unknown')
        
        return df
    
    def clean_clickstream(self, df):
        """Clean clickstream data"""
        df = df.copy()
        
        # aggregate by user - take mean of features
        feature_cols = [c for c in df.columns if c.startswith('fe_')]
        
        # convert to numeric first
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # remove negative values
        for col in feature_cols:
            df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
        
        # cap extreme outliers
        for col in feature_cols:
            q95 = df[col].quantile(0.95)
            df[col] = df[col].apply(lambda x: q95 if x > q95 else x)
        
        # aggregate
        agg_dict = {col: 'mean' for col in feature_cols}
        agg_dict['snapshot_date'] = 'first'
        
        df_agg = df.groupby('Customer_ID').agg(agg_dict).reset_index()
        
        return df_agg
    
    def create_silver_layer(self):
        """Silver layer - cleaned data"""
        print("Creating Silver layer...")
        
        # load bronze
        clickstream = pd.read_parquet(f'{self.output_path}/bronze/clickstream.parquet')
        attributes = pd.read_parquet(f'{self.output_path}/bronze/attributes.parquet')
        financials = pd.read_parquet(f'{self.output_path}/bronze/financials.parquet')
        loans = pd.read_parquet(f'{self.output_path}/bronze/loans.parquet')
        
        # clean
        attributes_clean = self.clean_attributes(attributes)
        financials_clean = self.clean_financials(financials)
        clickstream_clean = self.clean_clickstream(clickstream)
        
        # loans - convert dates
        loans['snapshot_date'] = pd.to_datetime(loans['snapshot_date'])
        loans['loan_start_date'] = pd.to_datetime(loans['loan_start_date'])
        
        os.makedirs(f'{self.output_path}/silver', exist_ok=True)
        
        attributes_clean.to_parquet(f'{self.output_path}/silver/attributes_clean.parquet')
        financials_clean.to_parquet(f'{self.output_path}/silver/financials_clean.parquet')
        clickstream_clean.to_parquet(f'{self.output_path}/silver/clickstream_clean.parquet')
        loans.to_parquet(f'{self.output_path}/silver/loans_clean.parquet')
        
        print("Silver layer created!")
        return attributes_clean, financials_clean, clickstream_clean, loans
    
    def create_labels(self, loans, overdue_threshold=0, mob_threshold=6):
        """Create labels based on overdue amount at 6 months on book"""
        
        # filter to 6 months on book
        loans_6mob = loans[loans['installment_num'] == mob_threshold].copy()
        
        # create label: 1 if overdue_amt > 0 (default), else 0
        loans_6mob['overdue_amt'] = pd.to_numeric(loans_6mob['overdue_amt'], errors='coerce').fillna(0)
        loans_6mob['label'] = (loans_6mob['overdue_amt'] > overdue_threshold).astype(int)
        
        # keep only necessary columns
        label_df = loans_6mob[['Customer_ID', 'loan_id', 'label', 'snapshot_date']].copy()
        
        return label_df
    
    def create_feature_store(self, attributes, financials, clickstream, labels):
        """Create ML feature store by joining all sources"""
        
        print("Creating feature store...")
        
        # start with labels
        feature_df = labels.copy()
        
        # drop snapshot_date from other tables to avoid conflicts
        if 'snapshot_date' in attributes.columns:
            attributes = attributes.drop(columns=['snapshot_date'])
        if 'snapshot_date' in financials.columns:
            financials = financials.drop(columns=['snapshot_date'])
        if 'snapshot_date' in clickstream.columns:
            clickstream = clickstream.drop(columns=['snapshot_date'])
        
        # join attributes
        feature_df = feature_df.merge(attributes, on='Customer_ID', how='left')
        
        # join financials
        feature_df = feature_df.merge(financials, on='Customer_ID', how='left')
        
        # join clickstream
        feature_df = feature_df.merge(clickstream, on='Customer_ID', how='left')
        
        # CRITICAL: convert ALL columns to appropriate types BEFORE any operations
        # Get all numeric columns
        numeric_cols = ['Outstanding_Debt', 'Monthly_Inhand_Salary', 'Total_EMI_per_month',
                       'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Age',
                       'Annual_Income', 'Interest_Rate', 'Delay_from_due_date',
                       'Num_of_Delayed_Payment', 'Num_Credit_Inquiries',
                       'Credit_Utilization_Ratio', 'Monthly_Balance', 'Amount_invested_monthly',
                       'Changed_Credit_Limit', 'credit_history_months']
        
        # Add all fe_ columns
        fe_cols = [c for c in feature_df.columns if c.startswith('fe_')]
        numeric_cols.extend(fe_cols)
        
        # Convert to numeric
        for col in numeric_cols:
            if col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        
        # NOW fill NaNs
        feature_df = feature_df.fillna(0)
        
        # feature engineering
        feature_df['debt_to_income'] = feature_df['Outstanding_Debt'] / (feature_df['Monthly_Inhand_Salary'] + 1)
        feature_df['emi_to_salary'] = feature_df['Total_EMI_per_month'] / (feature_df['Monthly_Inhand_Salary'] + 1)
        feature_df['num_products'] = feature_df['Num_Bank_Accounts'] + feature_df['Num_Credit_Card'] + feature_df['Num_of_Loan']
        feature_df['loan_per_product'] = feature_df['Num_of_Loan'] / (feature_df['num_products'] + 1)
        
        # age groups
        feature_df['age_group'] = pd.cut(feature_df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                                          labels=['<25', '25-35', '35-45', '45-55', '55+'])
        
        # convert all object columns to string to avoid parquet type errors
        for col in feature_df.select_dtypes(include=['object']).columns:
            feature_df[col] = feature_df[col].astype(str)
        
        # drop PII
        drop_cols = ['Name', 'SSN', 'Customer_ID', 'loan_id', 'snapshot_date', 'Credit_History_Age']
        drop_cols = [c for c in drop_cols if c in feature_df.columns]
        
        feature_df_clean = feature_df.drop(columns=drop_cols)
        
        return feature_df_clean, feature_df
    
    def create_gold_layer(self):
        """Gold layer - ML-ready feature store"""
        print("Creating Gold layer...")
        
        # load silver
        attributes = pd.read_parquet(f'{self.output_path}/silver/attributes_clean.parquet')
        financials = pd.read_parquet(f'{self.output_path}/silver/financials_clean.parquet')
        clickstream = pd.read_parquet(f'{self.output_path}/silver/clickstream_clean.parquet')
        loans = pd.read_parquet(f'{self.output_path}/silver/loans_clean.parquet')
        
        # create labels
        labels = self.create_labels(loans)
        
        # create features
        feature_df_clean, feature_df_full = self.create_feature_store(
            attributes, financials, clickstream, labels
        )
        
        os.makedirs(f'{self.output_path}/gold', exist_ok=True)
        
        # save
        labels.to_parquet(f'{self.output_path}/gold/label_store.parquet')
        feature_df_clean.to_parquet(f'{self.output_path}/gold/feature_store.parquet')
        feature_df_full.to_parquet(f'{self.output_path}/gold/feature_store_with_ids.parquet')
        
        print(f"Gold layer created! Features shape: {feature_df_clean.shape}")
        print(f"Label distribution:\n{labels['label'].value_counts()}")
        
        return feature_df_clean, labels

