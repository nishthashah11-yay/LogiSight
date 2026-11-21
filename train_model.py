"""
Machine Learning Training Pipeline for LogiSight Analytics Last-Mile Delivery
Trains, evaluates, and exports predictive models for Delivery Time and Delay Risk.
Supports both Regression (Delivery Time in minutes) and Classification (Delay Probability).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from data_processor import load_raw_data, clean_and_prepare_data


def train_and_export_models(data_path=None, model_dir="models"):
    """
    Trains regression and classification models, outputs detailed metrics,
    and saves serialized pipeline objects for interactive inference in Streamlit.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 70)
    print(" LOGISIGHT ANALYTICS: MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)
    
    # 1. Load and prepare data
    print("\n[Step 1/5] Ingesting and preprocessing delivery data...")
    raw_df = load_raw_data(data_path)
    df, threshold, mean_val, std_val = clean_and_prepare_data(raw_df)
    print(f" Dataset Shape: {df.shape[0]} records, {df.shape[1]} features")
    print(f" Late Delivery Benchmark: Delivery_Time > {threshold:.2f} min (Mean + 1 Std)")

    # 2. Define Features & Target
    numeric_features = ['Agent_Age', 'Agent_Rating', 'Distance_KM', 'Prep_Time_Min', 'Order_Hour']
    categorical_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    feature_cols = numeric_features + categorical_features

    X = df[feature_cols]
    y_reg = df['Delivery_Time']
    y_clf = df['Is_Late']

    print(f"\n[Step 2/5] Splitting data into 80% Train and 20% Test sets...")
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )
    print(f" Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 3. Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    # 4. Train Regression Models (Delivery Time Prediction)
    print("\n[Step 3/5] Training Regression Models for Delivery Time Prediction...")
    
    # Linear Regression (Baseline)
    lin_pipeline = Pipeline([('prep', preprocessor), ('regressor', LinearRegression())])
    lin_pipeline.fit(X_train, y_train_reg)
    lin_pred = lin_pipeline.predict(X_test)
    lin_mae = mean_absolute_error(y_test_reg, lin_pred)
    lin_rmse = np.sqrt(mean_squared_error(y_test_reg, lin_pred))
    lin_r2 = r2_score(y_test_reg, lin_pred)
    print(f"  > Linear Regression Baseline:  MAE = {lin_mae:.2f} min | RMSE = {lin_rmse:.2f} min | R² = {lin_r2:.4f}")

    # Random Forest Regressor (Production Model)
    rf_regressor = RandomForestRegressor(
        n_estimators=60,
        max_depth=14,
        min_samples_split=6,
        random_state=42,
        n_jobs=-1
    )
    rf_pipeline = Pipeline([('prep', preprocessor), ('regressor', rf_regressor)])
    rf_pipeline.fit(X_train, y_train_reg)
    rf_pred = rf_pipeline.predict(X_test)
    rf_mae = mean_absolute_error(y_test_reg, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test_reg, rf_pred))
    rf_r2 = r2_score(y_test_reg, rf_pred)
    print(f"  > Random Forest Regressor:     MAE = {rf_mae:.2f} min | RMSE = {rf_rmse:.2f} min | R² = {rf_r2:.4f}")

    # 5. Train Classification Models (Late Delivery Risk)
    print("\n[Step 4/5] Training Classification Models for Late Delivery Risk...")
    
    # Logistic Regression (Baseline)
    log_pipeline = Pipeline([('prep', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    log_pipeline.fit(X_train, y_train_clf)
    log_pred = log_pipeline.predict(X_test)
    log_acc = accuracy_score(y_test_clf, log_pred)
    print(f"  > Logistic Regression:         Accuracy = {log_acc * 100:.2f}%")

    # Random Forest Classifier (Production Model with balanced class weights)
    rf_classifier = RandomForestClassifier(
        n_estimators=60,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf_pipeline = Pipeline([('prep', preprocessor), ('classifier', rf_classifier)])
    clf_pipeline.fit(X_train, y_train_clf)
    clf_pred = clf_pipeline.predict(X_test)
    clf_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    
    clf_acc = accuracy_score(y_test_clf, clf_pred)
    clf_prec = precision_score(y_test_clf, clf_pred)
    clf_rec = recall_score(y_test_clf, clf_pred)
    clf_f1 = f1_score(y_test_clf, clf_pred)
    clf_auc = roc_auc_score(y_test_clf, clf_proba)
    print(f"  > Random Forest Classifier:    Accuracy = {clf_acc * 100:.2f}% | Precision = {clf_prec:.4f} | Recall = {clf_rec:.4f} | F1 = {clf_f1:.4f} | AUC = {clf_auc:.4f}")

    # 6. Extract Feature Importances
    fitted_prep = rf_pipeline.named_steps['prep']
    cat_names = list(fitted_prep.named_transformers_['cat'].get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_names
    importances = rf_pipeline.named_steps['regressor'].feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n Top 7 Most Influential Factors in Last-Mile Delivery Time:")
    for idx, row in feat_imp_df.head(7).iterrows():
        print(f"   - {row['Feature']}: {row['Importance'] * 100:.2f}%")

    # 7. Export Model Artifacts
    print("\n[Step 5/5] Exporting model artifacts to disk...")
    reg_path = os.path.join(model_dir, "delivery_regressor.pkl")
    clf_path = os.path.join(model_dir, "delay_classifier.pkl")
    metrics_path = os.path.join(model_dir, "metrics_summary.json")

    with open(reg_path, 'wb') as f:
        pickle.dump(rf_pipeline, f)
    
    with open(clf_path, 'wb') as f:
        pickle.dump(clf_pipeline, f)

    metrics_dict = {
        'late_threshold': threshold,
        'overall_mean_time': mean_val,
        'overall_std_time': std_val,
        'regression': {
            'linear_regression': {'mae': round(lin_mae, 2), 'rmse': round(lin_rmse, 2), 'r2': round(lin_r2, 4)},
            'random_forest': {'mae': round(rf_mae, 2), 'rmse': round(rf_rmse, 2), 'r2': round(rf_r2, 4)}
        },
        'classification': {
            'logistic_regression': {'accuracy': round(log_acc, 4)},
            'random_forest': {
                'accuracy': round(clf_acc, 4),
                'precision': round(clf_prec, 4),
                'recall': round(clf_rec, 4),
                'f1_score': round(clf_f1, 4),
                'roc_auc': round(clf_auc, 4)
            }
        },
        'top_features': feat_imp_df.head(10).to_dict(orient='records')
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print(f" Saved Regressor to: {reg_path}")
    print(f" Saved Classifier to: {clf_path}")
    print(f" Saved Evaluation Metrics to: {metrics_path}")
    print("=" * 70)
    print(" MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return rf_pipeline, clf_pipeline, metrics_dict


if __name__ == "__main__":
    train_and_export_models()
