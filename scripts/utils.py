"""
Diabetes Prediction Utilities

This module contains all utility functions for preprocessing, analyzing, and modeling
diabetes data from various datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from pathlib import Path
import json
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import shap


# -------------- DATA LOADING FUNCTIONS --------------

def load_dataset(file_path, dataset_type=None):
    """
    Load a diabetes dataset based on its type
    
    Args:
        file_path: Path to the dataset
        dataset_type: One of "cdc", "kaggle", or None to auto-detect
        
    Returns:
        DataFrame with loaded data
    """
    if dataset_type is None:
        df = pd.read_csv(file_path)
        if 'Diabetes_binary' in df.columns:
            dataset_type = "cdc"
        elif 'HbA1c_level' in df.columns:
            dataset_type = "kaggle"
        else:
            raise ValueError("Could not auto-detect dataset type. Please specify.")
    else:
        df = pd.read_csv(file_path)
    
    print(f"Loaded {dataset_type} dataset with shape: {df.shape}")
    
    if dataset_type == "cdc":
        df = df.rename(columns={'Diabetes_binary': 'diabetes'})
        
    elif dataset_type == "kaggle":
        pass
    
    return df


# -------------- PREPROCESSING FUNCTIONS --------------

def build_preprocessing_pipeline(df):
    """
    Build a scikit-learn preprocessing pipeline based on the dataset structure
    
    Args:
        df: DataFrame to analyze for preprocessing
        
    Returns:
        preprocessor: ColumnTransformer pipeline
        num_features: List of numerical feature names
        cat_features: List of categorical feature names
    """
    # Identify numeric and categorical columns (excluding target)
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target variable if it appears in the features
    if 'diabetes' in num_features:
        num_features.remove('diabetes')
    if 'diabetes' in cat_features:
        cat_features.remove('diabetes')
    
    # Define preprocessing for numerical features
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
    
    print(f"Preprocessing pipeline built with {len(num_features)} numerical features and {len(cat_features)} categorical features")
    
    return preprocessor, num_features, cat_features

def get_feature_names(preprocessor, num_features, cat_features):
    """
    Get feature names after preprocessing (including one-hot encoded features)
    
    Args:
        preprocessor: Fitted ColumnTransformer
        num_features: List of numerical feature names
        cat_features: List of categorical feature names
        
    Returns:
        List of feature names after transformation
    """
    feature_names = num_features.copy()
    
    # Get the one-hot encoded feature names
    if len(cat_features) > 0:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(cat_features)
        feature_names.extend(cat_feature_names)
    
    return feature_names


# -------------- EDA FUNCTIONS --------------

def perform_eda(df, target='diabetes'):
    """
    Perform exploratory data analysis on the dataset
    
    Args:
        df: DataFrame to analyze
        target: Target variable name
        
    Returns:
        None (displays visualizations)
    """
    print(f"Dataset shape: {df.shape}")
    
    print("\n--- Basic Statistics ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percent})
    print(missing_data[missing_data['Missing Values'] > 0])
    
    # Target distribution
    plt.figure(figsize=(10, 6))
    target_counts = df[target].value_counts()
    plt.bar(target_counts.index.astype(str), target_counts.values)
    plt.title(f'Distribution of {target}')
    plt.xlabel(target)
    plt.ylabel('Count')
    for i, v in enumerate(target_counts.values):
        plt.text(i, v + 0.1, str(v), ha='center')
    plt.show()
    
    # Class imbalance
    imbalance_ratio = target_counts.values[0] / target_counts.values[1] if len(target_counts) > 1 else float('inf')
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Correlation matrix (only for numerical features)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    return

def plot_feature_importance_to_target(df, target='diabetes', top_n=10):
    """
    Plot the most important features correlated with the target
    
    Args:
        df: DataFrame to analyze
        target: Target variable name
        top_n: Number of top features to display
        
    Returns:
        None (displays visualizations)
    """
    # Compute correlation with target
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [col for col in num_cols if col != target]
    
    if len(num_cols) > 0:
        correlations = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
        top_features = correlations.head(top_n)
        
        plt.figure(figsize=(12, 8))
        top_features.plot(kind='barh')
        plt.title(f'Top {top_n} Features Correlated with {target}')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.show()
    
    return

def plot_distributions_by_target(df, target='diabetes', num_features=None, cat_features=None):
    """
    Plot distributions of features split by target variable
    
    Args:
        df: DataFrame to analyze
        target: Target variable name
        num_features: List of numerical features to plot (if None, auto-detect)
        cat_features: List of categorical features to plot (if None, auto-detect)
        
    Returns:
        None (displays visualizations)
    """
    if num_features is None:
        num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target in num_features:
            num_features.remove(target)
    
    if cat_features is None:
        cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target in cat_features:
            cat_features.remove(target)
    
    # Plot distributions for numerical features
    if len(num_features) > 0:
        num_features = num_features[:min(5, len(num_features))]  # Limit to top 5 features
        n_cols = 2
        n_rows = (len(num_features) + 1) // n_cols
        
        plt.figure(figsize=(15, n_rows * 5))
        for i, feature in enumerate(num_features):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(data=df, x=feature, hue=target, kde=True, element="step", common_norm=False)
            plt.title(f'Distribution of {feature} by {target}')
        plt.tight_layout()
        plt.show()
    
    # Plot distributions for categorical features
    if len(cat_features) > 0:
        cat_features = cat_features[:min(4, len(cat_features))]  # Limit to top 4 features
        n_cols = 2
        n_rows = (len(cat_features) + 1) // n_cols
        
        plt.figure(figsize=(15, n_rows * 5))
        for i, feature in enumerate(cat_features):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.countplot(data=df, x=feature, hue=target)
            plt.title(f'Distribution of {feature} by {target}')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return

def create_feature_pairplot(df, target='diabetes', features=None):
    """
    Create a pairplot of selected features colored by target
    
    Args:
        df: DataFrame to analyze
        target: Target variable name
        features: List of features to include (if None, selects a sample)
        
    Returns:
        None (displays visualization)
    """
    if features is None:
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        num_cols = [col for col in num_cols if col != target]
        features = num_cols[:min(4, len(num_cols))]
    
    if len(features) > 1:
        plot_df = df[features + [target]].copy()
        sns.pairplot(plot_df, hue=target, corner=True, diag_kind='kde')
        plt.suptitle('Feature Relationships by Diabetes Status', y=1.02)
        plt.show()
    
    return


# -------------- MODEL TRAINING AND EVALUATION FUNCTIONS --------------

def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Split data into training and test sets
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        stratify: If provided, data is split in a stratified fashion using this as class labels
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """    
    print(f"Splitting data with test_size={test_size}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Class distribution in training set: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Class distribution in test set: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    return X_train, X_test, y_train, y_test


def train_or_load_model(X_train, y_train, model_type='xgboost', balance_method=None, 
                       cv=5, model_dir=None, model_name=None, force_retrain=False):
    """
    Train a specified model type or load an existing model if available
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: One of 'decision_tree', 'random_forest', 'xgboost'
        balance_method: Method to handle class imbalance, e.g., 'smote'
        cv: Number of cross-validation folds
        model_dir: Directory to check for existing model
        model_name: Base name for model files
        force_retrain: Whether to force retraining even if model exists
        
    Returns:
        Trained or loaded model
    """    
    if model_dir and model_name:
        specific_model_name = f"{model_name}_{model_type}"
        model_path = Path(model_dir) / f"{specific_model_name}_model.pkl"
        
        # Check if model exists and we don't want to force retrain
        if model_path.exists() and not force_retrain:
            print(f"Loading existing {model_type} model from {model_path}...")
            try:
                model = joblib.load(model_path)
                print(f"Model loaded successfully!")
                return model
            except Exception as e:
                print(f"Error loading model: {e}. Will retrain.")
    
    print(f"Training {model_type} model...")
    
    if balance_method == 'smote':
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"Applied SMOTE: {np.bincount(y_train)} -> {np.bincount(y_train_balanced)}")
        X_train, y_train = X_train_balanced, y_train_balanced
    
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=5, 
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    model.fit(X_train, y_train)
    
    if model_dir and model_name:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    return model

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate the model on test data with various metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: Names of features for importance visualization
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    feature_importance = None
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        feature_importance = pd.Series(model.feature_importances_, index=feature_names)
        feature_importance = feature_importance.sort_values(ascending=False)
    
    evaluation = {
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'feature_importance': feature_importance
    }
    
    return evaluation

def plot_evaluation_results(evaluation, model_name):
    """
    Plot evaluation metrics from model evaluation
    
    Args:
        evaluation: Dictionary from evaluate_model
        model_name: Name of the model for plot titles
        
    Returns:
        None (displays visualizations)
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = evaluation['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 2. ROC Curve
    plt.subplot(2, 3, 2)
    plt.plot(evaluation['fpr'], evaluation['tpr'], 
             label=f'ROC Curve (AUC = {evaluation["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    
    # 3. Precision-Recall Curve
    plt.subplot(2, 3, 3)
    plt.plot(evaluation['recall'], evaluation['precision'], 
             label=f'PR Curve (AUC = {evaluation["pr_auc"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    # 4. Feature Importance (if available)
    if evaluation['feature_importance'] is not None:
        plt.subplot(2, 3, 4)
        top_features = evaluation['feature_importance'].head(10)
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
    
    plt.suptitle(f'Evaluation Metrics for {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    print("\nClassification Report:")
    print(evaluation['classification_report'])
    
    return


# -------------- SHAP EXPLANATION FUNCTIONS --------------

def generate_shap_explanations(model, X, feature_names, sample_size=None):
    """
    Generate SHAP explanations for the model
    
    Args:
        model: Trained model
        X: Feature matrix for explanation
        feature_names: List of feature names
        sample_size: Number of samples to use for explanation (None for all)
        
    Returns:
        SHAP explainer and values
    """
    if sample_size is not None and sample_size < X.shape[0]:
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    print(f"Generating SHAP values for {X_sample.shape[0]} samples...")
    
    if isinstance(model, xgb.XGBClassifier):
        explainer = shap.Explainer(model, feature_names=feature_names)
    elif isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        explainer = shap.TreeExplainer(model)
    else:
        background = shap.kmeans(X_sample, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    shap_values = explainer(X_sample)
    
    if not hasattr(shap_values, 'feature_names') or shap_values.feature_names is None:
        shap_values.feature_names = feature_names
    
    return explainer, shap_values

def plot_shap_summary(shap_values, feature_names=None, plot_type="bar"):
    """
    Plot SHAP summary visualizations
    
    Args:
        shap_values: SHAP values from explainer
        feature_names: List of feature names
        plot_type: Type of plot ('bar', 'beeswarm', or 'both')
        
    Returns:
        None (displays visualizations)
    """
    if not hasattr(shap_values, 'feature_names') or shap_values.feature_names is None:
        if feature_names is not None:
            shap_values.feature_names = feature_names
    
    if plot_type in ["bar", "both"]:
        plt.figure(figsize=(12, 8))
        shap.plots.bar(shap_values, show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.show()
    
    if plot_type in ["beeswarm", "both"]:
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_values, show=False)
        plt.title("SHAP Feature Values")
        plt.tight_layout()
        plt.show()
    
    return

def plot_shap_dependence(shap_values, features, feature_names, top_n=3):
    """
    Plot SHAP dependence plots for top features
    
    Args:
        shap_values: SHAP values from explainer
        features: Original feature values
        feature_names: List of feature names
        top_n: Number of top features to show
        
    Returns:
        None (displays visualizations)
    """
    if not hasattr(shap_values, 'feature_names') or shap_values.feature_names is None:
        shap_values.feature_names = feature_names
    
    feature_importance = np.abs(shap_values.values).mean(0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    top_features = feature_importance_df.sort_values('importance', ascending=False).head(top_n)['feature'].tolist()
    
    print(f"SHAP values shape: {shap_values.values.shape}")
    print(f"Features shape: {features.shape}")
    
    if shap_values.values.shape[0] != features.shape[0]:
        print(f"Warning: Number of rows mismatch. Adjusting feature rows to match SHAP values.")
        if shap_values.values.shape[0] < features.shape[0]:
            features_subset = features[:shap_values.values.shape[0], :]
        else:
            raise ValueError("SHAP values have more rows than features, which shouldn't happen")
    else:
        features_subset = features
    
    for feature in top_features:
        plt.figure(figsize=(10, 7))
        try:
            feature_idx = list(feature_names).index(feature)
            
            plt.scatter(features_subset[:, feature_idx], 
                        shap_values.values[:, feature_idx],
                        alpha=0.5)
            plt.xlabel(feature)
            plt.ylabel(f"SHAP value for {feature}")
            plt.title(f"SHAP Dependence Plot for {feature}")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            z = np.polyfit(features_subset[:, feature_idx], shap_values.values[:, feature_idx], 1)
            p = np.poly1d(z)
            plt.plot(features_subset[:, feature_idx], p(features_subset[:, feature_idx]), 
                    "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting dependence for feature '{feature}': {e}")
            
    return

# -------------- HEALTHCARE WORKER DASHBOARD FUNCTIONS --------------

def create_patient_explainer(patient_data, model, explainer, feature_names, 
                            preprocessor):
    """
    Create an interactive explanation for a single patient
    
    Args:
        patient_data: DataFrame with patient data
        model: Trained model
        explainer: SHAP explainer
        feature_names: List of feature names
        preprocessor: Fitted preprocessor pipeline
        num_features: List of numerical feature names
        cat_features: List of categorical feature names
        
    Returns:
        HTML/Visualization of the explanation
    """
    patient_processed = preprocessor.transform(patient_data)
    
    prediction = model.predict(patient_processed)[0]
    probability = model.predict_proba(patient_processed)[0, 1]
    
    patient_shap = explainer(patient_processed)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Prediction info
    plt.subplot(2, 2, 1)
    plt.text(0.5, 0.5, f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}\n"
                        f"Probability: {probability:.2f}",
             horizontalalignment='center', verticalalignment='center',
             fontsize=15)
    plt.axis('off')
    
    # 2. SHAP force plot
    plt.subplot(2, 2, 2)
    shap.plots.waterfall(patient_shap[0], show=False)
    plt.title("Feature Contributions")
    
    # 3. Top contributing factors (positive and negative)
    plt.subplot(2, 1, 2)
    shap.plots.bar(patient_shap[0], show=False)
    plt.title("Top Factors Affecting Prediction")
    
    plt.tight_layout()
    plt.suptitle("Patient Diabetes Risk Explanation", fontsize=16, y=1.02)
    plt.show()
    
    print("\nRisk Factor Analysis:")
    
    values = patient_shap.values[0]
    indices = np.argsort(values)
    
    print("\nFactors DECREASING diabetes risk:")
    for i in indices[:3]:  
        if values[i] < 0:
            print(f"  - {feature_names[i]}: {values[i]:.4f} contribution")
    
    print("\nFactors INCREASING diabetes risk:")
    for i in indices[-3:]: 
        if values[i] > 0:
            print(f"  - {feature_names[i]}: {values[i]:.4f} contribution")
    
    return


# -------------- MODEL SAVE/LOAD FUNCTIONS --------------

def save_model_artifacts(model, preprocessor, feature_names, num_features, cat_features, 
                       model_dir, model_name, model_metadata=None):
    """
    Save model and associated artifacts for later use
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
        num_features: List of numerical feature names
        cat_features: List of categorical feature names
        model_dir: Directory to save model
        model_name: Name prefix for saved files
        model_metadata: Optional dictionary with additional model information
        
    Returns:
        Path to saved model
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    if model_metadata is None:
        model_metadata = {}
    
    model_metadata.update({
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_type': type(model).__name__,
        'num_features_count': len(num_features),
        'cat_features_count': len(cat_features),
        'total_features_count': len(feature_names)
    })
    
    if hasattr(model, 'n_estimators'):
        model_metadata['n_estimators'] = model.n_estimators
    if hasattr(model, 'max_depth'):
        model_metadata['max_depth'] = model.max_depth
    
    joblib.dump(model, model_path / f"{model_name}_model.pkl")
    joblib.dump(preprocessor, model_path / f"{model_name}_preprocessor.pkl")
    
    feature_info = {
        'feature_names': feature_names,
        'num_features': num_features,
        'cat_features': cat_features
    }
    
    with open(model_path / f"{model_name}_feature_info.pkl", 'wb') as f:
        pickle.dump(feature_info, f)
    
    with open(model_path / f"{model_name}_metadata.json", 'w') as f:
        for k, v in model_metadata.items():
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                model_metadata[k] = str(v)
        json.dump(model_metadata, f, indent=4)
    
    print(f"Model artifacts saved to {model_path}")
    return model_path

def load_model_artifacts(model_dir, model_name):
    """
    Load model and associated artifacts
    
    Args:
        model_dir: Directory where model is saved
        model_name: Name prefix for saved files
        
    Returns:
        Dictionary with model and associated artifacts
    """
    model_path = Path(model_dir)
    
    model_file = model_path / f"{model_name}_model.pkl"
    if model_file.exists():
        model = joblib.load(model_file)
    else:
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    preprocessor_file = model_path / f"{model_name}_preprocessor.pkl"
    if preprocessor_file.exists():
        preprocessor = joblib.load(preprocessor_file)
    else:
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_file}")
    
    feature_info_file = model_path / f"{model_name}_feature_info.pkl"
    if feature_info_file.exists():
        with open(feature_info_file, 'rb') as f:
            feature_info = pickle.load(f)
    else:
        raise FileNotFoundError(f"Feature info file not found: {feature_info_file}")
    
    metadata_file = model_path / f"{model_name}_metadata.json"
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    print(f"Model artifacts loaded from {model_path}")
    
    result = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_info['feature_names'],
        'num_features': feature_info['num_features'],
        'cat_features': feature_info['cat_features']
    }
    
    if metadata:
        result['metadata'] = metadata
    
    return result

def check_model_exists(model_dir, model_name):
    """
    Check if a saved model exists
    
    Args:
        model_dir: Directory where model would be saved
        model_name: Name prefix for saved files
        
    Returns:
        Boolean indicating if model exists
    """
    model_path = Path(model_dir)
    model_file = model_path / f"{model_name}_model.pkl"
    preprocessor_file = model_path / f"{model_name}_preprocessor.pkl"
    feature_info_file = model_path / f"{model_name}_feature_info.pkl"
    
    return model_file.exists() and preprocessor_file.exists() and feature_info_file.exists()


# -------------- DASHBOARD CREATION FUNCTION --------------

def create_healthcare_dashboard(model, preprocessor, feature_names, num_features, cat_features):
    """
    Create a dashboard for healthcare workers to use the model
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        feature_names: List of feature names after preprocessing
        num_features: List of numerical feature names
        cat_features: List of categorical feature names
        
    Returns:
        analyze_patient function
    """
    try:
        if isinstance(model, xgb.XGBClassifier):
            explainer = shap.Explainer(model, feature_names=feature_names)
        elif isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
            explainer = shap.TreeExplainer(model)
            explainer.feature_names = feature_names
        else:
            explainer = None
            print("Warning: SHAP explainer could not be initialized.")
    except Exception as e:
        explainer = None
        print(f"Warning: SHAP explainer could not be initialized: {e}")
    
    def analyze_patient(patient_data):
        """
        Analyze a new patient with the dashboard
        
        Args:
            patient_data: DataFrame with patient data (single row)
            
        Returns:
            Dictionary with analysis results
        """
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        
        for col in num_features + cat_features:
            if col not in patient_data.columns:
                if col in num_features:
                    patient_data[col] = 0
                else:
                    patient_data[col] = "Unknown"
        
        patient_processed = preprocessor.transform(patient_data)
        
        prediction = model.predict(patient_processed)[0]
        probability = model.predict_proba(patient_processed)[0, 1]
        
        analysis = {
            'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': probability,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }
        
        if explainer is not None:
            try:
                patient_shap = explainer(patient_processed)
                
                if not hasattr(patient_shap, 'feature_names') or patient_shap.feature_names is None:
                    patient_shap.feature_names = feature_names
                
                values = patient_shap.values[0]
                
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'value': values
                }).sort_values('value')
                
                decreasing_factors = []
                for _, row in shap_df.head(3).iterrows():
                    if row['value'] < 0:
                        decreasing_factors.append({
                            'feature': row['feature'],
                            'value': row['value']
                        })
                
                increasing_factors = []
                for _, row in shap_df.tail(3).iterrows():
                    if row['value'] > 0:
                        increasing_factors.append({
                            'feature': row['feature'],
                            'value': row['value']
                        })
                
                analysis['shap_available'] = True
                analysis['increasing_factors'] = increasing_factors
                analysis['decreasing_factors'] = decreasing_factors
                analysis['shap_values'] = patient_shap
                
                print(f"Prediction: {analysis['prediction']} (Probability: {probability:.2f})")
                print(f"Risk Level: {analysis['risk_level']}")
                
                plt.figure(figsize=(12, 6))
                
                plt.barh(range(len(values)), values)
                plt.yticks(range(len(values)), feature_names)
                plt.title('Feature Contributions')
                plt.axvline(x=0, color='black', linestyle='-')
                plt.grid(axis='x', linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.show()
                
                print("\n===== Risk Factor Analysis =====")
                print("\nFactors INCREASING diabetes risk:")
                for factor in increasing_factors:
                    print(f"  - {factor['feature']}: {factor['value']:.4f} contribution")
                
                print("\nFactors DECREASING diabetes risk:")
                for factor in decreasing_factors:
                    print(f"  - {factor['feature']}: {factor['value']:.4f} contribution")
                
            except Exception as e:
                print(f"Warning: SHAP explanation failed with error: {e}")
                analysis['shap_available'] = False
        else:
            analysis['shap_available'] = False
        
        return analysis
    
    print("Healthcare Dashboard created. Use analyze_patient() function to analyze patients.")
    return analyze_patient


# -------------- WHAT-IF ANALYSIS FUNCTION --------------

def create_what_if_analysis(patient_data, model, preprocessor, num_features, feature_ranges=None):
    """
    Create what-if analysis showing how changing a feature would affect prediction
    
    Args:
        patient_data: DataFrame with patient data (single row)
        model: Trained model
        preprocessor: Fitted preprocessor
        num_features: List of numerical feature names
        feature_ranges: Dictionary with {feature_name: (min, max, steps)} for features to analyze
        
    Returns:
        None (displays visualizations)
    """
    if feature_ranges is None:
        feature_ranges = {}
        
        for feature in num_features[:3]:
            if feature in patient_data.columns:
                current_value = patient_data[feature].values[0]
                
                min_val = max(0, current_value * 0.5)
                max_val = current_value * 1.5
                steps = 10
                
                feature_ranges[feature] = (min_val, max_val, steps)
    
    base_processed = preprocessor.transform(patient_data)
    base_probability = model.predict_proba(base_processed)[0, 1]
    
    plt.figure(figsize=(15, 5 * len(feature_ranges)))
    plot_idx = 1
    
    for feature, (min_val, max_val, steps) in feature_ranges.items():
        test_values = np.linspace(min_val, max_val, steps)
        probabilities = []
        
        for value in test_values:
            modified_data = patient_data.copy()
            modified_data[feature] = value
            
            modified_processed = preprocessor.transform(modified_data)
            modified_probability = model.predict_proba(modified_processed)[0, 1]
            probabilities.append(modified_probability)
        
        plt.subplot(len(feature_ranges), 1, plot_idx)
        plt.plot(test_values, probabilities, marker='o')
        
        current_value = patient_data[feature].values[0]
        plt.axvline(x=current_value, color='r', linestyle='--', 
                   label=f'Current: {current_value:.2f}')
        
        plt.axhline(y=base_probability, color='g', linestyle='--',
                   label=f'Current probability: {base_probability:.2f}')
        
        plt.title(f'Effect of Changing {feature} on Diabetes Probability')
        plt.xlabel(f'{feature} Value')
        plt.ylabel('Predicted Probability of Diabetes')
        plt.grid(True)
        plt.legend()
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    return