import os
# Fix for joblib CPU core detection issue on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4" # Explicitly set to a specific number instead of using cpu_count()
# Disable joblib's CPU count detection entirely
os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get("NUMBER_OF_PROCESSORS", "4")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import time
import traceback

# Set up directories for storing results
BASE_DIR = Path('credit_card_fraud_detection')
REPORTS_DIR = BASE_DIR / 'reports'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
DATA_DIR = BASE_DIR / 'data'

def setup_dirs():
    """Create all needed directories for the project"""
    for directory in [BASE_DIR, REPORTS_DIR, MODELS_DIR, LOGS_DIR, DATA_DIR]:
        directory.mkdir(exist_ok=True)
    print(f"Created directories at {BASE_DIR}")

# Create directories before setting up logging
setup_dirs()

# Set up logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'fraud_detection_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('fraud_detection')

def ensure_json_serializable(obj):
    """Convert any non-serializable objects to their serializable counterparts"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_json_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return list(ensure_json_serializable(item) for item in obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, datetime.timedelta):
        return str(obj)
    else:
        return obj

def save_json(data, filepath):
    """Save data as JSON with proper serialization"""
    try:
        serializable_data = ensure_json_serializable(data)
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        return False

def download_dataset():
    """Download the creditcard.csv dataset if not available locally"""
    try:
        import requests
        logger.info("Dataset not found. Attempting to download...")
        # URL for the Credit Card Fraud Detection dataset
        url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
        
        # Create a data directory if it doesn't exist
        file_path = DATA_DIR / "creditcard.csv"
        
        # Download the file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Dataset downloaded successfully to {file_path}")
            return file_path
        else:
            logger.error(f"Failed to download dataset. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return None

def load_data():
    """Load the credit card fraud detection dataset"""
    logger.info("Loading dataset...")
    try:
        # Try multiple potential paths for the dataset
        potential_paths = [
            'creditcard.csv',
            '../input/creditcardfraud/creditcard.csv',
            '../input/creditcard.csv',
            './data/creditcard.csv',
            DATA_DIR / 'creditcard.csv'
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info(f"Dataset loaded successfully from {path}")
                
                # Save a copy to our data directory
                if not os.path.exists(DATA_DIR / 'creditcard.csv'):
                    df.to_csv(DATA_DIR / 'creditcard.csv', index=False)
                    logger.info(f"Dataset saved to {DATA_DIR / 'creditcard.csv'}")
                
                return df
        
        # Try to download the dataset
        downloaded_path = download_dataset()
        if downloaded_path and os.path.exists(downloaded_path):
            df = pd.read_csv(downloaded_path)
            logger.info(f"Dataset loaded successfully from {downloaded_path}")
            return df
        
        # If not found, ask for the file path
        logger.warning("Dataset not found in common locations and download failed.")
        user_path = input("Please enter the path to the creditcard.csv file: ")
        if os.path.exists(user_path):
            df = pd.read_csv(user_path)
            logger.info(f"Dataset loaded successfully from {user_path}")
            
            # Save a copy to our data directory
            df.to_csv(DATA_DIR / 'creditcard.csv', index=False)
            logger.info(f"Dataset saved to {DATA_DIR / 'creditcard.csv'}")
            
            return df
        else:
            logger.error(f"File not found at {user_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def check_missing_values(df):
    """Check for missing values in the dataset"""
    missing_value = df.isnull().sum()
    percentage_missing_value = missing_value / len(df) * 100 
    result = pd.DataFrame({'missing_value': missing_value, 'percentage_missing_value': percentage_missing_value})
    
    # Save missing values report
    result.to_csv(REPORTS_DIR / 'missing_values_report.csv')
    logger.info(f"Missing values report saved to {REPORTS_DIR / 'missing_values_report.csv'}")
    
    return result

def handle_duplicates(df):
    """Remove duplicate records from the dataset"""
    duplicate_values = df.duplicated().sum()
    logger.info(f'Number of duplicate rows: {duplicate_values}')
    
    df = df.drop_duplicates()
    logger.info(f'Number of rows after dropping duplicates: {df.shape[0]}')
    return df

def check_class_balance(df):
    """Check the balance of classes in the dataset"""
    classes = df['Class'].value_counts()
    logger.info(f'normal_trans = {classes[0]}')
    logger.info(f'fraud_trans = {classes[1]}')
    logger.info(f'percentage_normal_trans = {(classes[0] / df["Class"].count())*100:.2f}%')
    logger.info(f'percentage_fraud_trans = {(classes[1] / df["Class"].count())*100:.2f}%')
    
    # Save class balance report
    class_balance = pd.DataFrame({
        'transaction_type': ['normal_trans', 'fraud_trans'],
        'count': [classes[0], classes[1]],
        'percentage': [(classes[0] / df["Class"].count())*100, (classes[1] / df["Class"].count())*100]
    })
    class_balance.to_csv(REPORTS_DIR / 'class_balance_report.csv', index=False)
    logger.info(f"Class balance report saved to {REPORTS_DIR / 'class_balance_report.csv'}")
    
    return classes

def plot_class_distribution(classes):
    """Plot the distribution of normal vs fraudulent transactions"""
    title = ['normal_trans', 'fraud_trans']
    value = [classes[0], classes[1]]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(title, value, color='lightblue')
    
    for bar in bars:
        yval = bar.get_height()  
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 20,  
                 str(int(yval)),  
                 ha='center', va='bottom', fontweight='bold') 
    
    plt.title('Class Distribution')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    
    # Save the plot
    plt.savefig(REPORTS_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"Class distribution plot saved to {REPORTS_DIR / 'class_distribution.png'}")
    
    plt.show()

def preprocess_data(df):
    """Preprocess the data for model training"""
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Testing set shape: {X_test.shape}")
    
    # Save scaler for future use
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    logger.info(f"Scaler saved to {MODELS_DIR / 'scaler.pkl'}")
    
    # Save splits information - Using the helper function to ensure JSON serialization
    splits_info = {
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'train_fraud_count': y_train.sum(),
        'test_fraud_count': y_test.sum(),
        'train_fraud_percentage': (y_train.sum() / len(y_train)) * 100,
        'test_fraud_percentage': (y_test.sum() / len(y_test)) * 100,
    }
    
    save_json(splits_info, REPORTS_DIR / 'data_splits_info.json')
    logger.info(f"Data splits information saved to {REPORTS_DIR / 'data_splits_info.json'}")
    
    # Save feature correlation matrix
    plt.figure(figsize=(20, 16))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(REPORTS_DIR / 'feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
    logger.info(f"Feature correlation matrix saved to {REPORTS_DIR / 'feature_correlation_matrix.png'}")
    plt.close()
    
    # Save a small sample of train and test data for reference
    X_train.head(100).to_csv(DATA_DIR / 'X_train_sample.csv', index=False)
    X_test.head(100).to_csv(DATA_DIR / 'X_test_sample.csv', index=False)
    pd.Series(y_train).head(100).to_csv(DATA_DIR / 'y_train_sample.csv', index=False)
    pd.Series(y_test).head(100).to_csv(DATA_DIR / 'y_test_sample.csv', index=False)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def handle_imbalanced_data(X_train, y_train):
    """Apply SMOTE to handle imbalanced data"""
    logger.info("Applying SMOTE to handle class imbalance...")
    # Use only random_state parameter without n_jobs
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Shape of X_train before SMOTE: {X_train.shape}")
    logger.info(f"Shape of X_train after SMOTE: {X_train_resampled.shape}")
    
    class_distribution = pd.Series(y_train_resampled).value_counts()
    logger.info("Class distribution after SMOTE:")
    logger.info(class_distribution)
    
    # Save SMOTE results - Using the helper function to ensure JSON serialization
    smote_info = {
        'original_shape': X_train.shape[0],
        'resampled_shape': X_train_resampled.shape[0],
        'original_fraud_count': pd.Series(y_train).sum(),
        'resampled_fraud_count': pd.Series(y_train_resampled).sum(),
        'original_normal_count': len(y_train) - pd.Series(y_train).sum(),
        'resampled_normal_count': len(y_train_resampled) - pd.Series(y_train_resampled).sum(),
    }
    
    save_json(smote_info, REPORTS_DIR / 'smote_resampling_info.json')
    logger.info(f"SMOTE resampling information saved to {REPORTS_DIR / 'smote_resampling_info.json'}")
    
    return X_train_resampled, y_train_resampled

def save_model_report(name, model, y_test, y_pred, y_prob=None, is_smote=False):
    """Save comprehensive report for a model"""
    suffix = "_smote" if is_smote else ""
    model_name_safe = name.lower().replace(' ', '_')
    
    # Create directory for this model
    model_dir = REPORTS_DIR / model_name_safe
    model_dir.mkdir(exist_ok=True)
    
    # Save performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_test, y_prob)
            metrics['auc'] = auc
            
            # Save ROC curve
            plt.figure(figsize=(10, 8))
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}{" (SMOTE)" if is_smote else ""}')
            plt.legend(loc='lower right')
            plt.savefig(model_dir / f'roc_curve{suffix}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save Precision-Recall curve
            plt.figure(figsize=(10, 8))
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {name}{" (SMOTE)" if is_smote else ""}')
            plt.savefig(model_dir / f'precision_recall_curve{suffix}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not calculate AUC for {name}: {str(e)}")
    
    # Save metrics
    save_json(metrics, model_dir / f'metrics{suffix}.json')
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}{" (SMOTE)" if is_smote else ""}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(model_dir / f'confusion_matrix{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save classification report
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    cls_report_df = pd.DataFrame(cls_report).transpose()
    cls_report_df.to_csv(model_dir / f'classification_report{suffix}.csv')
    
    # Save model parameters
    try:
        params = model.get_params()
        # Convert any non-serializable objects to strings
        save_json(params, model_dir / f'model_params{suffix}.json')
    except Exception as e:
        logger.warning(f"Could not save model parameters for {name}: {str(e)}")
    
    logger.info(f"All reports for {name}{' (SMOTE)' if is_smote else ''} saved to {model_dir}")
    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models=None, use_smote=True):
    """Train and evaluate multiple models on the dataset."""
    if models is None:
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
    results = {}
    best_models = {}
    
    # Train and evaluate models on original data
    logger.info("Training models on original data...")
    
    for name, model in models.items():
        try:
            logger.info(f"Training {name}...")
            start_time = time.time()
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities for AUC if the model supports it
            y_prob = None
            try:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities for {name}: {str(e)}")
            
            # Calculate and save metrics
            metrics = save_model_report(name, model, y_test, y_pred, y_prob, is_smote=False)
            
            # Save the model
            save_model(model, name, suffix="")
            
            # Add to results
            execution_time = time.time() - start_time
            results[name] = {**metrics, 'execution_time': execution_time}
            best_models[name] = model
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            traceback.print_exc()
    
    # Save original data results
    save_json(results, REPORTS_DIR / 'model_results_original.json')
    
    # Train and evaluate models with SMOTE if requested
    if use_smote:
        try:
            logger.info("Applying SMOTE for handling imbalanced data...")
            X_train_smote, y_train_smote = handle_imbalanced_data(X_train, y_train)
            
            logger.info("Training models on SMOTE-resampled data...")
            smote_results = {}
            
            for name, model in models.items():
                try:
                    logger.info(f"Training {name} with SMOTE...")
                    start_time = time.time()
                    
                    # Train the model
                    model.fit(X_train_smote, y_train_smote)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Get probabilities for AUC if the model supports it
                    y_prob = None
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_prob = model.predict_proba(X_test)[:, 1]
                    except Exception as e:
                        logger.warning(f"Could not get prediction probabilities for {name} with SMOTE: {str(e)}")
                    
                    # Calculate and save metrics
                    metrics = save_model_report(name, model, y_test, y_pred, y_prob, is_smote=True)
                    
                    # Save the model
                    save_model(model, name, suffix="_smote")
                    
                    # Add to results
                    execution_time = time.time() - start_time
                    smote_results[name] = {**metrics, 'execution_time': execution_time}
                    
                    # Update best models if SMOTE version is better
                    if metrics['f1_score'] > results.get(name, {}).get('f1_score', 0):
                        best_models[name] = model
                        logger.info(f"SMOTE version of {name} performs better (F1: {metrics['f1_score']:.4f} vs {results.get(name, {}).get('f1_score', 0):.4f})")
                    
                    logger.info(f"{name} with SMOTE - Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name} with SMOTE: {str(e)}")
                    traceback.print_exc()
            
            # Save SMOTE results
            save_json(smote_results, REPORTS_DIR / 'model_results_smote.json')
            
        except Exception as e:
            logger.error(f"Error applying SMOTE: {str(e)}")
            traceback.print_exc()
    
    # Determine top 3 models based on F1 score
    all_models = []
    
    for name, metrics in results.items():
        all_models.append({
            'name': name,
            'f1_score': metrics['f1_score'],
            'accuracy': metrics['accuracy'],
            'type': 'original'
        })
    
    if use_smote:
        for name, metrics in smote_results.items():
            all_models.append({
                'name': name,
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy'],
                'type': 'smote'
            })
    
    # Sort by F1 score
    top_models = sorted(all_models, key=lambda x: x['f1_score'], reverse=True)[:3]
    
    # Save top models info
    save_json({'top_models': top_models}, REPORTS_DIR / 'top_models.json')
    
    logger.info("Top 3 Models:")
    for i, model_info in enumerate(top_models, 1):
        logger.info(f"{i}. {model_info['name']} ({model_info['type']}) - F1: {model_info['f1_score']:.4f}, Acc: {model_info['accuracy']:.4f}")
    
    return best_models, results, smote_results if use_smote else None

def find_best_model(results):
    """Find the best performing model based on F1 score"""
    # Sort models by F1 score
    sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    # Save top 3 models info
    top_models = []
    for i, (name, details) in enumerate(sorted_models[:3], 1):
        top_models.append({
            'rank': i,
            'name': name,
            'f1_score': details['f1_score'],
            'accuracy': details['accuracy'],
            'auc': details.get('auc', 'N/A'),
            'path': details['path']
        })
    
    # Save top models information
    with open(MODELS_DIR / 'top_models.json', 'w') as f:
        json.dump(top_models, f, indent=4)
    
    logger.info("Top 3 models:")
    for model in top_models:
        logger.info(f"Rank {model['rank']}: {model['name']} (F1: {model['f1_score']:.4f})")
    
    # Get the best model
    best_model_name = sorted_models[0][0]
    best_model_info = results[best_model_name]
    
    logger.info(f"\nBest model: {best_model_name}")
    logger.info(f"Accuracy: {best_model_info['accuracy']:.4f}")
    logger.info(f"F1 Score: {best_model_info['f1_score']:.4f}")
    if best_model_info.get('auc'):
        logger.info(f"AUC: {best_model_info['auc']:.4f}")
    
    return best_model_name, best_model_info['model']

def save_model(model, model_name, suffix=""):
    """Save a trained model to disk"""
    try:
        # Create a safe filename from the model name
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '') + suffix
        
        # Save the model
        model_path = MODELS_DIR / f"{safe_name}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

def save_final_report(df, models_results, best_model_name, runtime):
    """Save a comprehensive final report with all details"""
    report = {
        'dataset': {
            'total_records': len(df),
            'features': df.shape[1] - 1,
            'normal_transactions': len(df[df['Class'] == 0]),
            'fraudulent_transactions': len(df[df['Class'] == 1]),
            'fraud_percentage': (len(df[df['Class'] == 1]) / len(df)) * 100
        },
        'data_processing': {
            'duplicates_removed': df.duplicated().sum(),
            'missing_values': df.isnull().sum().sum()
        },
        'models_summary': {
            model_name: {
                'accuracy': details['accuracy'],
                'f1_score': details['f1_score'],
                'auc': details.get('auc', 'N/A')
            }
            for model_name, details in models_results.items()
        },
        'best_model': {
            'name': best_model_name,
            'accuracy': models_results[best_model_name]['accuracy'],
            'f1_score': models_results[best_model_name]['f1_score'],
            'auc': models_results[best_model_name].get('auc', 'N/A'),
            'path': models_results[best_model_name]['path']
        },
        'runtime': {
            'start_time': runtime['start'],
            'end_time': runtime['end'],
            'duration_seconds': runtime['duration'].total_seconds(),
            'duration_formatted': str(runtime['duration'])
        }
    }
    
    # Save the report
    with open(REPORTS_DIR / 'final_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Final report saved to {REPORTS_DIR / 'final_report.json'}")
    
    # Create a summary text file
    with open(REPORTS_DIR / 'summary.txt', 'w') as f:
        f.write("Credit Card Fraud Detection Project Summary\n")
        f.write("=========================================\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  - Total records: {report['dataset']['total_records']}\n")
        f.write(f"  - Normal transactions: {report['dataset']['normal_transactions']}\n")
        f.write(f"  - Fraudulent transactions: {report['dataset']['fraudulent_transactions']}\n")
        f.write(f"  - Fraud percentage: {report['dataset']['fraud_percentage']:.4f}%\n\n")
        
        f.write("Models Performance:\n")
        for model_name, metrics in report['models_summary'].items():
            f.write(f"  - {model_name}:\n")
            f.write(f"    - Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"    - F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"    - AUC: {metrics['auc']}\n\n")
        
        f.write("Best Model:\n")
        f.write(f"  - Name: {report['best_model']['name']}\n")
        f.write(f"  - Accuracy: {report['best_model']['accuracy']:.4f}\n")
        f.write(f"  - F1 Score: {report['best_model']['f1_score']:.4f}\n")
        f.write(f"  - AUC: {report['best_model']['auc']}\n")
        f.write(f"  - File: {report['best_model']['path']}\n\n")
        
        f.write("Runtime Information:\n")
        f.write(f"  - Start time: {report['runtime']['start_time']}\n")
        f.write(f"  - End time: {report['runtime']['end_time']}\n")
        f.write(f"  - Duration: {report['runtime']['duration_formatted']}\n")
    
    logger.info(f"Summary report saved to {REPORTS_DIR / 'summary.txt'}")

def generate_comparative_visualizations(results, smote_results):
    """Generate visualizations comparing all models' performance"""
    try:
        logger.info("Generating comparative visualizations for all models")
        
        # Create a comparison dataframe
        comparison_data = []
        
        # Add original results
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Type': 'Original',
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_score'],
                'AUC': metrics.get('auc', None),
                'Execution Time': metrics['execution_time']
            })
        
        # Add SMOTE results if available
        if smote_results:
            for model_name, metrics in smote_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Type': 'SMOTE',
                    'Accuracy': metrics['accuracy'],
                    'F1 Score': metrics['f1_score'],
                    'AUC': metrics.get('auc', None),
                    'Execution Time': metrics['execution_time']
                })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        comparison_df.to_csv(REPORTS_DIR / 'model_comparison.csv', index=False)
        logger.info(f"Model comparison data saved to {REPORTS_DIR / 'model_comparison.csv'}")
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Plot Accuracy
        plt.subplot(2, 2, 1)
        sns.barplot(x='Model', y='Accuracy', hue='Type', data=comparison_df)
        plt.title('Accuracy Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='lower right')
        
        # Plot F1 Score
        plt.subplot(2, 2, 2)
        sns.barplot(x='Model', y='F1 Score', hue='Type', data=comparison_df)
        plt.title('F1 Score Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='lower right')
        
        # Plot AUC (if available)
        plt.subplot(2, 2, 3)
        auc_df = comparison_df.dropna(subset=['AUC'])
        if not auc_df.empty:
            sns.barplot(x='Model', y='AUC', hue='Type', data=auc_df)
            plt.title('AUC Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.legend(loc='lower right')
        else:
            plt.text(0.5, 0.5, 'AUC not available for any model', 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot Execution Time
        plt.subplot(2, 2, 4)
        sns.barplot(x='Model', y='Execution Time', hue='Type', data=comparison_df)
        plt.title('Execution Time Comparison (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparative visualizations saved to {REPORTS_DIR / 'model_comparison.png'}")
        
        # Create a ROC curve comparison for all models
        plt.figure(figsize=(12, 10))
        
        # Get all models that have AUC values
        models_with_auc = comparison_df.dropna(subset=['AUC'])
        
        if not models_with_auc.empty:
            # For each model with AUC, plot its ROC curve
            for idx, row in models_with_auc.iterrows():
                model_name = row['Model']
                model_type = row['Type']
                model_suffix = "_smote" if model_type == "SMOTE" else ""
                
                try:
                    # Load the ROC curve data
                    roc_path = REPORTS_DIR / model_name.lower().replace(' ', '_') / f'roc_curve{model_suffix}.png'
                    
                    # If we wanted to plot the actual curve here, we'd need to save/load the fpr, tpr values
                    # Instead, we're just noting that a separate ROC curve exists for this model
                    plt.text(0.5, 0.9 - 0.05 * idx, 
                            f"{model_name} ({model_type}): AUC = {row['AUC']:.4f}", 
                            horizontalalignment='center',
                            fontsize=12)
                except Exception as e:
                    logger.warning(f"Could not process ROC data for {model_name} ({model_type}): {str(e)}")
            
            plt.axis('off')
            plt.title('ROC Curves Available for Individual Models')
            plt.savefig(REPORTS_DIR / 'roc_curve_availability.png', dpi=300, bbox_inches='tight')
        else:
            plt.text(0.5, 0.5, 'AUC not available for any model', 
                   horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            plt.savefig(REPORTS_DIR / 'roc_curve_availability.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating comparative visualizations: {str(e)}")
        traceback.print_exc()

def ensure_dataset():
    """Ensure the dataset exists, either by loading or downloading it"""
    logger.info("Checking for dataset...")
    
    # Try to load the dataset
    df = load_data()
    
    if df is None:
        logger.error("Failed to find or download the dataset")
        return None
    
    # Display basic information
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Dataset preview (first 5 rows):\n{df.head()}")
    
    # Check for missing values
    logger.info("\nChecking for missing values:")
    missing_values = check_missing_values(df)
    
    # Handle duplicate records
    logger.info("\nHandling duplicate records:")
    df = handle_duplicates(df)
    
    # Check class balance
    logger.info("\nChecking class balance:")
    classes = check_class_balance(df)
    
    # Plot class distribution
    logger.info("\nPlotting class distribution:")
    plot_class_distribution(classes)
    
    return df

def main():
    """Main execution function"""
    try:
        logger.info("Starting credit card fraud detection analysis")
        
        # Load and preprocess data
        logger.info("\nLoading and preprocessing data:")
        df = ensure_dataset()
        if df is None:
            logger.error("Data loading failed.")
            return
            
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        if X_train is None or y_train is None:
            logger.error("Data preprocessing failed.")
            return
        
        # Handle imbalanced data using SMOTE
        logger.info("\nHandling imbalanced data with SMOTE:")
        X_train_resampled, y_train_resampled = handle_imbalanced_data(X_train, y_train)
        
        # Train and evaluate models
        logger.info("\nTraining and evaluating models:")
        logger.info("This may take some time depending on your system...")
        best_models, results, smote_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        if not results:
            logger.error("No models were successfully trained.")
            return
        
        # Generate comparative visualizations
        logger.info("\nGenerating comparative visualizations:")
        generate_comparative_visualizations(results, smote_results)
        
        logger.info("\nCredit card fraud detection analysis completed successfully!")
        logger.info(f"All results have been saved to {REPORTS_DIR}")
        logger.info(f"All models have been saved to {MODELS_DIR}")
        
    except Exception as e:
        logger.error(f"An error occurred in main execution: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
