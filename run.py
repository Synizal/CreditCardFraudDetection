import os
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, precision_recall_curve
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

# Set up paths
BASE_DIR = Path('credit_card_fraud_detection')
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'test_results'

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

# Create Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Global variables
loaded_models = {}
scaler = None
top_models = []

def load_models():
    """Load all available models from the models directory"""
    global loaded_models, scaler, top_models
    
    print("Loading models...")
    
    # Create directories if they don't exist
    for directory in [BASE_DIR, MODELS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Load the scaler
    scaler_path = MODELS_DIR / 'scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
        
        # Check if scaler has feature names attribute
        if hasattr(scaler, 'feature_names_in_'):
            print("Scaler feature order:", ", ".join(scaler.feature_names_in_))
    else:
        print(f"Warning: Scaler not found at {scaler_path}")
        # Create a dummy scaler if none exists
        scaler = StandardScaler()
    
    # Load top models info if available
    top_models_path = REPORTS_DIR / 'top_models.json'
    if os.path.exists(top_models_path):
        with open(top_models_path, 'r') as f:
            top_models_data = json.load(f)
            top_models = top_models_data.get('top_models', [])
        print(f"Loaded top models info from {top_models_path}")
    
    # Load all model files
    model_files = list(MODELS_DIR.glob('*.pkl'))
    if not model_files:
        print("No model files found. Please run main.py first to train models.")
        return False
    
    # Exclude scaler.pkl
    model_files = [f for f in model_files if f.name != 'scaler.pkl']
    
    # Load each model
    for model_path in model_files:
        model_name = model_path.stem  # Get filename without extension
        try:
            model = joblib.load(model_path)
            loaded_models[model_name] = {
                'model': model,
                'path': str(model_path)
            }
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
    
    print(f"Successfully loaded {len(loaded_models)} models")
    return len(loaded_models) > 0

def load_scaler():
    """Load the scaler used for data preprocessing"""
    try:
        scaler_path = MODELS_DIR / 'scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
            return scaler
        else:
            print(f"Warning: Scaler not found at {scaler_path}")
            return None
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
        return None

def load_test_data():
    """Load test data for evaluation"""
    try:
        # Try to load from data directory
        test_data_path = DATA_DIR / 'test_data.csv'
        
        if os.path.exists(test_data_path):
            df = pd.read_csv(test_data_path)
            print(f"Loaded test data from {test_data_path}")
            return df
        else:
            # If no specific test data, use a sample from the original dataset
            original_data_path = DATA_DIR / 'creditcard.csv'
            if os.path.exists(original_data_path):
                df = pd.read_csv(original_data_path)
                # Take a random sample of 1000 records
                test_df = df.sample(n=min(1000, len(df)), random_state=42)
                # Save this sample for future reference
                test_df.to_csv(test_data_path, index=False)
                print(f"Created test sample from original data and saved to {test_data_path}")
                return test_df
            else:
                print(f"No test data or original data found. Please provide a test dataset.")
                return None
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None

def preprocess_test_data(df, scaler):
    """Preprocess test data for prediction"""
    try:
        if 'Class' in df.columns:
            X_test = df.drop('Class', axis=1)
            y_test = df['Class']
            has_labels = True
        else:
            X_test = df
            y_test = None
            has_labels = False
            
        # Apply scaling
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            # If no scaler is available, use the data as is
            print("Warning: No scaler found. Using unscaled data.")
            X_test_scaled = X_test.values
            
        return X_test_scaled, y_test, has_labels
    except Exception as e:
        print(f"Error preprocessing test data: {str(e)}")
        return None, None, False

def evaluate_models(models, X_test, y_test=None, has_labels=False):
    """Evaluate all models on test data"""
    results = {}
    
    for model_info in models:
        model_name = model_info['name']
        model = model_info['model']
        
        try:
            print(f"\nEvaluating model: {model_name}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Try to get probability predictions
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                has_proba = True
            except:
                y_prob = None
                has_proba = False
                print(f"Warning: Model {model_name} does not support probability predictions")
            
            # Create a directory for this model's results
            model_dir = RESULTS_DIR / model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            model_dir.mkdir(exist_ok=True)
            
            # If we have labels, calculate performance metrics
            if has_labels and y_test is not None:
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1 Score: {f1:.4f}")
                
                # Save confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(model_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                report_df.to_csv(model_dir / 'classification_report.csv')
                
                # If we have probabilities, calculate AUC and plot ROC curve
                if has_proba:
                    auc = roc_auc_score(y_test, y_prob)
                    print(f"AUC: {auc:.4f}")
                    
                    # Plot and save ROC curve
                    plt.figure(figsize=(10, 8))
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name}')
                    plt.legend(loc='lower right')
                    plt.savefig(model_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Plot and save Precision-Recall curve
                    plt.figure(figsize=(10, 8))
                    precision, recall, _ = precision_recall_curve(y_test, y_prob)
                    plt.plot(recall, precision)
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'Precision-Recall Curve - {model_name}')
                    plt.savefig(model_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    auc = None
                
                # Store results
                results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'auc': auc,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': report
                }
            else:
                # If we don't have labels, just save predictions
                prediction_counts = {'normal': sum(y_pred == 0), 'fraud': sum(y_pred == 1)}
                print(f"Predictions: {prediction_counts['normal']} normal, {prediction_counts['fraud']} fraud")
                
                # Save predictions to CSV
                predictions_df = pd.DataFrame({
                    'prediction': y_pred
                })
                if has_proba:
                    predictions_df['probability'] = y_prob
                predictions_df.to_csv(model_dir / 'predictions.csv', index=False)
                
                # Store results
                results[model_name] = {
                    'prediction_counts': prediction_counts
                }
            
            print(f"Results saved to {model_dir}")
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {str(e)}")
            continue
    
    # Save overall results
    with open(RESULTS_DIR / 'evaluation_results.json', 'w') as f:
        # Convert any non-serializable objects to strings
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {}
            for k, v in model_results.items():
                try:
                    # Try to see if it's JSON serializable
                    json.dumps({k: v})
                    serializable_results[model_name][k] = v
                except (TypeError, OverflowError):
                    # If not, convert to string or skip complex objects
                    if isinstance(v, np.ndarray):
                        serializable_results[model_name][k] = v.tolist()
                    elif k == 'classification_report':
                        serializable_results[model_name][k] = v
                    else:
                        serializable_results[model_name][k] = str(v)
        
        json.dump(serializable_results, f, indent=4)
    
    print(f"\nAll evaluation results saved to {RESULTS_DIR}")
    return results

def compare_models(results):
    """Compare model performance and create visualizations"""
    if not results:
        print("No results to compare")
        return
    
    # Get models with performance metrics (those with labels)
    models_with_metrics = {name: details for name, details in results.items() 
                          if 'accuracy' in details and 'f1_score' in details}
    
    if not models_with_metrics:
        print("No models with performance metrics to compare")
        return
    
    # Create comparison dataframe
    comparison_data = []
    for name, details in models_with_metrics.items():
        comparison_data.append({
            'model': name,
            'accuracy': details['accuracy'],
            'f1_score': details['f1_score'],
            'auc': details.get('auc', 'N/A')
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    
    # Create comparison plots
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    sns.barplot(x='model', y='accuracy', data=comparison_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Accuracy Comparison')
    plt.tight_layout()
    
    # Plot F1 Score
    plt.subplot(1, 3, 2)
    sns.barplot(x='model', y='f1_score', data=comparison_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('F1 Score Comparison')
    plt.tight_layout()
    
    # Plot AUC (if available)
    plt.subplot(1, 3, 3)
    auc_df = comparison_df[comparison_df['auc'] != 'N/A'].copy()
    if not auc_df.empty:
        auc_df['auc'] = auc_df['auc'].astype(float)
        sns.barplot(x='model', y='auc', data=auc_df)
        plt.xticks(rotation=45, ha='right')
        plt.title('AUC Comparison')
        plt.tight_layout()
    
    plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison saved to {RESULTS_DIR / 'model_comparison.csv'} and {RESULTS_DIR / 'model_comparison.png'}")

def main():
    print("Credit Card Fraud Detection - Model Evaluation")
    print("=============================================")
    
    # Load the top models
    models = load_top_models()
    if not models:
        return
    
    # Load the scaler
    scaler = load_scaler()
    
    # Load test data
    test_data = load_test_data()
    if test_data is None:
        return
    
    # Preprocess test data
    X_test, y_test, has_labels = preprocess_test_data(test_data, scaler)
    if X_test is None:
        return
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test, has_labels)
    
    # Compare model performance
    if has_labels:
        compare_models(results)
    
    print("\nEvaluation completed successfully!")

def preprocess_input(data):
    """Preprocess input data for prediction"""
    # Convert input to appropriate format
    if isinstance(data, dict):
        # Single transaction as dictionary
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        # Multiple transactions as list of dictionaries
        df = pd.DataFrame(data)
    else:
        # Assume it's already a DataFrame
        df = data
    
    # Ensure all expected columns are present
    expected_columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Default value if column is missing
    
    # Drop non-feature columns if they exist
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)
    
    # Apply scaling
    if scaler is not None:
        # Save Amount for later reference before scaling
        amounts = df['Amount'].values
        times = df['Time'].values
        
        # Check if scaler has feature_names_in_ attribute (sklearn >= 1.0)
        if hasattr(scaler, 'feature_names_in_'):
            # Use the exact feature order from the scaler
            feature_order = list(scaler.feature_names_in_)
            print("Using scaler's original feature order")
        else:
            # Fallback to standard order
            feature_order = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            print("Scaler doesn't have feature_names_in_ attribute, using default order")
        
        # Create a copy with the right column order
        df_ordered = pd.DataFrame()
        for col in feature_order:
            if col in df.columns:
                df_ordered[col] = df[col]
            else:
                print(f"Warning: Feature {col} not in input data, using default value 0")
                df_ordered[col] = 0  # Default value if missing
        
        # Transform the data
        scaled_data = scaler.transform(df_ordered)
        df_scaled = pd.DataFrame(scaled_data, columns=df_ordered.columns)
        
        # For display purposes, restore the original Amount and Time
        df_scaled['Amount_Original'] = amounts
        df_scaled['Time_Original'] = times
        
        return df_scaled
    else:
        # If no scaler available, just return the original data
        df['Amount_Original'] = df['Amount']
        df['Time_Original'] = df['Time']
        return df

def get_model_by_name(model_name):
    """Get a model by its name"""
    # First, try exact match
    if model_name in loaded_models:
        return loaded_models[model_name]['model']
    
    # Try case-insensitive match
    for name, model_info in loaded_models.items():
        if name.lower() == model_name.lower():
            return model_info['model']
        
    # Try matching substrings
    for name, model_info in loaded_models.items():
        if model_name.lower() in name.lower():
            return model_info['model']
    
    return None

def get_prediction(input_data, model_name=None):
    """Get prediction for input data using specified or best model"""
    try:
        # Preprocess the input
        processed_data = preprocess_input(input_data)
        
        # Remove non-feature columns for prediction
        prediction_data = processed_data.drop(['Amount_Original', 'Time_Original'], axis=1, errors='ignore')
        
        # Select model
        model = None
        if model_name:
            model = get_model_by_name(model_name)
            if model is None:
                return {'error': f"Không tìm thấy mô hình '{model_name}'"}, 404
        else:
            # Use the first top model if available, otherwise use the first loaded model
            if top_models and len(top_models) > 0:
                top_model_name = top_models[0]['name']
                top_model_type = top_models[0]['type']
                model_key = f"{top_model_name.lower().replace(' ', '_')}"
                if top_model_type == 'smote':
                    model_key += "_smote"
                
                if model_key in loaded_models:
                    model = loaded_models[model_key]['model']
                    model_name = model_key
            
            # Fallback to first loaded model
            if model is None and loaded_models:
                model_name = list(loaded_models.keys())[0]
                model = loaded_models[model_name]['model']
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(prediction_data)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
        else:
            predictions = model.predict(prediction_data)
            probabilities = predictions.astype(float)
        
        # Prepare results
        results = []
        for i in range(len(predictions)):
            results.append({
                'prediction': int(predictions[i]),
                'fraud_probability': float(probabilities[i]),
                'amount': float(processed_data['Amount_Original'].iloc[i]),
                'time': float(processed_data['Time_Original'].iloc[i]),
                'is_fraud': bool(predictions[i] == 1)
            })
        
        return {
            'model_used': model_name,
            'predictions': results
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': f'Lỗi khi dự đoán: {str(e)}'}, 500

@app.route('/')
def home():
    """Render home page"""
    # Get list of available models for the dropdown
    available_models = list(loaded_models.keys())
    
    # Get top models information
    top_models_info = []
    for model in top_models:
        model_name = model['name'].lower().replace(' ', '_')
        if model['type'] == 'smote':
            model_name += "_smote"
        
        if model_name in loaded_models:
            top_models_info.append({
                'name': model['name'],
                'type': model['type'],
                'f1_score': model['f1_score'],
                'accuracy': model['accuracy'],
                'id': model_name
            })
    
    return render_template(
        'index.html', 
        available_models=available_models,
        top_models=top_models_info
    )

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for getting predictions"""
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({'error': 'Không có dữ liệu nào được cung cấp'}), 400
        
        # Get model name if specified
        model_name = data.get('model_name', None)
        
        # Get transaction data
        transaction_data = data.get('transaction', None)
        
        if not transaction_data:
            return jsonify({'error': 'Không có dữ liệu giao dịch nào được cung cấp'}), 400
        
        # Get prediction
        result = get_prediction(transaction_data, model_name)
        
        # Return result
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """API endpoint to get available models"""
    try:
        models_list = []
        for name, model_info in loaded_models.items():
            model_type = "smote" if "_smote" in name else "original"
            display_name = name.replace('_', ' ').title()
            if "_smote" in name:
                display_name = display_name.replace('Smote', '(SMOTE)')
            
            models_list.append({
                'id': name,
                'name': display_name,
                'type': model_type
            })
        
        return jsonify({
            'models': models_list,
            'top_models': top_models
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load models before starting the server
    if load_models():
        print("Models loaded successfully. Starting server...")
        app.run(debug=True, port=5000)
    else:
        print("Failed to load models. Please run main.py first to train models.") 