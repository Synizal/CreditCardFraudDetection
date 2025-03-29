# Credit Card Fraud Detection System

This project provides a machine learning-based solution for credit card fraud detection with a user-friendly web interface for testing the models.

## System Overview

The system consists of two main components:
1. **Model Training** (`main.py`): Trains several machine learning models to detect fraudulent credit card transactions
2. **Web Interface** (`run.py`): Provides a Flask API and web interface to test the trained models

## Requirements

- Python 3.6+
- Required packages in `requirements.txt`

## Setup Instructions

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the model training script (this might take a while):
   ```
   python main.py
   ```
4. Start the web interface:
   ```
   python run.py
   ```
5. Open a browser and go to http://localhost:5000

## Project Structure

- `main.py`: The main script for data loading, processing, model training, and evaluation
- `run.py`: Flask application that serves the API and web interface
- `credit_card_fraud_detection/`: Directory containing all trained models and reports
  - `models/`: Trained model files
  - `reports/`: Performance reports and visualizations
  - `data/`: Dataset and data samples
  - `logs/`: Log files

## Features

### Model Training (`main.py`)
- Loads the credit card fraud dataset
- Handles data preprocessing (scaling, missing values, etc.)
- Applies SMOTE to handle class imbalance
- Trains multiple ML models (Logistic Regression, Random Forest, XGBoost, etc.)
- Evaluates and compares model performance
- Generates detailed reports and visualizations

### Web Interface (`run.py`)
- User-friendly interface to test the trained models
- Support for selecting different models
- Input form for transaction details
- Real-time fraud detection
- Visualization of prediction results

## Dataset

This project uses the Credit Card Fraud Detection dataset, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

Features:
- Time: Seconds elapsed between this transaction and the first transaction
- Amount: Transaction amount
- V1-V28: PCA-transformed features (for confidentiality)
- Class: 1 for fraudulent transactions, 0 for normal transactions

## Testing the System

Once the models are trained and the web server is running:
1. Go to http://localhost:5000 in your browser
2. Enter transaction details in the form:
   - Amount: The transaction amount
   - Time: Seconds elapsed (can be any value for testing)
   - Advanced Features: Click "Generate Random Features" or enter specific values
3. Select a model from the dropdown or use one of the top-performing models
4. Click "Detect Fraud" to see the prediction results

## Performance

The system evaluates models based on:
- Accuracy
- F1 Score
- ROC AUC
- Precision-Recall metrics

Due to the highly imbalanced nature of the dataset, F1 Score is used as the primary evaluation metric. 