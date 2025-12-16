# Neural_-Network-_Based-_Predictive_-Analytics_-Platform-_for-_Investment
This repository contains an end-to-end machine learning pipeline for forecasting financial market trends using deep learning. It leverages multi-feature engineering, LSTM neural architectures, and robust evaluation metrics to predict short-term market movements for investment decision support.
Key Features

Automated market data ingestion via Yahoo Finance

Multi-factor technical feature engineering

Supervised time-series sequence generation

Deep learning model based on LSTM neural networks

Train/validation/test temporal split

Performance evaluation using:

RMSE

MAE

Directional accuracy

Visualization of predicted vs. actual market movement

Modular architecture for research extension, customization, and deployment
Project Structure
 neural-investment-forecasting
│
├── config.py                # Hyperparameters and global settings
├── data.py                  # Data download, cleaning, feature engineering, sequencing
├── model.py                 # LSTM neural network architecture
├── trainer.py               # Training, evaluation, metrics, dataloaders
├── main.py                  # Execution script for end-to-end pipeline
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
System Architecture
Raw Market Data
        ↓
Feature Engineering (volatility, returns, momentum, MA)
        ↓
Time-Series Sequence Builder (lookback → target horizon)
        ↓
Train / Validation / Test Split
        ↓
LSTM Neural Network Training
        ↓
Forecast Generation
        ↓
Performance Evaluation & Visualization
Install Dependencies
pip install -r requirements.txt
Run the Platform
python main.py
Model Overview
Why LSTM?

Financial markets are noisy, nonlinear, and sensitive to temporal dependencies.
LSTM models offer advantages:

Capture long-term patterns

Handle sequential volatility

Learn nonlinear relationships across indicators

Reduce vanishing gradient issues versus vanilla RNNs
License

