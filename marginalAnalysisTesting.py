import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate sample data
quarters = pd.date_range(start='2015-01-01', periods=40, freq='QE')  # Changed 'Q' to 'QE'

data = pd.DataFrame({
    'GDP_Growth': np.random.normal(2.0, 0.5, size=40),
    'CPI': np.random.normal(2.5, 0.3, size=40),
    'Unemployment': np.random.normal(5.0, 0.4, size=40),
    'Interest_Rate': np.random.normal(1.5, 0.2, size=40)
}, index=quarters)

# Save data
data.to_csv("macro_data.csv")
print("Sample data:")
print(data.head())

def preprocess_data(data):
    # Handle missing values
    data = data.fillna(method='ffill')
    
    # Remove outliers
    for col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        data[col] = data[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    
    return data

# Preprocess and normalize the data
data = preprocess_data(data)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, target, seq_len=5):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)

def create_model(input_shape, lstm_units=32):
    return Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units),
        Dense(1)
    ])

def evaluate_feature_set(feature_indices, verbose=0, lstm_units=32):
    selected_data = scaled_data[:, feature_indices]
    target = scaled_data[:, 0]  # assume GDP_growth is the target
    
    X, y = create_sequences(selected_data, target)
    tscv = TimeSeriesSplit(n_splits=3)
    
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = create_model((X.shape[1], X.shape[2]), lstm_units)
        model.compile(optimizer='adam', loss='mse')
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=8,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        preds = model.predict(X_val)
        cv_scores.append(mean_squared_error(y_val, preds))
    
    return np.mean(cv_scores)

# Baseline: all features
all_features = [0, 1, 2, 3]  # GDP_growth, CPI, Unemployment, Interest_rate
baseline_loss = evaluate_feature_set(all_features)
print(f"\nBaseline MSE (all features): {baseline_loss:.4f}")

# Marginal analysis: exclude one feature at a time
feature_names = ['GDP_growth', 'CPI', 'Unemployment', 'Interest_rate']
marginal_impacts = []

print("\nMarginal Analysis Results:")
for i in range(len(all_features)):
    subset = [j for j in all_features if j != i]
    loss = evaluate_feature_set(subset)
    marginal_impact = loss - baseline_loss
    marginal_impacts.append(marginal_impact)
    print(f"Excluding {feature_names[i]} → MSE: {loss:.4f} | Marginal Δ: {marginal_impact:.4f}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names, marginal_impacts)
plt.title('Feature Importance (Marginal Impact on MSE)')
plt.xticks(rotation=45)
plt.ylabel('Increase in MSE when excluded')
plt.tight_layout()
plt.show()