#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
import optuna

# Step 1: Load and preprocess the data
def load_data(file_path, sequence_length, horizon=1, features=None):
    # Load the dataset
    df = pd.read_csv("A.csv")
    if features is None:
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values

    # Scale data to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - horizon + 1):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i:i+horizon, 3])  # Target: next M days of Closing price
    X, y = np.array(X), np.array(y)

    # Split into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# Step 2: Define the RNN-based model
def build_model(model_type, input_shape, units=64, dropout=0.2):
    model = Sequential()
    if model_type == 'RNN':
        model.add(SimpleRNN(units, activation='tanh', return_sequences=False, input_shape=input_shape))
    elif model_type == 'GRU':
        model.add(GRU(units, return_sequences=False, input_shape=input_shape, dropout=dropout))
    elif model_type == 'LSTM':
        model.add(LSTM(units, return_sequences=False, input_shape=input_shape, dropout=dropout))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 3: Train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    return history

# Step 4: Evaluate and plot results
def evaluate_model(model, X_test, y_test, scaler, horizon=1):
    # Generate predictions
    predictions = model.predict(X_test)

    # Check dimensions
    print(f"Predictions shape: {predictions.shape}")
    print(f"y_test shape before processing: {y_test.shape}")

    # Handle multi-horizon targets by selecting the first horizon or averaging
    if y_test.shape[1] > 1:  # Multi-horizon case
        y_test = y_test[:, 0].reshape(-1, 1)  # Select the first horizon
        print("Using the first horizon for evaluation.")

    # Ensure predictions and y_test have matching dimensions
    assert predictions.shape[0] == y_test.shape[0], "Predictions and y_test lengths do not match!"

    # Prepare padding for inverse_transform
    padding = np.zeros((predictions.shape[0], len(scaler.min_) - 1))
    print(f"Padding shape: {padding.shape}")

    # Rescale predictions to original scale
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate([padding, predictions], axis=1)
    )[:, -1]  # Take the last column (target feature)

    # Rescale y_test to original scale
    y_test_rescaled = scaler.inverse_transform(
        np.concatenate([padding, y_test], axis=1)
    )[:, -1]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled)
    return rmse, mape, predictions_rescaled, y_test_rescaled

# Step 5: Hyperparameter Tuning
def optimize_hyperparameters(X_train, y_train, X_val, y_val, input_shape, model_type):
    def objective(trial):
        units = trial.suggest_int('units', 32, 128)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        model = build_model(model_type, input_shape, units, dropout)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64, callbacks=[early_stopping], verbose=0)
        return min(history.history['val_loss'])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params

# Step 6: Main execution
if __name__ == "__main__":
    file_path = "A.csv"  # Update with your dataset path
    sequence_length = 30
    horizon = 3  # Predict the next 3 days

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data(file_path, sequence_length, horizon)

    # Hyperparameter tuning for LSTM
    best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, (X_train.shape[1], X_train.shape[2]), 'LSTM')
    print(f"Best Hyperparameters: {best_params}")

    # Train and evaluate each model
    for model_type in ['RNN', 'GRU', 'LSTM']:
        print(f"Training {model_type} model...")
        model = build_model(model_type, (X_train.shape[1], X_train.shape[2]), units=best_params['units'], dropout=best_params['dropout'])
        history = train_model(model, X_train, y_train, X_val, y_val)

        # Evaluate model
        rmse, mape, predictions, actuals = evaluate_model(model, X_test, y_test, scaler, horizon)
        print(f"{model_type} RMSE: {rmse}, MAPE: {mape}")

        # Plot predictions
        plt.figure(figsize=(10, 5))
        plt.plot(actuals[:100], label='Actual Prices')
        plt.plot(predictions[:100], label='Predicted Prices')
        plt.title(f'{model_type} Predictions (First 100 Test Points)')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()


# In[ ]:




