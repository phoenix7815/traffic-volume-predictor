import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import os
from model.LSTM import LSTM


def create_sequences(data, input_size, output_size=1):
    """
    Create sequences for time series prediction
    Args:
        data: numpy array of shape (n_timesteps,) for a single sensor
        input_size: number of past timesteps to use as input
        output_size: number of future timesteps to predict
    Returns:
        X: input sequences of shape (n_samples, input_size)
        y: target values of shape (n_samples, output_size)
    """
    X, y = [], []
    for i in range(len(data) - input_size - output_size + 1):
        X.append(data[i:i + input_size])
        y.append(data[i + input_size:i + input_size + output_size])
    return np.array(X), np.array(y)


def train_lstm(loss, data:pd.DataFrame, input_size:int = 6, output_size: int = 1,
               model_name:str = "LSTM_default", epochs:int = 5, lr_rate:float = 0.001):
    """
    Train LSTM model on time series data from multiple sensors

    Args:
        loss: Loss function object (MSELoss, MAELoss, or HuberLoss)
        data: pandas DataFrame with sensors as columns and timesteps as rows
        input_size: number of past timesteps to use as input (sequence length)
        output_size: number of values to predict (typically 1 for next timestep)
        model_name: name to save the model file
        epochs: number of training epochs
        lr_rate: learning rate for optimizer

    Returns:
        model: trained LSTM model
        train_losses: list of training losses per epoch
    """

    # Create disk folder if it doesn't exist
    os.makedirs('disk', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data from all sensors
    X_train_list = []
    y_train_list = []

    print("Preparing time series sequences for each sensor...")
    for sensor in data.columns:
        sensor_data = data[sensor].values
        # Create sequences for this sensor
        X_sensor, y_sensor = create_sequences(sensor_data, input_size, output_size)
        X_train_list.append(X_sensor)
        y_train_list.append(y_sensor)

    # Concatenate data from all sensors
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    print(f"Total training samples: {len(X_train)}")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(device)  # Shape: (n_samples, input_size, 1)
    y_train = torch.FloatTensor(y_train).to(device)  # Shape: (n_samples, output_size)

    # Initialize model
    model = LSTM(
        input_size=1,  # Each timestep has 1 feature (traffic volume)
        hidden_size=64,  # Hidden layer size
        num_layers=2,  # Number of LSTM layers
        output_size=output_size
    ).to(device)

    print(f"Model architecture:\n{model}")

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # Training loop
    train_losses = []
    batch_size = 32

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Loss function: {loss.name}")
    print(f"Learning rate: {lr_rate}")
    print(f"Batch size: {batch_size}\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            # Forward pass
            predictions = model(batch_X)
            batch_loss = loss(predictions, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

    # Save model
    model_path = os.path.join('disk', f'{model_name}.pth')
    torch.save(model,model_path)
    print(f"\nModel saved to: {model_path}")

    return model, train_losses


