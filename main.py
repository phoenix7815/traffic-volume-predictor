from dataloader import DataLoader
from train import train_lstm
from eval_loss import MSELoss, MAELoss, HuberLoss

DATA_PATH = "data/traffic.csv"

if __name__ == "__main__":
    data_loader = DataLoader(DATA_PATH, sensor_split=0.1, time_split=0.1)
    train_data = data_loader.get_train_data()
    test_seen_data = data_loader.get_test_seen_sensors_data()
    test_unseen_data = data_loader.get_test_unseen_sensors_data()

    # Train with MSE Loss
    mae_loss = MAELoss()
    model_mse, train_losses_mse = train_lstm(mae_loss, train_data, model_name="LSTM_MSE", epochs=20, lr_rate=1e-3)