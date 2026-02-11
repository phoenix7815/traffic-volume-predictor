from dataloader import DataLoader
from train import train_lstm
from eval_loss import MSELoss, MAELoss, HuberLoss
from utils.pems import create_index

DATA_PATH = "data/PEMS03/PEMS03_data.csv"

if __name__ == "__main__":
    # if dataset is from PEMS dir
    pems_index = create_index("2018-09-01 00:00:00","2018-11-30 23:55:00", "5min")
    data_loader = DataLoader(DATA_PATH, sensor_split=0.1, time_split=0.1, index=pems_index)


    train_data = data_loader.get_train_data()
    test_seen_data = data_loader.get_test_seen_sensors_data()
    test_unseen_data = data_loader.get_test_unseen_sensors_data()

    # Train with MSE Loss
    mae_loss = MAELoss()
    model_mse, train_losses_mse = train_lstm(mae_loss, train_data, model_name="LSTM_MSE", epochs=5, lr_rate=1e-3)