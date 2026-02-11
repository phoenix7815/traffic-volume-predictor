import torch
import torch.nn as nn

models: dict[str, nn.Module] = {}

def predict(model_name:str, test_data: torch.Tensor) -> torch.Tensor:
    model = models[model_name]
    if model is None:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
    return predictions

def load_models(model_paths: dict[str, str]) -> None:
    for model_name, path in model_paths.items():
        try:
            model = torch.load(path)
            models[model_name] = model
            print(f"Loaded model '{model_name}' from {path}")
        except Exception as e:
            print(f"Error loading model '{model_name}' from {path}: {e}")



if __name__ == "__main__":
    model_paths = {
        "LSTM_MSE": "disk/LSTM_MSE.pth",
    }
    load_models(model_paths)

    test_data = torch.tensor([162.0, 120.0, 162.0, 120.0, 135.0, 151.0], dtype=torch.float32)
    test_data = test_data.unsqueeze(0).unsqueeze(-1)  # Shape: (1, 6, 1)
    predictions = predict("LSTM_MSE", test_data)
    print(predictions)