import torch.nn as nn

class MSELoss:

    def __init__(self):
        self.name = "MSE"
        self.loss_fn = nn.MSELoss()

    def __call__(self, predictions, targets):
        return self.loss_fn(predictions, targets)

    def compute(self, predictions, targets):
        return self.__call__(predictions, targets)


class MAELoss:

    def __init__(self):
        self.name = "MAE"
        self.loss_fn = nn.L1Loss()

    def __call__(self, predictions, targets):
        return self.loss_fn(predictions, targets)

    def compute(self, predictions, targets):
        return self.__call__(predictions, targets)


class HuberLoss:

    def __init__(self, delta=1e-4):
        """
        Initialize Huber Loss
        Args:
            delta: float - threshold parameter (default: 1.0)
                  When |y_pred - y_true| < delta, uses MSE
                  When |y_pred - y_true| >= delta, uses MAE
        """
        self.name = "Huber"
        self.delta = delta
        self.loss_fn = nn.HuberLoss(delta=delta)

    def __call__(self, predictions, targets):
        return self.loss_fn(predictions, targets)

    def compute(self, predictions, targets):
        return self.__call__(predictions, targets)

