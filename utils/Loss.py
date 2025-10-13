import numpy as np

class Loss:
    def calculate(self, output, y_true):
        sample_losses = self.forward(output, y_true)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        # Calculate the mean squared error
        sample_losses = (y_true - y_pred) ** 2
        return  np.mean(sample_losses, axis=-1)


    def backward(self, y_pred, y_true):
        # Number of samples
        samples = len(y_true)
        # Gradient with respect to predictions
        outputs = y_pred.shape[1]
        self.dinputs = -2 * (y_true - y_pred) / outputs
        self.dinputs = self.dinputs / samples
