import torch
import torch.nn.functional as F


class ModelEvaluator:
    """
    Evaluates a model's predictions and handles adversarial attacks.
    """

    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

    def evaluate(self, input_tensor):
        """
        Evaluate the model's prediction for a given input.

        Args:
                input_tensor: Preprocessed image tensor.

        Returns:
                Tuple: (Predicted label, confidence score)
        """
        with torch.no_grad():
            output = self.model(input_tensor)
            _, index = torch.max(output, 1)
            confidence = F.softmax(output, dim=1)[0] * 100
            return self.labels[str(int(index[0]))], confidence[index[0]].item()
