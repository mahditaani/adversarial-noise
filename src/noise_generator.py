import torch
import torch.nn.functional as F


class AdversarialNoiseGenerator:
    """
    Generates adversarial noise using various strategies.
    """

    def __init__(self, model, target_class, epsilon=0.01, iterations=10):
        self.model = model
        self.target_class = target_class
        self.epsilon = epsilon
        self.iterations = iterations

    def generate(self, input_tensor, dynamic_epsilon=False):
        """
        Generate adversarial noise.

        Args:
                input_tensor: Input image tensor.
                dynamic_epsilon: If True, use gradient-magnitude-based epsilon.

        Returns:
                torch.Tensor: Adversarially perturbed image tensor.
        """
        x_adv = input_tensor.clone().detach().requires_grad_(True)

        for _ in range(self.iterations):
            outputs = self.model(x_adv)
            loss = -F.cross_entropy(
                outputs, torch.tensor([self.target_class], dtype=torch.long)
            )
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                if dynamic_epsilon:
                    grad_magnitude = x_adv.grad.abs()
                    adjusted_epsilon = (
                        self.epsilon * grad_magnitude / grad_magnitude.mean()
                    )
                    x_adv += adjusted_epsilon * x_adv.grad.sign()
                else:
                    x_adv += self.epsilon * x_adv.grad.sign()

            x_adv.requires_grad_(True)

        return x_adv
