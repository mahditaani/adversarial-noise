from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.image_processor import ImageProcessor
from src.model_evaluator import ModelEvaluator
from src.noise_generator import AdversarialNoiseGenerator


class AdversarialAttackPipeline:
    """
    High-level pipeline to perform adversarial attacks and visualize results.
    """

    def __init__(
        self,
        model,
        labels,
        transform=None,
        inverse_transform=None,
        epsilon=0.01,
        iterations=10,
    ):
        self.image_processor = ImageProcessor(transform, inverse_transform)
        self.evaluator = ModelEvaluator(model, labels)
        self.noise_generator = AdversarialNoiseGenerator(
            model, target_class=None, epsilon=epsilon, iterations=iterations
        )

    def perform_attack(self, input_image, target_class, dynamic_epsilon=False):
        """
        Perform an adversarial attack on the input image.

        Args:
                input_image: File path, PIL image, or preprocessed tensor.
                target_class: Target class for the adversarial attack.
                dynamic_epsilon: If True, use gradient-magnitude-based epsilon.

        Returns:
                Tuple: (Original image tensor, Adversarial image tensor)
        """
        input_tensor = self.image_processor.process(input_image)
        self.noise_generator.target_class = target_class
        adversarial_tensor = self.noise_generator.generate(
            input_tensor, dynamic_epsilon
        )
        return input_tensor, adversarial_tensor

    def visualize_attack(
        self,
        input_tensor,
        adversarial_tensor,
        ax: Optional[List[Axes]] = None,
        show: bool = False,
    ):
        """
        Visualize the original and adversarial images side by side.

        Args:
                input_tensor: Tensor of the original image.
                adversarial_tensor: Tensor of the adversarial image.
        """
        original_img = self.image_processor.inverse_process(input_tensor)
        adversarial_img = self.image_processor.inverse_process(adversarial_tensor)

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        if len(ax) != 2:
            raise ValueError("The number of subplots must be 2.")

        ax[0].set_title("Original Image")
        ax[0].imshow(original_img)
        ax[0].axis("off")

        ax[1].set_title("Adversarial Image")
        ax[1].imshow(adversarial_img)
        ax[1].axis("off")

        if show:
            plt.show()
