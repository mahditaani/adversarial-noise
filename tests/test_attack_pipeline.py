from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.attack_pipeline import AdversarialAttackPipeline


@pytest.fixture
def mock_model():
    """Fixture to create a mock model."""
    mock = MagicMock()
    mock.forward = MagicMock(return_value=torch.rand(1, 10))  # Simulate model output
    return mock


@pytest.fixture
def mock_labels():
    """Fixture to provide mock label mappings."""
    return {str(i): f"Class {i}" for i in range(10)}


@pytest.fixture
def mock_image_processor():
    """Fixture to mock the ImageProcessor class."""
    mock_processor = MagicMock()
    mock_processor.process = MagicMock(return_value=torch.rand(1, 3, 224, 224))
    mock_processor.inverse_process = MagicMock(
        return_value=Image.new("RGB", (224, 224))
    )
    return mock_processor


@pytest.fixture
def mock_noise_generator():
    """Fixture to mock the AdversarialNoiseGenerator class."""
    mock_generator = MagicMock()
    mock_generator.generate = MagicMock(return_value=torch.rand(1, 3, 224, 224))
    return mock_generator


@pytest.fixture
def adversarial_pipeline(
    mock_model, mock_labels, mock_image_processor, mock_noise_generator
) -> AdversarialAttackPipeline:
    """Fixture to create an AdversarialAttackPipeline instance with mocked dependencies."""
    pipeline = AdversarialAttackPipeline(
        mock_model, mock_labels, epsilon=0.01, iterations=10
    )
    pipeline.image_processor = mock_image_processor
    pipeline.noise_generator = mock_noise_generator
    return pipeline


def test_pipeline_initialization(mock_model, mock_labels):
    """Test initialization of the AdversarialAttackPipeline."""
    pipeline = AdversarialAttackPipeline(
        mock_model, mock_labels, epsilon=0.02, iterations=5
    )
    assert pipeline.evaluator.model == mock_model
    assert pipeline.evaluator.labels == mock_labels
    assert pipeline.noise_generator.epsilon == 0.02
    assert pipeline.noise_generator.iterations == 5


def test_perform_attack(
    adversarial_pipeline, mock_image_processor, mock_noise_generator
):
    """Test the perform_attack method."""
    input_image = "path/to/image.jpg"
    target_class = 3

    input_tensor, adversarial_tensor = adversarial_pipeline.perform_attack(
        input_image, target_class
    )

    # Assertions
    mock_image_processor.process.assert_called_once_with(input_image)
    mock_noise_generator.generate.assert_called_once_with(
        mock_image_processor.process.return_value, False
    )
    assert input_tensor.shape == adversarial_tensor.shape


def test_perform_attack_with_dynamic_epsilon(
    adversarial_pipeline, mock_image_processor, mock_noise_generator
):
    """Test the perform_attack method with dynamic epsilon."""
    input_image = "path/to/image.jpg"
    target_class = 3

    adversarial_pipeline.perform_attack(input_image, target_class, dynamic_epsilon=True)

    # Assertions
    mock_noise_generator.generate.assert_called_once_with(
        mock_image_processor.process.return_value, True
    )


def test_visualize_attack(adversarial_pipeline, mock_image_processor):
    """Test the visualize_attack method."""
    input_tensor = MagicMock()
    adversarial_tensor = MagicMock()

    adversarial_pipeline.visualize_attack(input_tensor, adversarial_tensor)

    # Assertions
    mock_image_processor.inverse_process.assert_any_call(input_tensor)
    mock_image_processor.inverse_process.assert_any_call(adversarial_tensor)


def test_visualize_attack_with_ax(adversarial_pipeline, mock_image_processor):

    input_tensor = MagicMock()
    adversarial_tensor = MagicMock()

    input_image = MagicMock()
    adversarial_image = MagicMock()

    def mock_image_processor_return(val):
        if val is input_tensor:
            return input_image
        elif val is adversarial_tensor:
            return adversarial_image
        else:
            return None

    mock_image_processor.inverse_process.side_effect = mock_image_processor_return

    ax = [MagicMock(), MagicMock()]

    adversarial_pipeline.visualize_attack(input_tensor, adversarial_tensor, ax=ax)

    # Assertions
    ax[0].imshow.assert_called_once_with(input_image)
    ax[1].imshow.assert_called_once_with(adversarial_image)


def test_visualize_attack_invalid_ax(adversarial_pipeline):
    """Test the visualize_attack method with an invalid number of subplots."""
    input_tensor = MagicMock()
    adversarial_tensor = MagicMock()

    with pytest.raises(ValueError):
        adversarial_pipeline.visualize_attack(
            input_tensor, adversarial_tensor, ax=[MagicMock()]
        )


def test_visualize_attack_creates_ax(adversarial_pipeline):
    """Test the visualize_attack method when ax is None."""
    input_tensor = MagicMock()
    adversarial_tensor = MagicMock()

    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        mock_fig, mock_ax = MagicMock(), [MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_ax)
        adversarial_pipeline.visualize_attack(input_tensor, adversarial_tensor)

        # Assertions
        mock_subplots.assert_called_once_with(1, 2, figsize=(10, 5))

        mock_ax[0].imshow.assert_called_once()
        mock_ax[1].imshow.assert_called_once()
