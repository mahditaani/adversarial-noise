import pytest
import torch
import torch.nn as nn

from src.noise_generator import AdversarialNoiseGenerator


class MockModel(nn.Module):
    """A simple mock model for testing purposes."""

    def __init__(self):
        super(MockModel, self).__init__()
        self.fc = nn.Linear(3 * 32 * 32, 10)  # Example for 10 classes and 32x32 image

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)


@pytest.fixture
def mock_model():
    """Fixture to create a mock model."""
    model = MockModel()
    return model


@pytest.fixture
def input_tensor():
    """Fixture to create a sample input tensor."""
    return torch.rand(1, 3, 32, 32)  # A batch of one 32x32 RGB image


@pytest.fixture
def noise_generator(mock_model):
    """Fixture to create an AdversarialNoiseGenerator instance."""
    return AdversarialNoiseGenerator(
        mock_model, target_class=1, epsilon=0.01, iterations=5
    )


def test_initialization(mock_model):
    """Test initialization of AdversarialNoiseGenerator."""
    generator = AdversarialNoiseGenerator(
        mock_model, target_class=1, epsilon=0.01, iterations=10
    )
    assert generator.model == mock_model
    assert generator.target_class == 1
    assert generator.epsilon == 0.01
    assert generator.iterations == 10


def test_generate_static_epsilon(noise_generator, input_tensor):
    """Test adversarial noise generation with static epsilon."""
    perturbed_image = noise_generator.generate(input_tensor)
    assert isinstance(perturbed_image, torch.Tensor)
    assert perturbed_image.shape == input_tensor.shape
    assert not torch.equal(
        input_tensor, perturbed_image
    )  # Ensure the input is perturbed


def test_generate_dynamic_epsilon(noise_generator, input_tensor):
    """Test adversarial noise generation with dynamic epsilon."""
    perturbed_image = noise_generator.generate(input_tensor, dynamic_epsilon=True)
    assert isinstance(perturbed_image, torch.Tensor)
    assert perturbed_image.shape == input_tensor.shape
    assert not torch.equal(
        input_tensor, perturbed_image
    )  # Ensure the input is perturbed


def test_invalid_input_tensor(noise_generator):
    """Test handling of invalid input tensor."""
    with pytest.raises(RuntimeError):
        noise_generator.generate(torch.rand(3, 32, 32))  # Missing batch dimension
