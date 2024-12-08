import pytest
import torch
import torch.nn as nn

from src.model_evaluator import ModelEvaluator


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
def mock_labels():
    """Fixture to provide mock label mappings."""
    return {str(i): f"Class {i}" for i in range(10)}


@pytest.fixture
def input_tensor():
    """Fixture to create a sample input tensor."""
    return torch.rand(1, 3, 32, 32)  # A batch of one 32x32 RGB image


@pytest.fixture
def model_evaluator(mock_model, mock_labels):
    """Fixture to create a ModelEvaluator instance."""
    return ModelEvaluator(mock_model, mock_labels)


def test_initialization(mock_model, mock_labels):
    """Test initialization of ModelEvaluator."""
    evaluator = ModelEvaluator(mock_model, mock_labels)
    assert evaluator.model == mock_model
    assert evaluator.labels == mock_labels


def test_evaluate(mock_model, model_evaluator, input_tensor):
    """Test the evaluate method."""
    # Mock the model output
    mock_output = torch.tensor([[1.0, 2.0, 0.5, 3.0, 0.1, -1.0, 0.0, -0.5, 0.8, 2.5]])
    mock_model.forward = lambda x: mock_output

    predicted_label, confidence = model_evaluator.evaluate(input_tensor)
    assert predicted_label == "Class 3"  # Class with highest score
    assert confidence > 0  # Ensure the confidence is positive


def test_evaluate_invalid_labels(mock_model):
    """Test evaluation when labels do not match model output indices."""
    invalid_labels = {"5": "Class 5", "6": "Class 6"}  # Missing most labels
    evaluator = ModelEvaluator(mock_model, invalid_labels)

    # Mock the model output
    mock_output = torch.tensor([[1.0, 2.0, 0.5, 3.0, 0.1, -1.0, 0.0, -0.5, 0.8, 2.5]])
    mock_model.forward = lambda x: mock_output

    input_tensor = torch.rand(1, 3, 32, 32)
    with pytest.raises(KeyError):
        evaluator.evaluate(input_tensor)


def test_evaluate_confidence(mock_model, model_evaluator, input_tensor):
    """Test confidence score calculation."""
    # Mock the model output
    mock_output = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    mock_model.forward = lambda x: mock_output

    _, confidence = model_evaluator.evaluate(input_tensor)
    assert isinstance(confidence, float)  # Ensure confidence is a scalar
    assert 0 <= confidence <= 100  # Confidence should be in the range [0, 100]


def test_evaluate_batch_input(mock_model, model_evaluator):
    """Test evaluation on a batch of images."""
    batch_input = torch.rand(5, 3, 32, 32)  # Batch of 5 images

    # Mock the model output
    mock_output = torch.tensor(
        [
            [1.0, 2.0, 0.5, 3.0, 0.1, -1.0, 0.0, -0.5, 0.8, 2.5],
            [0.5, 1.5, 0.5, 0.5, 0.1, 0.0, 2.0, 0.5, 1.0, 3.0],
            [2.0, 1.0, 0.5, 0.0, 1.0, -1.0, -0.5, -0.5, 0.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    mock_model.forward = lambda x: mock_output

    for i in range(batch_input.size(0)):
        predicted_label, confidence = model_evaluator.evaluate(
            batch_input[i].unsqueeze(0)
        )
        assert isinstance(predicted_label, str)
        assert 0 <= confidence <= 100
