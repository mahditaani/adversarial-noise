import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.image_processor import ImageProcessor  # Replace with the correct file name


@pytest.fixture
def sample_image():
    """Fixture to create a sample PIL image."""
    img = Image.new("RGB", (100, 100), color="red")
    return img


@pytest.fixture
def sample_tensor():
    """Fixture to create a sample tensor image."""
    return torch.rand(3, 224, 224)


@pytest.fixture
def processor():
    """Fixture to create an ImageProcessor instance."""
    return ImageProcessor()


def test_initialization_without_transforms():
    """Test default initialization of ImageProcessor."""
    processor = ImageProcessor()
    assert processor.transform is not None
    assert processor.inverse_transform is not None


def test_initialization_with_invalid_transforms():
    """Test initialization with only one of the transforms provided."""
    transform = transforms.ToTensor()
    with pytest.raises(ValueError):
        ImageProcessor(transform=transform)


def test_process_pil_image(sample_image, processor):
    """Test processing a PIL image."""
    processed_tensor = processor.process(sample_image)
    assert isinstance(processed_tensor, torch.Tensor)
    assert processed_tensor.shape == (1, 3, 224, 224)


def test_process_tensor(sample_tensor, processor):
    """Test processing a tensor."""
    processed_tensor = processor.process(sample_tensor)
    assert isinstance(processed_tensor, torch.Tensor)
    assert processed_tensor.shape == (1, 3, 224, 224)


def test_process_tensor_4d(sample_tensor, processor):
    """Test processing a 4D tensor (batch of images)."""
    batch_tensor = sample_tensor.unsqueeze(0)  # Add batch dimension
    processed_tensor = processor.process(batch_tensor)
    assert torch.equal(processed_tensor, batch_tensor)


def test_process_invalid_input(processor):
    """Test processing an invalid input type."""
    with pytest.raises(ValueError):
        processor.process(12345)  # Invalid input type


def test_inverse_process(sample_image, processor):
    """Test inverse processing a tensor."""
    # Simulate the expected input tensor after processing
    processed_tensor = processor.process(sample_image)
    inversed_image = processor.inverse_process(processed_tensor)
    assert isinstance(inversed_image, Image.Image)

