import torch
from PIL import Image
from torchvision import transforms


class ImageProcessor:
    """
    Handles image transformations and inverse transformations for visualization.
    """

    def __init__(self, transform=None, inverse_transform=None):
        # ensure both transform and inverse_transform are not None or are both None
        if any([transform, inverse_transform]) and not all(
            [transform, inverse_transform]
        ):
            raise ValueError("Both transform and inverse_transform must be provided.")

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.inverse_transform = inverse_transform or transforms.Compose(
            [
                transforms.Normalize(
                    mean=[
                        -0.485 / 0.229,
                        -0.456 / 0.224,
                        -0.406 / 0.225,
                    ],  # Note corrected std divisor for mean
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
                transforms.ToPILImage(),
            ]
        )

    def process(self, input_image):
        """
        Process an input image (file path, PIL image, or tensor).

        Args:
                input_image: Input image in one of the following forms:
                - File path to the image
                - PIL Image object
                - Preprocessed tensor

        Returns:
                torch.Tensor: Preprocessed image tensor.
        """
        if isinstance(input_image, str):  # File path
            img = Image.open(input_image).convert(
                "RGB"
            )  # so we can ignore alpha channel
        elif isinstance(input_image, Image.Image):  # PIL Image
            img = input_image.convert("RGB")  # so we can ignore alpha channel
        elif isinstance(input_image, torch.Tensor):  # Already a tensor
            return input_image.unsqueeze(0) if input_image.ndim == 3 else input_image
        else:
            raise ValueError(
                "Unsupported input type. Provide file path, PIL image, or tensor."
            )

        img_t = self.transform(img)
        return img_t.unsqueeze(0)

    def inverse_process(self, tensor):
        """
        Apply the inverse transformation to a tensor for visualization.

        Args:
                tensor: Torch tensor to be transformed back into an image.

        Returns:
                PIL Image: The transformed image.
        """
        return self.inverse_transform(tensor.squeeze())
