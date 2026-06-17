import torch

from src.models import SmallCNN


def test_small_cnn_output_shape():
    model = SmallCNN(num_classes=2)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)

    assert y.shape == (4, 2)
