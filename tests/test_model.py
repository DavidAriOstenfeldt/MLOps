import torch
import pytest
from src.models.model import MyAwesomeModel


class TestClass:
    model = MyAwesomeModel("corruptmnist")

    def test_model(self):
        assert self.model(torch.rand(1, 1, 28, 28)).shape == (
            1,
            10,
        ), "Model gives wrong dimensions"

    def test_error_on_wrong_shape(self):
        with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
            self.model(torch.rand(1, 28, 28)).shape == (1, 10)

        with pytest.raises(
            ValueError, match="Expected each sample to have shape \[1, 28, 28\]"
        ):
            self.model(torch.rand(1, 3, 28, 28)).shape == (1, 10)
            self.model(torch.rand(1, 1, 6, 28)).shape == (1, 10)
            self.model(torch.rand(1, 1, 28, 6)).shape == (1, 10)