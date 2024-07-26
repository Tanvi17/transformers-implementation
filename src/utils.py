import copy
import torch
import torch.nn as nn


def layer_clones(module: nn.Module, number_of_layers: int) -> nn.ModuleList:
    """Clones a module n times. Produces a ModuleList containing N layers.

    Args:
        module (nn.Module): The module to clone.
        N (int): The number of clones to create.

    Returns:
        nn.ModuleList: The cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(number_of_layers)])


class LayerNorm(nn.Module):
    """Builds a layer normalization module."""

    def __init__(self, features: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization."""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.epsilon) + self.shift


class SubLayerConnection(nn.Module):
    """A residual connection followed by a layer norm."""

    def __init__(self, size: int, dropout_rate: float) -> None:
        super().__init__()
        self.norm_layer = LayerNorm(size)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """Applies the residual connection and layer normalization."""
        return x + self.dropout_layer(sublayer(self.norm_layer(x)))
