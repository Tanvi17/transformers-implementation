import torch
import torch.nn as nn
import utils


class Decoder(nn.Module):
    """Decoder component with stack of N identical layers."""

    def __init__(self, layer: nn.Module, number_of_layers: int) -> None:
        super().__init__()
        self.layers = utils.layer_clones(layer, number_of_layers)
        self.normalization = utils.LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tenor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pass the input and mask through each decoder layer and apply normalization.

        Args:
            x (torch.Tensor): Input tensor.
            memory (torch.Tensor): Memory tensor.
            source_mask (torch.Tensor): Mask tensor for source.
            target_mask (torch.Tensor): Mask tensor for target."""
        for layer in self.layer:
            x = layer(x, memory, source_mask, target_mask)
        return self.normalization(x)
