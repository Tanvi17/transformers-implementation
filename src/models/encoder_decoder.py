from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    Standard Encoder-Decoder architecture.

    This class represents a standard Encoder-Decoder architecture used in sequence-to-sequence models.
    It consists of an encoder, a decoder, source and target embeddings, and a generator.

    Args:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        source_embedding (nn.Module): The source embedding module.
        target_embedding (nn.Module): The target embedding module.
        generator (nn.Module): The generator module.

    Attributes:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        source_embedding (nn.Module): The source embedding module.
        target_embedding (nn.Module): The target embedding module.
        generator (nn.Module): The generator module.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        source_embedding: nn.Module,
        target_embedding: nn.Module,
        generator: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.generator = generator

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Takes in and processes masked source and target sequences.

        Args:
            source (torch.Tensor): The source sequence.
            target (torch.Tensor): The target sequence.
            source_mask (torch.Tensor): The source mask.
            target_mask (torch.Tensor): The target mask.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.decode(
            self.encode(source, source_mask), source_mask, target, target_mask
        )

    def encode(self, source: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """Encodes the source sequence.

        Args:
            source (torch.Tensor): The source sequence.
            source_mask (torch.Tensor): The source mask.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        return self.encoder(self.source_embedding(source), source_mask)

    def decode(
        self,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes the target sequence.

        Args:
            memory (torch.Tensor): The memory tensor.
            source_mask (torch.Tensor): The source mask.
            target (torch.Tensor): The target sequence.
            target_mask (torch.Tensor): The target mask.

        Returns:
            torch.Tensor: The decoded tensor.
        """
        return self.decoder(
            self.target_embedding(target), memory, source_mask, target_mask
        )


class Generator(nn.Module):
    """Define standard Linear layer and softmax function"""

    def __init__(self, d_model: int, vocab: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, seq_len, d_model) where d_model is 512 (default value from the transformer paper)
        return F.log_softmax(self.projection(x), dim=-1)
