import torch
from torch import nn


class Embedding(nn.Module):
    """A class for the embedding layer.

    These embeddings are often used to represent textual data inputs.
    """

    def __init__(self, vocabulary_size: int, d_model: int, padding_idx: int = 0):
        """Embedding layer initialization.

        Args:
            vocabulary_size: Data vocabulary size (i.e. the number of embeddings to store).
            d_model: Embedding dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocabulary_size, d_model, padding_idx=padding_idx)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Embedding layer.

        Args:
            inputs: A tensor of shape (batch size, sequence length) representing raw inputs data.

        Returns:
            Tensor of shape (batch_size, sequence length, d_model) representing the inputs embeddings.
        """
        return self.embeddings(inputs)