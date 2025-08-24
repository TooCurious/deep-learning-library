import torch
from torch import nn

from model.embeddings import Embedding

class SkipGram(nn.Module):

    def __init__(self, vocabulary_size: int, d_model: int):

        super().__init__()

        self.input_embeddings = Embedding(vocabulary_size, d_model)
        self.output_embeddings = Embedding(vocabulary_size, d_model)


    def forward(self, center_word_idxs: torch.Tensor) -> torch.Tensor:

        center_embedding = self.input_embeddings(center_word_idxs)
        all_embedding = next(self.output_embeddings.parameters())

        logits = torch.matmul(center_embedding, all_embedding.T)

        return logits








