import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        max_length: int,
        value_length: int,
        dropout: float = 0.1,
    ):
        """
        Here, we implement the full Transformer model.

        The Transformer model will take in a source sequence
        and a target sequence and will output a sequence of
        logits representing the next token in the target.

        The Transformer model will consist of an Encoder and a
        Decoder. The Encoder will take in the source sequence
        and will output an encoded representation of the source.

        The Decoder will take in the target sequence and the
        encoded representation of the source and will output
        a sequence of logits representing the next token in
        the target sequence.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # Hint: This should be relatively simple, as you've
        # already implemented the Encoder and Decoder layers.
        # Check the `Attention Is All You Need` paper for guidance.
        self.encoder = Encoder(
            vocab_size,
            num_layers,
            num_heads,
            ffn_hidden_dim,
            embedding_dim,
            qk_length,
            value_length,
            max_length,
            dropout,
        )

        self.decoder = Decoder(
            vocab_size,
            num_layers,
            num_heads,
            ffn_hidden_dim,
            embedding_dim,
            qk_length,
            value_length,
            max_length,
            dropout,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Transformer model.

        Args:
            src: torch.Tensor with shape (B, T1) representing the source tokens
            tgt: torch.Tensor with shape (B, T2) representing the target tokens

        Returns:
            torch.Tensor with shape (B, T2, C) representing the output logits
        """
        enc_x, src_mask = self.encoder(src)
        dec_x = self.decoder(tgt, enc_x, src_mask)
        return dec_x
