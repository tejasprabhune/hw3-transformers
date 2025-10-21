import math

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, FeedForwardNN


class PositionalEncoding(nn.Module):
    """
    The PositionalEncoding layer will take in an input tensor
    of shape (B, T, C) and will output a tensor of the same
    shape, but with positional encodings added to the input.

    We provide you with the full implementation for this
    homework.

    Based on:
        https://web.archive.org/web/20230315052215/https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
        """Initialize the PositionalEncoding layer."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape (B, T, C)
        """
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float,
    ):
        """
        Each encoder layer will take in an embedding of
        shape (B, T, C) and will output an encoded representation
        of the same shape.

        The encoder layer will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        self.multi_head_attention = MultiHeadAttention(
            num_heads, embedding_dim, qk_length, value_length
        )
        self.feed_forward_nn = FeedForwardNN(embedding_dim, ffn_hidden_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the EncoderLayer.
        """
        attention_output = self.multi_head_attention(x, x, x)
        attention_output = self.dropout(attention_output)
        x = self.layer_norm1(x + attention_output)
        feed_forward_output = self.feed_forward_nn(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = self.layer_norm2(x + feed_forward_output)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        max_length: int,
        dropout: float,
    ):
        """
        Remember that the encoder will take in a sequence
        of tokens and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Encoder layers,
        each of which will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer. For this, we
        need to specify the number of layers and the number of
        heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
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
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers in some kind of list comprehension.
        #
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer encoder.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, dropout, max_length
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    num_heads,
                    embedding_dim,
                    ffn_hidden_dim,
                    qk_length,
                    value_length,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Encoder.
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x
