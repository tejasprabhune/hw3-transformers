import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention, FeedForwardNN
from .encoder import PositionalEncoding
from seq2seq.data.fr_en import tokenizer


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float = 0.1,
    ):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        self.self_attention = MultiHeadAttention(
            num_heads, embedding_dim, qk_length, value_length
        )
        self.cross_attention = MultiHeadAttention(
            num_heads, embedding_dim, qk_length, value_length
        )
        self.feed_forward_nn = FeedForwardNN(embedding_dim, ffn_hidden_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None,
        tgt_mask: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The forward pass of the DecoderLayer.
        """
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        self_attention_output = self.dropout(self_attention_output)
        x = self.layer_norm1(x + self_attention_output)

        # If this is seq2seq, also need a cross-attention block
        if enc_x is not None:
            cross_attention_output = self.cross_attention(x, enc_x, enc_x, src_mask)
            cross_attention_output = self.dropout(cross_attention_output)
            x = self.layer_norm2(x + cross_attention_output)

        feed_forward_output = self.feed_forward_nn(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = self.layer_norm3(x + feed_forward_output)

        return x


class Decoder(nn.Module):
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
        dropout: float = 0.1,
    ):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers
        and the number of heads.

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
        # build out the Transformer decoder.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, dropout, max_length
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
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
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a mask to prevent attention to future tokens.
        """
        B, T, C = x.size()
        ones = torch.ones((1, T, T))
        out = torch.tril(ones, 1)

        return out

    def forward(
        self, x: torch.Tensor, enc_x: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """
        tgt_padding_mask = (x != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

        look_ahead_mask = self.make_mask(x).to(x.device)

        tgt_mask = tgt_padding_mask & look_ahead_mask

        src_padding_mask = None
        if enc_x is not None:
            src_padding_mask = (
                (enc_x != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
            )

        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_x, tgt_mask, src_padding_mask)

        x = self.fc(x)
        return x
