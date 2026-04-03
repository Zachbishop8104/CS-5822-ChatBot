import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=1024):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embed(positions)

# Simple self attention head
# communication between tokens
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, max_seq_len=1024):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        seq_len = x.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ V


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.projection(attended)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len=1024):
        super().__init__()
        
        # create multiple attention heads and a final projection layer
        self.attention = CausalSelfAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.attention(x)

# computation within each token
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        # expand and contract the embedding dimension with a nonlinearity in between
        # expanding to 4x the embeding size in the middle gives the model more room to learn complex transformations
        # before compressing it back down
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# x -> LayerNorm -> Attention -> add input back
# x -> LayerNorm -> FeedForward -> add input back
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, max_seq_len=max_seq_len)
        self.feed_forward = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Stop the model from overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=max_seq_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.positional_encoding(self.embedding(x))
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return self.output_projection(x)