import torch
from torch import nn


class RelativePositionalBias(nn.Module):
    """
    相对位置偏置模块（可学习），用于加到注意力分数上。
    支持最大长度为 max_position 的相对位置偏移，共有 num_heads 个头。
    """
    def __init__(self, max_position: int, num_heads: int):
        super().__init__()
        self.max_position = max_position
        self.num_heads = num_heads

        # 参数矩阵：[2 * max_position - 1, num_heads]
        # 表示从 -(max-1) 到 +(max-1) 之间所有可能相对距离的偏置
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_position - 1, num_heads))
        nn.init.xavier_uniform_(self.rel_pos_bias)  # 初始化参数

    def forward(self, seq_len: int):
        """
        构造 (seq_len, seq_len) 的相对位置矩阵，并返回对应偏置值。

        返回形状为 (num_heads, seq_len, seq_len)，用于加到注意力分数上。
        """
        positions = torch.arange(seq_len, dtype=torch.long)  # [0, 1, ..., seq_len-1]
        rel_positions = positions[None, :] - positions[:, None]  # 计算相对位置 (T, T)
        rel_positions += self.max_position - 1  # 平移为非负索引
        rel_positions.clamp_(0, 2 * self.max_position - 2)  # 防止越界
        bias = self.rel_pos_bias[rel_positions]  # 查表：(T, T, num_heads)
        return bias.permute(2, 0, 1)  # -> (num_heads, T, T)


class HSTULayer(nn.Module):
    """
    HSTU Layer 是一个轻量 Transformer 层：
    包含标准的注意力机制（带相对位置编码）+ 前馈网络 + 三个 LayerNorm。

    支持注意力掩码（attention_mask）作为输入。
    """
    def __init__(self, embed_dim, num_heads, max_position=512, feedforward_dim=None, dropout=0.1):
        super().__init__()
        if feedforward_dim is None:
            feedforward_dim = 4 * embed_dim  # 默认前馈层扩大 4 倍维度

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        # Linear 预处理：可选（比如做残差前线性映射）
        self.linear_projection = nn.Linear(embed_dim, embed_dim)

        # QKV 投影层（共输出 3D）
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 相对位置偏置模块
        self.rel_pos_bias = RelativePositionalBias(max_position, num_heads)

        # 前馈网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )

        # 三个 LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        """
        输入:
            x: [B, T, D]
            attention_mask: [B, 1, T, T] 或 [B, num_heads, T, T]
        返回:
            x_ffn: [B, T, D]
        """
        batch_size, seq_len, _ = x.size()

        # 第一个 LayerNorm 和线性预投影
        x_proj = self.linear_projection(self.norm1(x))

        # 第二个 LayerNorm 和 QKV 投影
        qkv_input = self.norm2(x_proj)
        qkv = self.qkv_proj(qkv_input)  # -> [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # 分为 Q/K/V

        # reshape 为多头格式：[B, num_heads, T, head_dim]
        def shape(x):
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = shape(q)
        k = shape(k)
        v = shape(v)

        # Scaled Dot-Product Attention 分数：[B, num_heads, T, T]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 加相对位置偏置
        rel_bias = self.rel_pos_bias(seq_len)  # [num_heads, T, T]
        attn_scores += rel_bias.unsqueeze(0)  # [1, num_heads, T, T]

        # 加 attention mask（常见用于 padding mask 或 causal mask）
        attn_scores += attention_mask  # [B, num_heads, T, T]

        # Softmax + Attention 输出
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, T, head_dim]

        # 合并 heads -> [B, T, D]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)  # 输出线性变换

        # 残差连接 + Dropout
        x_attn = x_proj + self.dropout_attn(attn_output)

        # 第三个 LayerNorm + FFN + 残差
        x_ffn = x_attn + self.dropout_ffn(self.ffn(self.norm3(x_attn)))

        return x_ffn  # [B, T, D]
