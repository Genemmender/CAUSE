from typing import Optional

import torch
import torch.nn as nn

from models.backbone import BackboneConfig, BackboneModel
from models.modules.hstu import HSTULayer


class HSTUConfig(BackboneConfig):
    """
    HSTU 模型的配置类，继承自 BackboneConfig，额外加入 num_heads（多头注意力头数）。
    """
    def __init__(self, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads


class HSTUModel(BackboneModel):
    """
    多层 HSTU 层组成的序列建模主干模型。
    支持自定义 attention mask，适用于推荐中的因果建模（Causal Attention）。
    """
    config: HSTUConfig
    config_class = HSTUConfig

    def __init__(self, config: HSTUConfig):
        super().__init__(config=config)

        # 初始化多层 HSTU Transformer，每层结构由 HSTULayer 定义
        self.layers = nn.ModuleList([
            HSTULayer(
                embed_dim=config.embedding_dim,
                num_heads=config.num_heads,
                feedforward_dim=None  # 使用默认的 4x hidden dim
            ) for _ in range(config.num_layers)
        ])

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,     # 输入嵌入向量 [B, T, D]
        attention_mask: Optional[torch.FloatTensor] = None     # 可选 mask: [B, T]
    ):
        """
        Args:
            inputs_embeds: 输入嵌入，shape [batch_size, seq_len, embed_dim]
            attention_mask: 掩码张量，shape [batch_size, seq_len]，1 表示有效 token，0 表示 padding

        Returns:
            最后一层输出表示，shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        outputs = inputs_embeds

        # ----------- 构造 attention mask（Causal + Padding）-----------
        if attention_mask is None:
            # 如果未传入 mask，默认所有位置有效
            attention_mask = torch.ones(batch_size, seq_len, device=inputs_embeds.device)

        # padding_mask: shape [B, 1, 1, T]，后续可 broadcast 到注意力矩阵
        padding_mask = attention_mask[:, None, None, :]  # [B, 1, 1, T]

        # causal_mask: shape [1, 1, T, T]，保证自回归结构（只能看自己和过去）
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=inputs_embeds.device)
        ).unsqueeze(0).unsqueeze(0)

        # combined_mask: shape [B, 1, T, T]
        combined_mask = causal_mask * padding_mask  # padding 屏蔽未来位

        # 将 masked 掩码填充为 -inf，未 masked 为 0，用于 softmax 后置 0
        combined_mask = (1.0 - combined_mask) * torch.finfo(inputs_embeds.dtype).min  # float16/32 兼容

        # ----------- 输入多层 HSTULayer -----------
        for layer in self.layers:
            outputs = layer(outputs, combined_mask)  # 每层使用同样的 mask
        return outputs  # 最终输出 shape [B, T, D]
