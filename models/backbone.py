from typing import Literal, Optional

import torch
from torch import nn


class BackboneConfig:
    """
    模型配置类，主要用于保存 backbone 模型的结构参数。
    该类的实例会被传入 BackboneModel 中，用于控制网络结构。
    """
    def __init__(
            self,
            num_layers,       # 层数（如 transformer 层数量）
            embedding_dim,    # 嵌入维度
            max_length,       # 序列最大长度
            **kwargs,         # 兼容扩展参数
    ):
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.max_length = max_length


class BackboneModel(nn.Module):
    """
    基础模型类，所有 backbone 模型都应继承此类。
    提供统一的接口，约束 forward 输入格式。
    """
    config: BackboneConfig                        # 模型配置实例
    config_class = BackboneConfig                 # 配置类，用于动态构建配置

    def __init__(self, config):
        """
        初始化模型，接收 BackboneConfig 配置对象。
        """
        super().__init__()
        self.config = config

    def forward(
            self,
            inputs_embeds: Optional[torch.FloatTensor] = None,       # 输入的 embedding 向量 [B, T, D]
            attention_mask: Optional[torch.FloatTensor] = None,      # 注意力掩码 [B, 1, T, T] 或 [B, T]
    ):
        """
        子类需要实现该方法，用于定义前向传播过程。
        """
        raise NotImplementedError
