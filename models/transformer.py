from typing import Optional, cast

import torch
from transformers import GPT2Config, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from loaders.env import Env
from models.backbone import BackboneConfig, BackboneModel


class TransformerConfig(BackboneConfig):
    """
    Transformer 配置类，继承自通用 BackboneConfig。
    包括 Transformer 特有的参数：注意力头数 num_heads。
    """
    def __init__(self, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads


class TransformerModel(BackboneModel):
    """
    使用 GPT2Model 的 Transformer 编码器，用于序列建模。
    实现位置编码、Causal Attention Mask、自定义嵌入输入等。
    """
    config: TransformerConfig
    config_class = TransformerConfig

    def __init__(self, config):
        super().__init__(config=config)

        # 创建 GPT2 配置（不使用词表，因此 vocab_size=2 是 dummy 值）
        gpt_config = GPT2Config(
            vocab_size=2,  # 实际不使用词嵌入，inputs_embeds 自行提供
            n_positions=self.config.max_length,   # 最大序列长度
            n_embd=self.config.embedding_dim,     # 嵌入维度
            n_layer=self.config.num_layers,       # Transformer 层数
            n_head=self.config.num_heads,         # 注意力头数
        )
        self.model = GPT2Model(gpt_config)  # 初始化 GPT2 模型实例

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入向量 [B, T, D]
        attention_mask: Optional[torch.FloatTensor] = None, # padding 掩码 [B, T]
    ):
        input_shape = inputs_embeds.size()[:-1]     # [B, T]
        batch_size = inputs_embeds.shape[0]

        # 创建位置编码索引（shared across batch）
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=Env.device)  # [T]
        position_ids = position_ids.unsqueeze(0)  # [1, T]

        # 获取 GPT2 模型自带的位置嵌入（位置编码）[1, T, D]
        position_embeds = self.model.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds  # 将位置编码加到输入中

        # 展平 attention_mask [B, T] -> [B, 1, 1, T] 用于 broadcast
        attention_mask = attention_mask.view(batch_size, -1)

        # 创建 causal mask（上三角置 0），用于自回归建模 [1, 1, T, T]
        seq_len = attention_mask.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_mask.device))

        attention_mask = attention_mask[:, None, None, :]      # [B, 1, 1, T]
        causal_mask = causal_mask[None, None, :, :]            # [1, 1, T, T]

        # 合并 mask：padding_mask * causal_mask -> [B, 1, T, T]
        combined_mask = causal_mask * attention_mask

        # 转为模型 dtype 并构造 final mask，填 -inf 表示不关注
        combined_mask = combined_mask.to(dtype=self.model.dtype)
        attention_mask = (1.0 - combined_mask) * torch.finfo(self.model.dtype).min

        # 通过 dropout 处理输入
        hidden_states = self.model.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)  # [B, T, D]

        # 遍历 transformer blocks，依次前向
        for i in range(len(self.model.h)):
            block = cast(GPT2Block, self.model.h[i])

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,  # shape [B, 1, T, T]
            )

            hidden_states = outputs[0]  # 取出输出 hidden_state

        # 最后层 LayerNorm
        hidden_states = self.model.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return hidden_states  # 返回最终编码结果 [B, T, D]
