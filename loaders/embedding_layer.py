from typing import cast, Literal

import torch
from torch import nn

from loaders.column_map import ColumnMap
from loaders.datasets.backbone import BackboneDataset
from loaders.embedding_hub import EmbeddingHub
from loaders.env import Env


class EmbeddingLayer(nn.Module):
    """
    EmbeddingLayer 模块用于将多个特征列的 ID 输入映射为嵌入向量，并进行池化聚合。
    同时支持额外的特征（如 tag、group）进行嵌入，适配序列推荐场景。
    """

    def __init__(
            self,
            eh: EmbeddingHub,
            pooling: Literal['sum', 'mean', 'max'] = 'sum',
            decay_rate: float = 1.0,
            fusion: str = 'sum'
    ):
        """
        :param eh: EmbeddingHub 实例，封装了所有的嵌入表
        :param pooling: 多特征聚合方式：sum / mean / max
        """
        super().__init__()

        self.eh = eh
        self.cm: ColumnMap = eh.cm
        self.table = self.eh.get_table()
        self.pooling = pooling
        self.decay_rate = decay_rate

        assert fusion in ['sum', 'concat'], f'Unsupported fusion method: {fusion}'
        self.fusion = fusion

        self.item_proj = self.concat_proj = None
        if self.fusion == 'concat':
            self.item_proj = nn.Linear(self.eh.embedding_dim, self.eh.embedding_dim)
            self.concat_proj = nn.Linear(self.eh.embedding_dim * 2, self.eh.embedding_dim)

    def forward(self, input_ids: torch.Tensor, type_ids: torch.Tensor, feature_ids: dict):
        """
        主前向过程，输入主 token ids、对应 type ids 和额外的 feature ids，输出最终的 token embedding。
        :param input_ids: [B, L, F] or [B, L]，主 token id
        :param type_ids: [B, L, F] or [B, L]，每个 token 的类型索引（如 user/item/action）
        :param seg_ids: [B, L, F] or [B, L]，每个 token 的 segment id
        :param feature_ids: dict[str, Tensor]，每种额外特征名称对应的 index tensor
        :return: 最终 token embedding，形状 [B, L, D]
        """
        if len(input_ids.shape) == 2:
            # 若为 2D，升维为 3D（单个 token 的多特征）
            input_ids = input_ids.unsqueeze(2)
            type_ids = type_ids.unsqueeze(2)

        batch_size, seq_len, num_feat = input_ids.shape
        input_embeds = torch.zeros(batch_size, seq_len, num_feat, self.eh.embedding_dim, dtype=torch.float).to(Env.device)

        num_types = self.cm.types.size
        for i in range(num_types):
            mask = cast(torch.Tensor, type_ids == i)  # 取出当前类型的位置
            if not mask.any():
                continue

            input_ids_ = input_ids * mask.long()  # 非当前类型位置置零
            col_name = self.cm.types[i]

            embedding = self[col_name](input_ids_)  # lookup embedding
            embedding *= mask.unsqueeze(-1).float()  # 遮蔽
            input_embeds += embedding

        if self.pooling == 'sum':
            input_embeds = input_embeds.sum(dim=2)
        elif self.pooling == 'max':
            input_embeds = input_embeds.max(dim=2)[0]
        else:
            num_valid_features = type_ids.ne(-1).sum(dim=2, keepdim=True).float()
            input_embeds = input_embeds.sum(dim=2) / (num_valid_features + 1e-8)

        feature_embeds = torch.zeros(batch_size, seq_len, self.eh.embedding_dim, dtype=torch.float).to(Env.device)
        for index, feature in enumerate(feature_ids):
            feature_indices = feature_ids[feature].to(Env.device)  # [B, L] 或 [B, L, F]
            mask = cast(torch.Tensor, 1 - feature_indices == Env.null)
            feature_indices = feature_indices * mask.long()

            embedding = self[feature](feature_indices)
            embedding *= mask.unsqueeze(-1).float()

            if len(embedding.shape) == 4:  # [B, L, F, D]
                num_valid_features = mask.sum(dim=2, keepdim=True).float()
                embedding = embedding.sum(dim=2) / (num_valid_features + 1e-8)  # 平均池化

            feature_embeds += embedding

        return input_embeds + feature_embeds

    def get_segment_embeddings(self, seg_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seg_ids *= attention_mask.long()

        seg_embeds = self[BackboneDataset.SEG_VOCAB.name](seg_ids.to(Env.device))
        seg_embeds = seg_embeds * attention_mask.unsqueeze(-1).float().to(Env.device)
        return seg_embeds

    def integrate_group_feature(self, group_length: torch.Tensor, group_index: torch.Tensor, group_sequence: torch.Tensor):
        """
        聚合 group 信息的嵌入（如兴趣簇、历史行为簇等）

        :param group_length: 每个 group 的长度 [B]
        :param group_index: 每个用户包含的 group 索引 [B, K]
        :param group_sequence: 每个 group 的行为序列 [B, K, L]
        :return: 每个 group 的向量表示 [B, K, D]
        """
        group_index = group_index.to(Env.device)
        group_index_mask = cast(torch.Tensor, group_index != Env.null).to(Env.device)
        group_index = group_index * group_index_mask.long()
        group_index_embedding = self[self.cm.history_group_col](group_index)
        group_index_embedding = group_index_embedding * group_index_mask.unsqueeze(-1).float()

        group_sequence = group_sequence.to(Env.device)
        group_sequence_mask = cast(torch.Tensor, group_sequence != Env.null).to(Env.device)
        group_sequence = group_sequence * group_sequence_mask.long()

        group_sequence_embedding = self[self.cm.items_col](group_sequence)
        group_sequence_embedding = group_sequence_embedding * group_sequence_mask.unsqueeze(-1).float()

        if self.item_proj is not None:
            group_sequence_embedding = self.item_proj(group_sequence_embedding)

        if group_sequence.dim() != 3:
            return None
        seq_len = group_sequence.size(2)  # [B, G, L]

        decay_weights = self.decay_rate ** torch.arange(0, seq_len, device=Env.device).float()
        decay_weights = decay_weights.view(1, 1, seq_len, 1)

        decay_weights = decay_weights * group_sequence_mask.unsqueeze(-1).float()

        # 按权重加权平均
        weighted_sum = (group_sequence_embedding * decay_weights).sum(dim=2)
        weight_sum = decay_weights.sum(dim=2) + 1e-8
        group_sequence_embedding = weighted_sum / weight_sum

        if self.concat_proj is not None:
            group_embedding = self.concat_proj(torch.cat([group_index_embedding, group_sequence_embedding], dim=-1))
        else:
            group_embedding = group_index_embedding + group_sequence_embedding
        group_embedding = group_embedding * group_index_mask.unsqueeze(-1).float()

        return group_embedding

    def __getitem__(self, key) -> nn.Embedding:
        """
        按 key 获取对应的 embedding layer。
        """
        key = self.eh.get_key(key)
        if key not in self.table:
            raise ValueError(f'{key} is not registered in the embedding table.')
        return cast(nn.Embedding, self.table[key])

    def __contains__(self, key):
        """
        判断某个 embedding key 是否已经注册。
        """
        key = self.eh.get_key(key)
        return key in self.table
