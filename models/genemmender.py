from typing import cast

import torch
from torch import nn

from loaders.embedding_layer import EmbeddingLayer
from loaders.env import Env
from loaders.symbols import Symbols as Syb
from models.backbone import BackboneModel
from models.classifier import ClassifierOutput, SamplingClassifier, VanillaClassifier


class GenemmenderOutput:
    """
    推荐器 Genemmender 的输出封装类，统一管理 action/item 分类器输出，以及 seen_mask（历史曝光屏蔽）。
    """
    def __init__(self, action_co: ClassifierOutput, item_co: ClassifierOutput, seen_mask):
        self.action_co = action_co        # 行为（如点击/点赞）分类器输出
        self.item_co = item_co            # item 分类器输出（top-K 推荐）
        self.seen_mask = seen_mask        # 曝光 mask（是否在历史序列中出现）

    def get_loss(self, aw, iw):
        """
        加权计算总 loss（action_loss + item_loss）。

        :param aw: 行为 loss 的权重
        :param iw: item loss 的权重
        :return: 总 loss 值
        """
        action_loss = self.action_co and aw * self.action_co.loss
        item_loss = self.item_co and iw * self.item_co.loss
        zero = torch.tensor(0.0).to(Env.device)
        return (action_loss or zero) + (item_loss or zero)

    def prepare_evaluation(self):
        """
        准备评估指标：计算每个正样本的排名位置（rank）。

        - 按 logits 从高到低排序，取出正样本的排名
        - 可选地剔除无效类别（如 valid_classes 和 seen_mask）

        :return:
            ranks: list[int] 每个样本的排名（rank）
            flags: list[int] 是否是有效类（用于过滤 AUC/NDCG 评估）
        """
        logits = self.item_co.logits

        # 屏蔽无效类别（如 sampling 中未采样的类别）
        if self.item_co.valid_classes is not None and Env.global_validation:
            logits = logits.masked_fill(~self.item_co.valid_classes, -float('inf'))

        # 屏蔽历史上出现过的 item（只评估未见过的推荐）
        if self.seen_mask is not None:
            logits = logits.masked_fill(self.seen_mask, -float('inf'))

        # 排序，取 label 的 rank
        sorted_scores, sorted_indices = logits.sort(dim=1, descending=True)
        label_positions = cast(torch.Tensor, sorted_indices == self.item_co.labels.unsqueeze(1))
        ranks = (label_positions.nonzero(as_tuple=False)[:, 1] + 1).cpu().tolist()

        # 标记当前 label 是否属于 sampling 的有效集合（如 eval AUC 时排除冷门）
        flags = self.item_co.flags
        if flags is not None:
            flags = flags.cpu().tolist()
        return ranks, flags


class Genemmender(nn.Module):
    """
    主模型类：封装整个 embedding + backbone + classifier 的 forward 流程。

    - 支持两个分类器：一个用于行为预测，一个用于 item 推荐。
    - 可支持 group embedding（如 user-group 聚合）。
    - 可选启用 seen_mask，在评估阶段屏蔽历史 item。
    """
    def __init__(
            self,
            el: EmbeddingLayer,                 # Embedding 层
            backbone: BackboneModel,            # 编码器（Transformer/LSTM）
            action_classifier: VanillaClassifier,
            item_classifier: SamplingClassifier,
            use_mask_seen: bool,                # 是否在评估时屏蔽历史 item
            global_validation: bool = True,    # 是否全局验证（影响 seen_mask 的处理方式）
            use_segment_ids: bool = False,
            use_separators: bool = False,
    ):
        super().__init__()

        self.el = el
        self.backbone = backbone
        self.action_classifier = action_classifier
        self.item_classifier = item_classifier
        self.use_mask_seen = use_mask_seen
        self.global_validation = global_validation
        self.use_segment_ids = use_segment_ids
        self.use_separators = use_separators

        self.history_start_index = 3 if self.use_separators else 1

    def forward(self, batch: dict):
        """
        前向计算函数：完成完整的嵌入 -> 编码 -> 分类器预测流程。

        :param batch: 一个批次的输入字典，包含 input_ids, labels, mask 等
        :return: GenemmenderOutput（包含预测结果和屏蔽信息）
        """
        # 从 batch 中读取输入字段，并转移到 device 上
        input_ids = batch[Syb.input_ids].to(Env.device)
        type_ids = batch[Syb.type_ids].to(Env.device)
        seg_ids = batch[Syb.seg_ids].to(Env.device)
        feature_ids = batch[Syb.feature_ids]
        action_labels = batch[Syb.action_labels].to(Env.device)
        item_labels = batch[Syb.item_labels].to(Env.device)
        attention_mask = batch[Syb.attention_mask].to(Env.device)

        # 获取 input embedding（包括输入 token、type、以及特征）
        input_embeds = self.el(input_ids, type_ids, feature_ids)  # [B, L, D]

        if self.use_segment_ids:
            seg_embeds = self.el.get_segment_embeddings(seg_ids, attention_mask)
            input_embeds += seg_embeds

        # 如果使用 group embedding，则进行融合
        if Syb.group_length in batch:
            group_embeds = self.el.integrate_group_feature(
                group_length=batch[Syb.group_length],
                group_index=batch[Syb.group_index],
                group_sequence=batch[Syb.group_seq],
            )  # [B, K, D]

            if group_embeds is not None:
                # 将 group embedding 加到输入 embedding 的对应位置（通常是位置 1~K）
                group_length = group_embeds.shape[1]
                input_embeds[:, self.history_start_index:self.history_start_index + group_length, :] += group_embeds

        # 编码（Transformer/LSTM 等）
        last_hidden_state = self.backbone(input_embeds, attention_mask=attention_mask)  # [B, L, D]

        # 行为预测分类器（如点击、收藏等）
        action_co: ClassifierOutput = self.action_classifier(last_hidden_state, action_labels)

        # item 推荐分类器（采样或全类别）
        item_co: ClassifierOutput = self.item_classifier(last_hidden_state, item_labels)

        seen_mask = None
        # 如果配置了历史屏蔽，并提供了 Syb.seen_mask，就构造 mask
        if Syb.seen_mask in batch and self.use_mask_seen and self.global_validation:
            seen_mask = batch[Syb.seen_mask].to(Env.device)  # [B, K, N] = [batch_size, max_eval_nums, item_size]
            seen_mask_count = batch[Syb.seen_mask_count].to(Env.device)  # 每个样本实际的有效个数 [B]

            batch_size, max_eval_nums, item_size = seen_mask.shape

            # 构造 mask：只保留有效 eval 数的那几条（可能每个样本不一样）
            mask = torch.arange(max_eval_nums).to(Env.device).unsqueeze(0) < seen_mask_count.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand(-1, -1, item_size).reshape(-1, item_size)  # [B*K, N]
            seen_mask = seen_mask.reshape(-1, item_size)  # [B*K, N]
            seen_mask = seen_mask[mask.any(dim=1)]  # 去掉空的条目
            assert seen_mask.shape[0] == item_co.logits.shape[0]

        return GenemmenderOutput(action_co=action_co, item_co=item_co, seen_mask=seen_mask)
