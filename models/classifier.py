from typing import cast
import torch
from pigmento import pnt  # 用于打印调试信息的工具库
from torch import nn
from loaders.env import Env


class ClassifierOutput:
    """
    分类器输出结果对象，统一包装 logits、标签、loss 以及附加信息（如 flags 和 valid_classes）。
    """
    def __init__(self, logits, labels, loss, flags=None, valid_classes=None):
        self.logits = logits              # 分类输出 logits [B, num_classes] 或 [B, 1+neg_num]
        self.labels = labels              # 真实标签
        self.loss = loss                  # loss 值
        self.flags = flags                # optional: 标注标签是否在有效池中
        self.valid_classes = valid_classes  # optional: 有效类别的 mask（如用于 AUC 评估时筛选）


class BaseClassifier(nn.Module):
    """
    分类器基类，定义通用接口。支持多种分类器变种的继承实现。
    """
    projector: nn.Linear  # 可选线性映射层（将 embedding 投影）

    def classify(self, target_x, target_y):
        """
        子类需实现此方法：给定目标向量 target_x 和对应标签 target_y 进行分类。

        :param target_x: 目标 embedding 向量 [num_targets, dim]
        :param target_y: 目标标签 [num_targets]
        :return: ClassifierOutput 对象
        """
        raise NotImplementedError

    def forward(self, embeds, labels):
        """
        分类器前向函数，从输入 embedding 和标签中提取目标位点并分类。

        :param embeds: 序列 embedding，形状 [B, T, D]
        :param labels: 标签序列 [B, T]，其中 Env.null 表示无效位
        :return: ClassifierOutput（logits、labels、loss 等）
        """
        # 先通过 projector 做 embedding 投影
        embeds = self.projector(embeds)

        batch_size, seq_len, _ = embeds.shape
        # 找出有效标签位置
        mask = cast(torch.Tensor, labels != Env.null)
        pos_indices = torch.nonzero(mask, as_tuple=False)  # [num_targets, 2]

        if pos_indices.shape[0] == 0:
            return None

        # 获取有效位置上的 embedding 和 label
        target_x = embeds[pos_indices[:, 0], pos_indices[:, 1]]  # [num_targets, dim]
        target_y = labels[pos_indices[:, 0], pos_indices[:, 1]]  # [num_targets]

        return self.classify(target_x, target_y)


class VanillaClassifier(BaseClassifier):
    """
    最普通的全连接分类器，直接接一个 Linear 层预测类别。
    """
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.num_classes: int = num_classes

        # 分类器权重，形状 [embedding_dim, num_classes]
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes, bias=False)

        # 投影层（可加可不加，这里加了一个 residual 投影）
        self.projector = nn.Linear(self.embedding_dim, self.embedding_dim)

        pnt(f'Classifier {self.embedding_dim, self.num_classes}')

    def classify(self, target_x, target_y):
        # 获取 logits，计算交叉熵损失
        logits = self.classifier(target_x)
        loss = nn.functional.cross_entropy(logits, target_y, ignore_index=Env.null)
        return ClassifierOutput(logits, target_y, loss)


class DotClassifier(BaseClassifier):
    """
    使用点积计算 logits 的分类器（适用于 item-level softmax 分类场景）。
    通常用于嵌入对齐任务，如推荐任务中 item embedding 作为分类权重。
    """
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_classes, self.embedding_dim = embedding.weight.shape
        self.embedding = embedding  # 分类权重为嵌入表中的每个 item 向量

        self.projector = nn.Linear(self.embedding_dim, self.embedding_dim)

    def classify(self, target_x, target_y):
        # 点积计算 logits，相当于 target_x 与所有类 embedding 计算相似度
        logits = target_x @ self.embedding.weight.T  # [num_targets, num_classes]
        loss = nn.functional.cross_entropy(logits, target_y)
        return ClassifierOutput(logits, target_y, loss)


class SamplingClassifier(DotClassifier):
    """
    负采样分类器，训练时只采样部分负样本计算 softmax，推理时全类比较。
    适用于大规模类别空间（如百万级别 item）。
    """
    def __init__(self, neg_num: int, temperature: float = 1.0, neg_pool: set = None, **kwargs):
        super().__init__(**kwargs)

        self.neg_num = neg_num                          # 每个正样本采样多少负样本
        self.temperature = temperature                  # softmax 温度调节因子

        # 负采样候选池
        self.neg_flag = neg_pool and torch.tensor(list(neg_pool)).to(Env.device)
        self.neg_pool = list(neg_pool) if neg_pool else list(range(self.num_classes))
        self.neg_pool = torch.tensor(self.neg_pool).to(Env.device)
        self.neg_pool_size = len(self.neg_pool)

        # 用于 AUC 评估时判断某个 label 是否来自于 neg_pool
        self.valid_classes = torch.zeros(self.num_classes, dtype=torch.bool).to(Env.device)
        valid_indices = torch.tensor(list(self.neg_pool), dtype=torch.long)
        self.valid_classes[valid_indices] = True

        # 设置全局 item 数量
        Env.num_items = self.neg_pool_size

    @property
    def global_validation(self):
        return Env.global_validation and Env.is_evaluating()

    def classify(self, target_x, target_y):
        if self.global_validation:
            # 推理阶段，使用全量类进行打分
            flags = None if self.neg_flag is None else torch.isin(target_y, self.neg_flag).long()
            co = super().classify(target_x, target_y)
            co.valid_classes = self.valid_classes
            co.flags = flags
            return co

        # 训练阶段：使用负采样
        num_targets = target_x.shape[0]

        # 从负样本池中采样 [num_targets, neg_num] 个负样本
        neg_samples = torch.randint(
            low=0,
            high=self.neg_pool_size,
            size=(num_targets, self.neg_num),
            dtype=torch.long
        ).to(Env.device)
        neg_samples = self.neg_pool[neg_samples]
        neg_embs = self.embedding(neg_samples)  # [B, neg_num, D]

        pos_embs = self.embedding(target_y)     # [B, D]
        flags = None if self.neg_flag is None else torch.isin(target_y, self.neg_flag).long()

        # 点积得到分数
        pos_score = torch.einsum('bd,bd->b', target_x, pos_embs).unsqueeze(-1)  # [B, 1]
        neg_score = torch.einsum('bd,bnd->bn', target_x, neg_embs)              # [B, neg_num]

        # 拼接正负样本，构造 softmax 分类器输入
        logits = torch.cat([pos_score, neg_score], dim=1).to(Env.device)        # [B, 1+neg_num]
        labels = torch.zeros(num_targets, dtype=torch.long).to(Env.device)      # 所有正样本都在 index=0

        # 使用 softmax 交叉熵计算损失
        loss = nn.functional.cross_entropy(logits / self.temperature, labels)

        return ClassifierOutput(logits, labels, loss, flags=flags, valid_classes=self.valid_classes)
