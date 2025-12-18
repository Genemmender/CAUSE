from typing import Union, Optional

import torch
from pigmento import pnt
from torch.utils.data import Dataset
from tqdm import tqdm
from unitok import UniTok, Symbol, Vocab

from loaders.column_map import ColumnMap
from loaders.env import Env
from loaders.symbols import Symbols as Syb


class BackboneDataset(Dataset):
    """
    推荐系统通用数据集基类，支持序列建模和群体表示（group-based embedding）。
    接收 UniTok 序列数据和 item 特征，生成模型可用的输入格式。
    """

    SEG_USER_ID = 'user'
    SEG_ITEMS = 'items'
    SEG_HISTORY = 'history'

    SEG_VOCAB = Vocab(name='segment')
    SEG_VOCAB.append(SEG_USER_ID)
    SEG_VOCAB.append(SEG_ITEMS)
    SEG_VOCAB.append(SEG_HISTORY)

    SEG_USER_ID_INDEX = SEG_VOCAB[SEG_USER_ID]
    SEG_ITEMS_INDEX = SEG_VOCAB[SEG_ITEMS]
    SEG_HISTORY_INDEX = SEG_VOCAB[SEG_HISTORY]

    def __init__(
            self,
            ut: UniTok,
            mode: Symbol,
            cm: ColumnMap,
            history_config,
            item_ut: UniTok = None,
            selector: Optional[Union[bool, tuple]] = None,
            use_separators: bool = False,
    ):
        """
        初始化 BackboneDataset

        :param ut: 主序列 UniTok 数据集
        :param cm: ColumnMap，用于字段名注册及 vocab 映射
        :param history_config: 历史行为配置
        :param item_ut: item 特征 UniTok（可选）
        :param selector: 特征选择器（用于 item_ut 中特征过滤）
        """
        super().__init__()
        self.ut = ut
        self.cm = cm
        self.mode = mode
        self.use_separators = use_separators

        self.item_size = self.ut.meta.jobs[self.cm.items_col].tokenizer.vocab.size
        self.user_type = self.cm.types[self.cm.user_col]
        self.item_type = self.cm.types[self.cm.items_col]
        self.label_type = self.cm.types[self.cm.actions_col]
        self.sep_type = self.cm.types.append(self.SEG_VOCAB.name)

        self.item_ut = item_ut
        self.selector = selector

        self.cached_item_features = [None] * len(self.ut)

        self.max_seq_item_num = self.cm.get_max_length(self.cm.items_col)
        self.max_length = self.get_max_length()

        self.history_config = history_config
        self.history_group_col = self.history_config.group
        self.history_skip_group = self.history_config.skip_group or 0

        if history_config.activate:
            assert history_config.group, ValueError('group column is required when history is activated')
            # assert not self.item_ut.meta.jobs[self.history_group].return_list, ValueError('history group should be a single-value column')
            self.cm.history_group_col = self.history_group_col

        self.max_groups = self.history_config.max_groups or 0
        self.max_items_per_group = self.history_config.max_items_per_group or 32
        self.history_cache = None

        assert self.max_length - 1 - self.max_groups > 0, ValueError('max_groups is too large')

        if self.history_config.activate:
            self.build_history_group()

        self.max_eval_nums = self.get_max_eval_nums()
        pnt(f'Max Eval Nums: {self.max_eval_nums}')

    @property
    def global_validation(self):
        return Env.global_validation and self.mode is not Syb.train

    def get_max_eval_nums(self):
        """
        计算每条样本中最大评估数（用于构造 seen_mask）
        """
        eval_nums = 0
        for evals in self.ut.data[self.cm.evals_col]:
            eval_nums = max(eval_nums, sum(evals))
        return eval_nums

    def _build_history_group(self, index):
        """
        构造某一条样本的群体历史行为，按 group 聚合。
        """
        history = self.ut[index][self.cm.histories_col]
        hg_vocab = self.item_ut.meta.vocabularies[self.history_group_col]
        group_dict, group_vocab = dict(), dict()

        for item in history[::-1]:
            group_feats = self.item_ut[item][self.history_group_col]

            if not self.item_ut.meta.features[self.history_group_col].return_list:
                group_feats = [group_feats]

            for group_feat in group_feats:
                if str(hg_vocab[group_feat]) == '-1':
                    continue

                if group_feat not in group_dict:
                    group_vocab[len(group_vocab)] = group_feat
                    group_dict[group_feat] = []
                group_dict[group_feat].append(item)

        group_sequence = [group_dict[group_vocab[i]] for i in range(len(group_vocab))]
        group_index = [int(group_vocab[i]) for i in range(len(group_vocab))]
        return group_sequence, group_index

    def build_history_group(self):
        """
        批量构建所有样本的群体历史行为结构，用于 group-based embedding。
        自动截断 group 数量和 group 内 item 数量。
        """
        max_group = 0  # 最大的输入到用户序列的 group token 个数
        max_items_per_group = 0  # 每个 group 中最大的 item 数量

        group_sequence_series = []
        group_index_series = []

        pnt(f'Building history groups for {len(self.ut)} sequences...')
        for index in tqdm(range(len(self.ut)), total=len(self.ut)):
            group_sequence, group_index = self._build_history_group(index)
            group_sequence = group_sequence[self.history_skip_group:]  # type: list[list]
            group_index = group_index[self.history_skip_group:]  # type: list

            max_group = max(max_group, len(group_index))

            if self.max_groups and len(group_index) > self.max_groups:
                group_sequence = group_sequence[:self.max_groups]
                group_index = group_index[:self.max_groups]

            for i, gs in enumerate(group_sequence):
                max_items_per_group = max(max_items_per_group, len(gs))
                if self.max_items_per_group and len(gs) > self.max_items_per_group:
                    group_sequence[i] = gs[:self.max_items_per_group]

            group_sequence_series.append(group_sequence)
            group_index_series.append(group_index)

        pnt(f'Original Max Group: {max_group} -> {self.max_groups or "Unchanged"}')
        pnt(f'Original Max Items Per Group: {max_items_per_group} -> {self.max_items_per_group or "Unchanged"}')

        self.max_groups = self.max_groups or max_group
        self.max_items_per_group = self.max_items_per_group or max_items_per_group

        group_length_series = []
        for idx in range(len(group_sequence_series)):
            group_sequence, group_index = group_sequence_series[idx], group_index_series[idx]
            group_length_series.append(len(group_sequence))

            group_index = group_index + [Env.null] * (self.max_groups - len(group_index))
            for i, gs in enumerate(group_sequence):
                group_sequence[i] = gs + [Env.null] * (self.max_items_per_group - len(gs))
            group_sequence = group_sequence + [[Env.null] * self.max_items_per_group] * (self.max_groups - len(group_sequence))

            group_sequence_series[idx] = group_sequence
            group_index_series[idx] = group_index

        self.history_cache = (group_length_series, group_index_series, group_sequence_series)

    def get_max_length(self):
        """
        获取最大输入序列长度，包含 item 序列 + 当前 user
        """
        return self.max_seq_item_num + 1 + (3 if self.use_separators else 0)

    def pad(self, tokens):
        """
        对 token 序列进行 padding，支持 1D 或 2D 列表
        """
        assert len(tokens) <= self.max_length, ValueError(f'Tokens length {len(tokens)} exceeds max length {self.max_length}')
        append_len = self.max_length - len(tokens)
        append = [[Env.null] * len(tokens[0])] * append_len if isinstance(tokens[0], list) else [Env.null] * append_len
        return torch.tensor(tokens + append, dtype=torch.long)

    def build_inputs(
            self,
            group_length: int,
            user_id: int,
            item_seq: list,
            action_label_seq: list,
            eval_seq: list,
            feature_seq: dict,
            seen_mask: torch.Tensor,
    ):
        """
        子类实现：根据基础字段构造模型输入字段（input_ids/type_ids/labels等）
        """
        raise NotImplementedError

    def init_feature_ids(self, feature_seq):
        """
        初始化 item 侧特征序列的 tensor 结构（用于填充）
        """
        feature_ids = dict()
        for feature in feature_seq:
            feature_ids[feature] = []
        self.fill_feature_ids(feature_ids, feature_seq)
        return feature_ids

    def fill_feature_ids(self, feature_ids, feature_seq):
        """
        按照 feature_seq 的结构，填充空值 Env.null
        """
        for feature in feature_seq:
            job = self.item_ut.meta.jobs[feature]
            if not job.return_list:
                feature_ids[feature].append(Env.null)
            else:
                feature_ids[feature].append([Env.null] * job.max_len)

    def load_item_features(self, index, item_seq: list):
        """
        加载 item 侧的所有 token 特征，并缓存到 self.cached_item_features
        """
        if self.cached_item_features[index] is not None:
            return self.cached_item_features[index]

        feature_seq = dict()
        for item_id in item_seq:
            feature_dict = {} if self.selector is False else self.item_ut[item_id, self.selector]
            if self.item_ut.key_feature.name in feature_dict:
                del feature_dict[self.item_ut.key_feature.name]

            for feature, value in feature_dict.items():
                job = self.item_ut.meta.jobs[feature]
                if job.return_list:
                    value = value + [Env.null] * (job.max_len - len(value))
                if feature not in feature_seq:
                    feature_seq[feature] = []
                feature_seq[feature].append(value)

        self.cached_item_features[index] = feature_seq
        return feature_seq

    def __getitem__(self, index):
        """
        获取模型输入数据（字典形式）
        """
        sample = self.ut[index]
        user_id = sample[self.cm.user_col]
        item_seq = sample[self.cm.items_col]
        label_seq = sample[self.cm.actions_col]
        eval_seq = sample[self.cm.evals_col]
        feature_seq = self.load_item_features(index, item_seq)

        group_dict = {}
        gl = 0
        if self.history_config.activate:
            gl, gi, gs = self.history_cache[0][index], self.history_cache[1][index], self.history_cache[2][index]
            group_dict = {
                Syb.group_length: torch.tensor(gl, dtype=torch.long),
                Syb.group_index: torch.tensor(gi, dtype=torch.long),
                Syb.group_seq: torch.tensor(gs, dtype=torch.long),
            }

        seen_mask = None
        if self.global_validation:
            seen_mask = torch.zeros(self.item_size, dtype=torch.bool)
            if self.cm.histories_col in self.ut.meta.jobs:
                history = torch.tensor(self.ut[index][self.cm.histories_col], dtype=torch.long)
                seen_mask[history] = True
            seen_mask = seen_mask.unsqueeze(0).repeat(self.max_eval_nums, 1)

        input_ids, type_ids, action_labels, item_labels, feature_ids, seen_mask_count = self.build_inputs(
            gl, user_id, item_seq, label_seq, eval_seq, feature_seq, seen_mask
        )
        seg_ids = ([self.SEG_VOCAB[self.SEG_USER_ID]] +
                   [self.SEG_VOCAB[self.SEG_HISTORY]] * gl +
                   [self.SEG_VOCAB[self.SEG_ITEMS]] * (len(input_ids) - 1 - gl))

        seen_mask_dict = {}
        if self.global_validation:
            seen_mask_dict = {
                Syb.seen_mask: seen_mask,
                Syb.seen_mask_count: seen_mask_count,
            }

        for feature in feature_ids:
            feature_ids[feature] = self.pad(feature_ids[feature])

        seq_len = len(input_ids)
        attention_mask = [1] * seq_len

        data = {
            Syb.input_ids: self.pad(input_ids),
            Syb.type_ids: self.pad(type_ids),
            Syb.seg_ids: self.pad(seg_ids),
            Syb.feature_ids: feature_ids,
            Syb.action_labels: self.pad(action_labels),
            Syb.item_labels: self.pad(item_labels),
            Syb.attention_mask: self.pad(attention_mask),
            Syb.seq_len: seq_len,
            Syb.user_id: user_id,
            Syb.index: index,
            **seen_mask_dict,
            **group_dict,
        }
        return data

    def __len__(self):
        return len(self.ut)
