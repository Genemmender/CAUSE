from typing import List

from loaders.symbols import Symbols as Syb
from processors.backbone import Sequence
from processors.kuairand import KuairandProcessor


class KuairandBaseBuilder:
    """
    Kuairand 构建器的基类，定义构建数据集的基础逻辑，如切片、评估序列构造等。
    子类可实现不同的数据构建策略。
    """

    def __init__(self, processor: KuairandProcessor):
        self.processor = processor  # 上层 Processor，提供 daily_sequence、配置参数等

    def get_slice(self, mode):
        """
        根据模式返回最后一天的截取片段，子类需要实现。
        """
        raise NotImplementedError

    @staticmethod
    def _construct_eval_seq(last_num, last_value, max_length):
        """
        构造评估标签序列（长度为 max_length），
        最后 last_num 个值为 last_value，其余为 1 - last_value。
        """
        start_num = max_length - last_num
        start_value = 1 - last_value
        return [start_value] * start_num + [last_value] * last_num

    def construct_eval_seq(self, mode, val_num, test_num, max_length):
        """
        构造评估序列接口，子类需要根据 mode 实现。
        """
        raise NotImplementedError

    def slice_sequence(self, available_length, start_day_index):
        """
        将整个序列划分为多个子序列，供训练或评估使用。

        参数:
            available_length: 总长度
            start_day_index: 起始（训练开始）位置
        返回:
            histories: 历史行为的切片列表
            sequences: 当前序列的切片列表
        """
        histories = []
        sequences = []
        while available_length > start_day_index:
            right = available_length
            left = max(0, right - self.processor.max_length)
            histories.append(slice(0, left))
            sequences.append(slice(left, right))
            available_length = left
        return histories, sequences

    def construct_sequences(
            self,
            user,
            start_from: int,
            total_days: int,
            final_day_slice: slice = None,
    ) -> List[Sequence]:
        """
        构建用户的多个序列（含历史和当前行为），从某天开始往后 total_days 天。

        参数:
            user: 用户 ID
            start_from: 开始的日期索引
            total_days: 总共多少天
            final_day_slice: 对最后一天的切片
        返回:
            多个带有 history 的 Sequence 序列
        """
        sequences = []
        sequence = Sequence.empty()
        start_day_index = 0
        for date_index in range(start_from + total_days):
            date = self.processor.date_map[date_index]

            if date_index == start_from:
                start_day_index = len(sequence)

            if date not in self.processor.daily_sequence[user]:
                continue

            day_sequence = self.processor.daily_sequence[user][date]

            if date_index + 1 == start_from + total_days and final_day_slice is not None:
                day_sequence = day_sequence[final_day_slice]

            sequence.extend(day_sequence)

        history_slices, sequence_slices = self.slice_sequence(len(sequence), start_day_index)
        for hs, ss in zip(history_slices, sequence_slices):
            s = sequence[ss]
            s.histories = self.processor.filter.filter_history(sequence[hs])
            sequences.append(s)
        return sequences

    def build_train_data(self, item_vocab, user_vocab):
        """
        构建训练数据，调用 processor 内部的 construct_dataset 和 tokenize 接口。
        """
        train_data = self.processor.construct_dataset(
            start_from=self.processor.train_from,
            total_days=self.processor.test_at - self.processor.train_from,
            mode=Syb.train,
        )
        self.processor.tokenize_interactions(train_data, item_vocab, user_vocab, self.processor.train_dir)
        return train_data

    def build_val_data(self, item_vocab, user_vocab):
        """
        构建验证集，基类未实现，由子类实现。
        """
        raise NotImplementedError

    def build_test_data(self, item_vocab, user_vocab):
        """
        构建测试数据集（从 test_at 开始取一天数据）
        """
        test_data = self.processor.construct_dataset(
            start_from=self.processor.test_at,
            total_days=1,
            mode=Syb.test,
        )
        self.processor.tokenize_interactions(test_data, item_vocab, user_vocab, self.processor.test_dir)
        return test_data


class KuairandV1Builder(KuairandBaseBuilder):
    """
    Kuairand 构建器 V1 版本：
    - 用户序列互斥（训练、验证、测试不重叠）
    - 验证集在完整训练序列的最后
    """

    def get_slice(self, mode):
        if mode is Syb.test:
            return self.processor.get_slice(self.processor.test_num)
        return None

    def construct_eval_seq(self, mode, val_num, test_num, max_length):
        if mode is Syb.train:
            return self._construct_eval_seq(val_num, 0, max_length)
        if mode is Syb.val:
            return self._construct_eval_seq(val_num, 1, max_length)
        return self._construct_eval_seq(test_num, 1, max_length)

    def build_val_data(self, item_vocab, user_vocab):
        val_data = self.processor.construct_dataset(
            start_from=self.processor.train_from,
            total_days=self.processor.test_at - self.processor.train_from,
            mode=Syb.val,
        )
        self.processor.tokenize_interactions(val_data, item_vocab, user_vocab, self.processor.val_dir)
        return val_data


class KuairandV2Builder(KuairandBaseBuilder):
    """
    Kuairand 构建器 V2 版本：
    - 用户序列互斥
    - 验证集和测试集都来自最后一天，通过截取划分
    """

    def get_slice(self, mode):
        if mode is Syb.test:
            return self.processor.get_slice(self.processor.test_num + self.processor.val_num)
        if mode is Syb.val:
            return self.processor.get_slice(self.processor.val_num)
        return None

    def build_val_data(self, item_vocab, user_vocab):
        val_data = self.processor.construct_dataset(
            start_from=self.processor.test_at,
            total_days=1,
            mode=Syb.val,
        )
        self.processor.tokenize_interactions(val_data, item_vocab, user_vocab, self.processor.val_dir)
        return val_data

    def construct_eval_seq(self, mode, val_num, test_num, max_length):
        if mode is Syb.train:
            return self._construct_eval_seq(0, 0, max_length)
        if mode is Syb.val:
            return self._construct_eval_seq(val_num, 1, max_length)
        return self._construct_eval_seq(test_num, 1, max_length)


class KuairandV3Builder(KuairandV2Builder):
    """
    Kuairand 构建器 V3 版本：
    - 序列不互斥，用户每天都可以形成多个序列
    - 验证集和测试集仍然位于最后一天
    """

    def construct_sequences(
            self,
            user,
            start_from: int,
            total_days: int,
            final_day_slice: slice = None,
    ) -> List[Sequence]:
        sequences = []
        sequence = Sequence.empty()
        start_day_index = None
        user_sequence_last_indices = []

        for date_index in range(start_from + total_days):
            date = self.processor.date_map[date_index]

            if date_index == start_from:
                start_day_index = len(sequence)

            if date not in self.processor.daily_sequence[user]:
                continue

            day_sequence = self.processor.daily_sequence[user][date]

            if date_index + 1 == start_from + total_days and final_day_slice is not None:
                day_sequence = day_sequence[final_day_slice]

            sequence.extend(day_sequence)

            if start_day_index is not None:
                user_sequence_last_indices.append(len(sequence))

        # 反向构造所有序列，不断向前滑动
        history_slices = []
        sequence_slices = []
        for last_index in user_sequence_last_indices[::-1]:
            right = last_index
            left = max(0, right - self.processor.max_length)
            history_slices.append(slice(0, left))
            sequence_slices.append(slice(left, right))
            if left == 0:
                break

        for hs, ss in zip(history_slices, sequence_slices):
            s = sequence[ss]
            s.histories = self.processor.filter.filter_history(sequence[hs])
            sequences.append(s)
        return sequences
