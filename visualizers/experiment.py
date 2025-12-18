import os.path
import re
from typing import List, Type

from oba import Obj
from unitok import UniTok

from loaders.class_hub import ClassHub
from loaders.symbols import Symbols as Syb
from processor import MetaProcessor
from processors.backbone import BackboneProcessor
from utils import io
from utils.path_hub import BasePathHub


class InteractionStatistics:
    def __init__(self, path, processor: BackboneProcessor):
        self.ut = UniTok.load(path)
        self.processor = processor

        self.size = len(self.ut)

        history_lengths = []
        sequence_lengths = []
        for sequence in self.ut.data[self.processor.ITEMS_FEAT]:
            sequence_lengths.append(len(sequence))

        self.has_history = 0
        for history in self.ut.data[self.processor.HISTORY_FEAT]:
            history_lengths.append(len(history))
            if history:
                self.has_history += 1

        self.history_length_mean = sum(history_lengths) / len(history_lengths)
        self.history_length_max = max(history_lengths)
        self.sequence_length_mean = sum(sequence_lengths) / len(sequence_lengths)
        self.sequence_length_max = max(sequence_lengths)

        self.item_size = self.ut.meta.jobs[self.processor.ITEMS_FEAT].tokenizer.vocab.size
        self.user_size = self.ut.meta.jobs[self.processor.USER_FEAT].tokenizer.vocab.size
        self.action_size = self.ut.meta.jobs[self.processor.ACTIONS_FEAT].tokenizer.vocab.size

        self.sequence_per_user = self.size / self.user_size
        self.sequence_per_item = sum(sequence_lengths) / self.item_size
        self.history_per_item = sum(history_lengths) / self.item_size


class Statistics:
    _CACHED_DATASET = dict()

    def __init__(self, data_name):
        self.data_name = data_name
        self.processor = self.get_processor()

        self.train = InteractionStatistics(self.processor.train_dir, self.processor)
        self.val = InteractionStatistics(self.processor.val_dir, self.processor)
        self.test = InteractionStatistics(self.processor.test_dir, self.processor)

    @classmethod
    def load(cls, data_name):
        if data_name not in cls._CACHED_DATASET:
            cls._CACHED_DATASET[data_name] = cls(data_name)

        return cls._CACHED_DATASET[data_name]

    def get_processor(self) -> BackboneProcessor:
        assert '.' in self.data_name, f'data should be in <dataset>.<name> format, but got {self.data_name}'
        dataset, alias = self.data_name.split('.', maxsplit=1)
        data_dir = MetaProcessor.get_data_dir(dataset)

        processors = ClassHub.processors()
        processor_class: Type[BackboneProcessor] = processors[dataset]
        return processor_class(name=alias, config=None, data_dir=data_dir)


class EpochTime:
    def __init__(self):
        self.epoch_times = [0]

    def push(self, epoch: int, time: int):
        if epoch < len(self.epoch_times):
            return

        epoch_interval = epoch - len(self.epoch_times) + 1
        time_interval = time - self.epoch_times[-1]

        time_unit = time_interval // epoch_interval

        for i in range(epoch_interval):
            self.epoch_times.append(self.epoch_times[-1] + time_unit)


class Experiment(BasePathHub):
    """
    表示一个具体的实验版本，继承路径管理类 BasePathHub。
    提供了读取配置、训练进度、当前状态的方法。
    """

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name  # 人类可读的实验别名

        # 从 config 文件中提取：数据集全名、总 epoch 数、early stop 的 patience、输入类型
        self.data_full, self.epoch, self.patience, self.input_type = self.get_exp_config()

        # 获取训练进度：epoch 数组、interval 数组（如 save 或 val 记录）
        self.epoches, self.intervals = self.get_progress()

        # 当前 epoch（可能来自 epoch 或 interval）
        self.current_epoch = self.get_current_epoch()

        # 最终评估指标
        # self.test_results = self.get_test_results()

    def get_dataset_info(self):
        stats = Statistics.load(self.data_full)  # type: Statistics
        info = Obj()
        info.name = self.data_full
        info.item_size = stats.train.item_size
        info.user_size = stats.train.user_size
        info.action_size = stats.train.action_size

        for mode, stat in zip(['train', 'val', 'test'], [stats.train, stats.val, stats.test]):  # type: str, InteractionStatistics
            info[mode + ':size'] = stat.size
            info[mode + ':history_length_mean'] = f'{stat.history_length_mean:.4f}'
            info[mode + ':history_length_max'] = f'{stat.history_length_max:.4f}'
            info[mode + ':sequence_length_mean'] = f'{stat.sequence_length_mean:.4f}'
            info[mode + ':sequence_length_max'] = f'{stat.sequence_length_max:.4f}'
            info[mode + ':sequence_per_user'] = f'{stat.sequence_per_user:.4f}'
            info[mode + ':sequence_per_item'] = f'{stat.sequence_per_item:.4f}'
            info[mode + ':history_per_item'] = f'{stat.history_per_item:.4f}'
            info[mode + ':has_history'] = stat.has_history

        return info()

    def get_test_results(self, metrics: List[str]):
        if not os.path.exists(self.result_path):
            return None

        lines = list(filter(lambda x: x.strip(), io.file_load(self.result_path).split('\n')))
        lines = list(map(lambda x: x.split(','), lines))
        return {line[0]: line[1] for line in lines if any(line[0].lower().startswith(m.lower()) for m in metrics)}

    def get_exp_config(self):
        """
        从 cfg.json 中读取核心配置字段。
        """
        config = Obj(io.json_load(self.cfg_path))
        data_full = config.data
        epoch = config.exp.save.epoch
        patience = config.exp.save.patience
        input_type = config.model.config.seq_type
        return data_full, epoch, patience, input_type

    def get_progress(self):
        """
        从 progress.jsonl 中读取所有 epoch 和 interval 类型的日志记录。
        区分 `epoch`（训练结束）和 `interval`（验证、保存等阶段）类型。
        """
        progress: List[dict] = io.jsonl_load(self.progress_path)

        epoches = []
        intervals = []
        index = 0

        for p in progress:
            op = Obj(p)
            epoch, type_ = op.epoch, op.type or Syb.epoch.name
            assert epoch in [index, index + 1], f'Exp ({self.name}): epoch {epoch} is not continuous, expected {index} or {index + 1}'
            index = epoch

            if type_ == Syb.epoch.name:
                epoches.append(p)
            else:
                intervals.append(p)

        return epoches, intervals

    def get_epoch_times(self):
        full_log = io.file_load(self.log_path)
        lines = full_log.splitlines(keepends=False)
        # [00:01:56] |Trainer| [epoch 1]
        template = r'\[(\d\d):(\d\d):(\d\d)\] \|Trainer\| \[epoch (\d+)\]'
        et = EpochTime()

        for line in lines:
            match = re.search(template, line)
            if match:
                hours, minutes, seconds, epoch = match.groups()
                et.push(int(epoch), int(hours) * 3600 + int(minutes) * 60 + int(seconds))
        return et.epoch_times

    def get_current_epoch(self):
        """
        获取当前 epoch（最大训练或验证记录的 epoch）。
        """
        epoch = 0
        if len(self.epoches) > 0:
            epoch = self.epoches[-1]['epoch']
        if len(self.intervals) > 1:
            epoch = max(epoch, self.intervals[-1]['epoch'])
        return epoch

    def get_loss_curve_by_epoch(self):
        points = []
        for interval in self.intervals:
            interval = Obj(interval)
            epoch, loss = interval.epoch, interval.loss
            points.append((epoch, loss))
        return points

    def get_loss_curve_by_time(self):
        points = []
        epoch_times = self.get_epoch_times()
        for interval in self.intervals:
            interval = Obj(interval)
            epoch, loss = interval.epoch, interval.loss

            if epoch >= len(epoch_times):
                break

            time = epoch_times[epoch]
            points.append((time, loss))
        return points

    def get_metric(self, d: dict):
        known_keys = ['data', 'epoch', 'type']
        for key in d.keys():
            if key not in known_keys:
                return key
        raise ValueError(f'No metric found in the experiment {self.name}')

    def get_evaluation_curve_by_epoch(self):
        metric = self.get_metric(self.epoches[0])

        points = dict()
        for line in self.epoches:
            line = Obj(line)
            if line.data not in points:
                points[line.data] = []

            if metric not in line:
                raise ValueError(f'Metric {metric} not found in the line: {line()}')
            points[line.data].append((line.epoch, line[metric]))

        return metric, points

    def get_evaluation_curve_by_time(self):
        metric = self.get_metric(self.epoches[0])

        points = dict()
        epoch_times = self.get_epoch_times()
        for line in self.epoches:
            line = Obj(line)
            if line.data not in points:
                points[line.data] = []

            if line.epoch >= len(epoch_times):
                break

            if metric not in line:
                raise ValueError(f'Metric {metric} not found in the line: {line()}')
            points[line.data].append((epoch_times[line.epoch], line[metric]))

        return metric, points

    def json(self):
        """
        转换为字典，便于保存到 meta 文件。
        """
        return dict(
            data=self.data_name,
            model=self.model_name,
            signature=self.signature,
        )
