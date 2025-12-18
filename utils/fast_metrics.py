import math
from collections import OrderedDict
from multiprocessing import Pool
from typing import Dict, Union, List

import torch

from loaders.env import Env


class FastMetric:
    name: str
    minimize: bool

    def calculate(self, rank: int) -> Union[int, float]:
        pass

    def __call__(self, *args, **kwargs) -> Union[int, float]:
        return self.calculate(*args, **kwargs)

    def __str__(self):
        return self.name


class GAUC(FastMetric):
    name = 'GAUC'
    minimize = False

    def calculate(self, rank: int) -> Union[int, float]:
        return 1.0 - (rank - 1) / (Env.num_items - 1)


class MRR0(FastMetric):
    name = 'MRR0'
    minimize = False

    def calculate(self, rank: int) -> Union[int, float]:
        return 1.0 / rank


class MRR(MRR0):
    name = 'MRR'


class HitRatio(FastMetric):
    name = 'HitRatio'
    group = True
    minimize = False

    def __init__(self, n):
        self.n = n

    def calculate(self, rank: int) -> Union[int, float]:
        return 1.0 if rank <= self.n else 0.0

    def __str__(self):
        return f'{self.name}@{self.n}'


class Recall(HitRatio):
    name = 'Recall'


class NDCG(FastMetric):
    name = 'NDCG'
    minimize = False

    def __init__(self, n):
        self.n = n

    def calculate(self, rank: int) -> Union[int, float]:
        if rank > self.n:
            return 0.0
        return 1.0 / math.log2(rank + 1)

    def __str__(self):
        return f'{self.name}@{self.n}'


class MetricPool:
    metric_list = [Recall, HitRatio, GAUC, NDCG, MRR, MRR0]
    metric_dict = {m.name.upper(): m for m in metric_list}

    def __init__(self, metrics):
        self.metrics = metrics  # type: List[FastMetric]
        self.values = OrderedDict()  # type: Dict[str, Union[list, float]]

        for metric in self.metrics:
            self.values[str(metric)] = []

    @classmethod
    def parse(cls, metrics_config):
        metrics = []
        for m in metrics_config:
            at = m.find('@')
            argument = []
            if at > -1:
                m, argument = m[:at], [int(m[at+1:])]
            if m.upper() not in MetricPool.metric_dict:
                raise ValueError(f'Metric {m} not found')
            metric = MetricPool.metric_dict[m.upper()](*argument)
            metrics.append(metric)
        return cls(metrics)

    def calculate(self, ranks_series, num_workers=10):
        # 计算多个指标的值，ranks_series是多个样本的排名列表（List[List[int]])
        # 支持多进程加速计算
        if not self.metrics:
            return {}

        for metric in self.metrics:
            tasks = []
            pool = Pool(processes=num_workers)  # 创建进程池
            for ranks in ranks_series:
                # 为每个样本的rank调用指标计算函数，提交任务
                tasks.append(pool.apply_async(metric, args=(ranks,)))
            pool.close()
            pool.join()
            values = [t.get() for t in tasks]  # 获取所有计算结果
            # 转成tensor计算均值并保存
            self.values[str(metric)] = torch.tensor(values, dtype=torch.float).mean().item()
        return self.values

    def __call__(self, *args, **kwargs):
        # 允许直接调用实例计算指标，调用calculate方法
        return self.calculate(*args, **kwargs)

    @classmethod
    def is_minimize(cls, metric: str):
        # 判断某指标是否是最小化指标，支持传入FastMetric实例或字符串指标名
        if isinstance(metric, FastMetric):
            return metric.minimize
        assert isinstance(metric, str)
        metric = metric.split('@')[0]  # 去掉可能存在的@参数部分
        return cls.metric_dict[metric].minimize
