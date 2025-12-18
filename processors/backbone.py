import os.path

import pandas as pd
from oba import Obj, NotFound
from pigmento import pnt

from loaders.column_map import ColumnMap
from utils.alias_config import AliasConfig


class Sequence:
    """
    表示一个用户的行为序列，包括 item 列表、action 列表 和 时间列表。
    """

    def __init__(self, items: list, actions: list, times: list):
        self.items = items
        self.actions = actions
        self.times = times
        self.histories = None

    @classmethod
    def empty(cls):
        """返回一个空的 Sequence 实例"""
        return cls([], [], [])

    def copy(self):
        """深拷贝当前序列"""
        return Sequence(self.items.copy(), self.actions.copy(), self.times.copy())

    def append(self, item, action, time):
        """添加一个新的交互记录"""
        self.items.append(item)
        self.actions.append(action)
        self.times.append(time)

    def extend(self, other: 'Sequence'):
        """将另一个序列拼接到当前序列"""
        self.items += other.items
        self.actions += other.actions
        self.times += other.times
        return self

    def __len__(self):
        return len(self.items)

    def __getitem__(self, slice_):
        assert isinstance(slice_, slice)
        return Sequence(self.items[slice_], self.actions[slice_], self.times[slice_])

    def __str__(self):
        return f'Sequence(len={len(self)}, history={self.histories and len(self.histories)})'

    def __repr__(self):
        return str(self)

    def sort(self):
        """根据时间进行排序"""
        zipped = list(zip(self.items, self.actions, self.times))
        zipped.sort(key=lambda x: x[2])
        self.items, self.actions, self.times = zip(*zipped)


class BackboneProcessor(AliasConfig):
    """
    序列数据处理器基类，提供配置管理、目录组织、签名注册与基础数据预处理能力。
    """

    # 必需字段名称（需要子类赋值）
    ITEM_COL: str
    USER_COL: str
    TIME_COL: str

    ITEM_COUNT: str = 'item_count'
    USER_COUNT: str = 'user_count'

    # 特征列命名（处理后）
    ITEM_FEAT = 'item'
    USER_FEAT = 'user'
    ITEMS_FEAT = 'items'
    ACTIONS_FEAT = 'actions'
    EVALS_FEAT = 'evals'
    TIMES_FEAT = 'times'
    HISTORY_FEAT = 'histories'

    SIGNATURE_KEYS: list  # 用于注册签名的关键字段名列表

    BASE_DATA_DIR = 'data'  # 保存处理数据的根目录

    signature: str
    max_length: int
    min_length: int

    def __init__(
            self,
            name: str,
            config,
            data_dir,
            force: bool = False,
            **kwargs
    ):
        super().__init__(save_dir=os.path.join(self.BASE_DATA_DIR, self.get_classname()))
        self.name = name
        self.data_dir = data_dir
        self.config = self.parse_config(config)
        self.force = force

    @classmethod
    def build_column_map(cls):
        """返回当前处理器标准使用的 ColumnMap 映射配置"""
        return ColumnMap(
            user_col=cls.USER_FEAT,
            items_col=cls.ITEMS_FEAT,
            actions_col=cls.ACTIONS_FEAT,
            evals_col=cls.EVALS_FEAT,
            histories_col=cls.HISTORY_FEAT,
        )

    def parse_config(self, config):
        """解析配置，如果为 None，则根据名称从别名或签名加载配置文件"""
        if config is not None:
            return config

        if self.name.startswith('#'):
            pnt(f'load config by alias: {self.name}')
            self.name = self.name[1:]
            sign = self.aliases[self.name]
        elif self.name.startswith('@'):
            pnt(f'load config by sign: {self.name}')
            sign = self.name[1:]
            self.name = None
        elif self.name in self.aliases:
            pnt(f'load config by alias: #{self.name}')
            sign = self.aliases[self.name]
        elif self.name in self.signs:
            pnt(f'load config by sign: @{self.name}')
            sign = self.name
            self.name = None
        else:
            raise ValueError(f'cannot match config by {self.name}')

        config = self.load_config(sign)
        return Obj(config)

    # 各种阶段性处理结果的保存路径
    @property
    def item_dir(self) -> str:
        return os.path.join(self.save_dir, self.signature, 'items')

    @property
    def train_dir(self) -> str:
        return os.path.join(self.save_dir, self.signature, 'train')

    @property
    def val_dir(self) -> str:
        return os.path.join(self.save_dir, self.signature, 'val')

    @property
    def test_dir(self) -> str:
        return os.path.join(self.save_dir, self.signature, 'test')

    def process(self):
        """具体数据处理逻辑，由子类实现"""
        raise NotImplementedError

    def analyse(self):
        """数据分析逻辑，由子类实现"""
        raise NotImplementedError

    @classmethod
    def get_classname(cls):
        """返回数据集名称（小写）"""
        return cls.__name__.replace('Processor', '').lower()

    def get_sign_config(self):
        """获取构造签名的配置内容（字段由 SIGNATURE_KEYS 决定）"""
        config = dict()
        for key in self.SIGNATURE_KEYS:
            value = getattr(self, key)
            if isinstance(value, NotFound):
                continue
            config[key] = value
        return config

    def get_signature(self):
        """注册配置并生成签名，同时为别名创建软链接"""
        sign_config = self.get_sign_config()
        pnt('Dataset:', self.get_classname())
        pnt('Config:', sign_config)
        pnt('Force:', self.force)
        signature = self.register(sign_config, alias=self.name, force=self.force)
        if self.name is not None:
            os.system(f'ln -sf {signature} {os.path.join(self.save_dir, self.name)}')
        return signature

    @staticmethod
    def get_slice(truncate):
        """根据 truncate 返回一个 slice 对象（正数保留前 N，负数保留后 N）"""
        if truncate is None:
            truncate = 0
        if truncate > 0:
            return slice(0, truncate)
        if truncate < 0:
            return slice(truncate, None)
        return slice(None)

    def _core_filter(self, interactions: pd.DataFrame, n_core):
        """
        执行核心过滤操作：移除低频用户和低频物品（小于 n_core 的）。
        """
        last_size = len(interactions)

        while True:
            # 删除旧的统计列，防止 join 出现重复列
            interactions = interactions.drop(columns=[self.ITEM_COUNT, self.USER_COUNT], errors="ignore")

            # 统计 item 和 user 的交互次数
            item_id_count = interactions.groupby(self.ITEM_COL).size().reset_index(name=self.ITEM_COUNT)
            user_id_count = interactions.groupby(self.USER_COL).size().reset_index(name=self.USER_COUNT)

            # 将统计列 join 回原始数据中
            interactions = interactions.join(item_id_count.set_index(self.ITEM_COL), on=self.ITEM_COL)
            interactions = interactions.join(user_id_count.set_index(self.USER_COL), on=self.USER_COL)

            # 执行过滤
            interactions = interactions[interactions[self.ITEM_COUNT] >= n_core]
            interactions = interactions[interactions[self.USER_COUNT] >= n_core]

            current_size = len(interactions)

            if current_size == last_size:
                break  # 若没有变化则停止过滤
            pnt(f'filter interactions from {last_size} to {current_size}')
            last_size = current_size

        # 删除辅助列后返回
        return interactions.drop(columns=[self.ITEM_COUNT, self.USER_COUNT], errors="ignore")
