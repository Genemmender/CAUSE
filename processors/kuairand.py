import hashlib
import os
import collections
from collections import defaultdict
from typing import Optional

import pandas as pd
from pigmento import pnt
from tqdm import tqdm
from unitok import Vocab, UniTok, EntityTokenizer, EntitiesTokenizer, DigitsTokenizer, Symbol, SplitTokenizer

from processors.backbone import BackboneProcessor, Sequence
from utils.plot import Plot
from utils import io


class KuairandProcessor(BackboneProcessor):
    # 原始列名（来自数据文件）
    ITEM_COL = 'video_id'
    USER_COL = 'user_id'
    TIME_COL = 'time_ms'
    ACTION_COL = 'action'

    # 用于中间过滤的列名
    ITEM_COUNT = 'item_count'
    USER_COUNT = 'user_count'

    # 行为交互列
    DATE_COL = 'date'
    IS_CLICK = 'is_click'
    IS_LIKE = 'is_like'
    IS_FOLLOW = 'is_follow'
    IS_COMMENT = 'is_comment'
    IS_FORWARD = 'is_forward'
    IS_HATE = 'is_hate'
    LONG_VIEW = 'long_view'

    # 行为列的类型校验
    INTERACTION_COLUMN_TYPE_ASSERTIONS = {
        DATE_COL: str,
        IS_CLICK: int,
        IS_LIKE: int,
        IS_FOLLOW: int,
        IS_COMMENT: int,
        IS_FORWARD: int,
        IS_HATE: int,
        LONG_VIEW: int
    }

    # item 特征列
    MUSIC_TYPE = 'music_type'
    TAG = 'tag'

    # item 列的类型校验
    ITEM_COLUMN_TYPE_ASSERTIONS = {
        MUSIC_TYPE: int,
        TAG: str,
    }

    # 用于唯一确定配置签名的关键字段
    SIGNATURE_KEYS = [
        'train_from', 'test_at', 'max_length', 'min_length',
        'test_num', 'val_num', 'n_core', 'filter_key', 'builder_key',
        'group_from', 'group_num'
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.daily_sequence: Optional[pd.DataFrame] = None
        self.date_map: Optional[Vocab] = None

        # 配置参数
        self.train_from = self.config.train_from
        self.test_at = self.config.test_at
        self.max_length = self.config.max_length
        self.min_length = self.config.min_length
        self.test_num = self.config.test_num
        self.test_slice = self.get_slice(self.config.test_num)
        self.val_num = self.config.val_num
        self.n_core = self.config.n_core
        self.filter_key = self.config.filter_key
        self.builder_key = self.config.builder_key
        self.group_from = self.config.group_from
        self.group_num = self.config.group_num

        # 生成签名、加载 reader / filter / builder
        self.signature: str = self.get_signature()
        self.reader = self.load_reader()
        self.filter = self.load_filter()
        self.builder = self.load_builder()

    def load_builder(self):
        import processors.kuairand_builders as builders
        if self.builder_key.lower() == 'v1':
            return builders.KuairandV1Builder(self)
        if self.builder_key.lower() == 'v2':
            return builders.KuairandV2Builder(self)
        if self.builder_key.lower() == 'v3':
            return builders.KuairandV3Builder(self)
        raise ValueError(f'Unknown dataset builder: {self.builder_key}')

    def load_reader(self):
        import processors.kuairand_readers as readers
        return readers.Kuairand1KReader(self)

    def load_filter(self):
        import processors.kuairand_filters as filters
        if self.filter_key.lower() == 'ma':
            return filters.KuairandMAFilter(self)
        if self.filter_key.lower() == 'sort':
            return filters.KuairandSortFilter(self)
        raise ValueError(f'Unknown dataset filter: {self.filter_key}')

    def _load_interactions(self, path: str):
        """底层方法，从文件路径中读取交互日志"""
        return pd.read_csv(
            filepath_or_buffer=path,
            sep=',',
            names=[
                self.USER_COL, self.ITEM_COL, self.DATE_COL, 'hourmin',
                self.TIME_COL, self.IS_CLICK, self.IS_LIKE, self.IS_FOLLOW, self.IS_COMMENT,
                self.IS_FORWARD, self.IS_HATE, self.LONG_VIEW, 'play_time_ms', 'duraiton_ms',
                'profile_stay_time', 'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab',
            ],
            usecols=[
                self.USER_COL, self.ITEM_COL, self.TIME_COL, self.DATE_COL, self.IS_CLICK, self.IS_LIKE, self.IS_FOLLOW,
                self.IS_COMMENT, self.IS_FORWARD, self.IS_HATE, self.LONG_VIEW,
            ],
        )

    def load_interactions(self) -> pd.DataFrame:
        interaction_cache_path = os.path.join(self.data_dir, 'interactions.pkl')
        if os.path.exists(interaction_cache_path):
            return io.pkl_load(interaction_cache_path)

        interactions = self.reader.read_interactions()
        pnt(f'Scenario filter ...')
        interactions = self.filter.scenario_filter(interactions)
        pnt(f'Build action column ...')
        interactions = self.filter.build_action(interactions)
        assert self.ACTION_COL in interactions.columns

        pnt(f'N-Core filter ...')
        interactions = self._core_filter(interactions, self.n_core)
        interactions = interactions.reset_index(drop=True)

        print(f'Total interactions before filter: {len(interactions)}')
        interactions = interactions.drop_duplicates(subset=[self.USER_COL, self.ITEM_COL], keep='first')
        interactions = interactions.reset_index(drop=True)
        print(f'Total interactions after filter: {len(interactions)}')

        io.pkl_save(interactions, interaction_cache_path)
        return interactions

    def load_items(self, interactions: pd.DataFrame) -> pd.DataFrame:
        items = self.reader.read_items()
        used_items = interactions[self.ITEM_COL].unique()
        items = items[items[self.ITEM_COL].isin(used_items)]
        items = items.reset_index(drop=True)
        items[self.MUSIC_TYPE] = items[self.MUSIC_TYPE].fillna(-1).apply(lambda x: int(float(x)))
        items[self.TAG] = items[self.TAG].fillna('')
        return items

    def build_daily_sequence(self, interactions: pd.DataFrame):
        """将交互数据构建为每天一个序列"""
        min_date = interactions[self.DATE_COL].min()
        max_date = interactions[self.DATE_COL].max()
        date_range = pd.date_range(start=min_date, end=max_date).strftime('%Y%m%d')
        dates = list(date_range)
        date_map = Vocab(name='date')
        for date in dates:
            date_map.append(date)

        daily_sequence = defaultdict(lambda: defaultdict(Sequence.empty))
        for row in tqdm(interactions.itertuples(index=False), total=len(interactions)):
            user = str(getattr(row, self.USER_COL))
            date = str(getattr(row, self.DATE_COL))
            item = getattr(row, self.ITEM_COL)
            time = getattr(row, self.TIME_COL)
            action = getattr(row, self.ACTION_COL)
            daily_sequence[user][date].append(item, action, time)

        pnt('sorting daily sequence')
        for user in tqdm(daily_sequence):
            for date in daily_sequence[user]:
                daily_sequence[user][date].sort()

        return date_map, daily_sequence

    def construct_dataset(self, start_from: int, total_days: int, mode: Symbol) -> pd.DataFrame:
        """从 daily_sequence 构造某个模式（train/val/test）的数据集"""
        assert self.date_map.size > start_from >= 0
        assert self.date_map.size > start_from + total_days >= 0
        assert total_days > 0

        dataset = []
        for user in self.daily_sequence:
            sequences = self.builder.construct_sequences(
                user=user,
                start_from=start_from,
                total_days=total_days,
                final_day_slice=self.builder.get_slice(mode),
            )
            for sequence in sequences:
                dataset.append({
                    self.USER_COL: user,
                    self.ACTIONS_FEAT: sequence.actions,
                    self.ITEMS_FEAT: sequence.items,
                    self.TIMES_FEAT: sequence.times,
                    self.EVALS_FEAT: self.builder.construct_eval_seq(mode, self.val_num, self.test_num, len(sequence)),
                    self.HISTORY_FEAT: sequence.histories,
                })

        return pd.DataFrame(dataset)

    def tokenize_interactions(self, interactions: pd.DataFrame, item_vocab: Vocab, user_vocab: Vocab, dir_name):
        """对交互序列进行token化并存储"""
        with UniTok() as ut:
            ut.add_index_feature()
            ut.add_feature(tokenizer=EntityTokenizer(vocab=user_vocab), column=self.USER_COL, name=self.USER_FEAT)
            ut.add_feature(tokenizer=EntitiesTokenizer(vocab=item_vocab), column=self.ITEMS_FEAT, name=self.ITEMS_FEAT)
            ut.add_feature(tokenizer=DigitsTokenizer(vocab='action'), column=self.ACTIONS_FEAT, name=self.ACTIONS_FEAT)
            ut.add_feature(tokenizer=DigitsTokenizer(vocab='eval'), column=self.EVALS_FEAT, name=self.EVALS_FEAT)
            ut.add_feature(tokenizer=EntitiesTokenizer(vocab=item_vocab), column=self.HISTORY_FEAT, name=self.HISTORY_FEAT)
            ut.tokenize(interactions).save(dir_name)

    def tokenize_items(self, items, item_vocab):
        """对 item 表进行token化"""
        with UniTok() as ut:
            ut.add_feature(tokenizer=EntityTokenizer(vocab=item_vocab), column=self.ITEM_COL, name=self.ITEM_FEAT, key=True)
            ut.add_feature(tokenizer=EntityTokenizer(vocab=self.MUSIC_TYPE))
            ut.add_feature(tokenizer=SplitTokenizer(vocab=self.TAG, sep=','))
            ut.tokenize(items).save(self.item_dir)

    @staticmethod
    def _get_checker(dtype):
        """生成类型检查函数"""
        def checker(value):
            if not isinstance(value, dtype):
                raise TypeError(f'{value} is not {dtype}')
        return checker

    def _type_checking(self, interactions, items):
        """对输入数据执行列类型检查"""
        pnt('start type checking for interactions')
        for col, dtype in self.INTERACTION_COLUMN_TYPE_ASSERTIONS.items():
            pnt(f'  {col} {dtype} checking')
            checker = self._get_checker(dtype)
            interactions[col].apply(checker)
            pnt(f'  {col} {dtype} checked')
        pnt('end type checking for interactions')

        pnt('start type checking for items')
        for col, dtype in self.ITEM_COLUMN_TYPE_ASSERTIONS.items():
            pnt(f'  {col} {dtype} checking')
            checker = self._get_checker(dtype)
            items[col].apply(checker)
            pnt(f'  {col} {dtype} checked')
        pnt('end type checking for items')

    @staticmethod
    def hash_to_index(s, k):
        h = hashlib.md5(s.encode('utf-8')).hexdigest()  # 生成 32 位十六进制哈希
        h_int = int(h, 16)  # 转成整数
        return h_int % k

    def filter_by_grouping(self):
        """
        用户桶训练设置
        """
        if self.group_from is None or self.group_num is None:
            return

        user_group = {
            user: self.hash_to_index(user, self.group_num)
            for user in self.daily_sequence
        }

        date_group = dict()

        for date in self.date_map:
            date_index = self.date_map[date]
            if date_index < self.group_from:
                continue
            group_index = (date_index - self.group_from) % self.group_num
            date_group[date] = group_index

        for user, date_dict in self.daily_sequence.items():
            u_group = user_group[user]
            to_delete = [
                d for d in list(date_dict.keys())
                if d in date_group and date_group[d] != u_group
            ]
            for d in to_delete:
                del date_dict[d]

    def process(self):
        """完整处理流程"""
        interactions = self.load_interactions()
        items = self.load_items(interactions=interactions)
        self._type_checking(interactions=interactions, items=items)

        item_vocab = Vocab(name='item')
        user_vocab = Vocab(name='user')

        self.tokenize_items(items, item_vocab=item_vocab)

        values = interactions[self.DATE_COL].tolist()
        groups = collections.Counter(values)
        Plot.draw_groups(groups)

        self.date_map, self.daily_sequence = self.build_daily_sequence(interactions)
        for user in self.daily_sequence:
            user_vocab.append(user)

        self.filter_by_grouping()

        item_vocab.deny_edit()
        user_vocab.deny_edit()

        self.builder.build_train_data(item_vocab=item_vocab, user_vocab=user_vocab)
        self.builder.build_val_data(item_vocab=item_vocab, user_vocab=user_vocab)
        self.builder.build_test_data(item_vocab=item_vocab, user_vocab=user_vocab)

    def analyse(self):
        """输出数据集分析统计信息"""
        interactions = self.load_interactions()
        items = self.load_items(interactions=interactions)
        self._type_checking(interactions=interactions, items=items)

        print(f'Dataset Statistics for {self.get_classname()}')
        print(f'Interactions: {len(interactions)}')
        print(f'Items: {len(items)}\n')

        for column in list(self.INTERACTION_COLUMN_TYPE_ASSERTIONS.keys()) + [self.ACTION_COL]:
            print(f'  Column: {column}')
            values = interactions[column].tolist()
            counts = collections.Counter(values)
            keys = sorted(counts.keys())
            for key in keys:
                count = counts[key]
                print(f'    {key}: {count} ({count / len(interactions)*100:.2f}%)')
            print()


