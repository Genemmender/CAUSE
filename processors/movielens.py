import os
from typing import Optional

import pandas as pd
from tqdm import tqdm
from unitok import Vocab, UniTok, EntityTokenizer, EntitiesTokenizer, DigitsTokenizer, SplitTokenizer

from processors.backbone import BackboneProcessor


class MovieLensProcessor(BackboneProcessor):
    # PRIMARY ORIGINAL COLUMN NAMES
    ITEM_COL = 'movie_id'
    USER_COL = 'user_id'
    TIME_COL = 'time_ms'
    ACTION_COL = 'action'

    # TEMPORARY FILTER COLUMNS
    ITEM_COUNT = 'item_count'
    USER_COUNT = 'user_count'

    # USED INTERACTION COLUMN NAMES
    DATE_COL = 'date'
    IS_CLICK = 'is_click'
    IS_LIKE = 'is_like'
    IS_FOLLOW = 'is_follow'
    IS_COMMENT = 'is_comment'
    IS_FORWARD = 'is_forward'
    IS_HATE = 'is_hate'
    LONG_VIEW = 'long_view'

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

    SIGNATURE_KEYS = ['min_length', 'max_length']

    # USED ITEM COLUMN NAMES
    GENRES = 'genres'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.daily_sequence: Optional[pd.DataFrame] = None
        self.date_map: Optional[Vocab] = None

        self.test_num = 1
        self.min_length = self.config.min_length
        self.max_length = self.config.max_length

        self.signature: str = self.get_signature()

    @property
    def val_dir(self) -> str:
        return os.path.join(self.save_dir, self.signature, 'test')

    def read_items(self):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'movies.csv'),
            sep=',',
            names=[self.ITEM_COL, 'title', self.GENRES],
            usecols=[self.ITEM_COL, self.GENRES],
            header=0,
        )

    def load_items(self, sequences: pd.DataFrame) -> pd.DataFrame:
        used_items = set()
        sequences[self.ITEMS_FEAT].apply(lambda x: used_items.update(set(x)))
        sequences[self.HISTORY_FEAT].apply(lambda x: used_items.update(set(x)))

        items = self.read_items()
        items[self.ITEM_COL] = items[self.ITEM_COL].astype(str)
        items = items[items[self.ITEM_COL].isin(used_items)]
        items[self.GENRES] = items[self.GENRES].fillna('')
        return items

    def read_sequences(self) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'sasrec_format.csv'),
            sep=',',
            names=['index', self.USER_COL, self.ITEMS_FEAT, self.ACTIONS_FEAT, self.TIMES_FEAT],
            usecols=[self.USER_COL, self.ITEMS_FEAT, self.ACTIONS_FEAT],
            header=0
        )

    def scenario_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        def row_filter(row):
            filtered_items = [item for item, action in zip(row[self.ITEMS_FEAT], row[self.ACTIONS_FEAT]) if action > 3]
            filtered_actions = [action for action in row[self.ACTIONS_FEAT] if action > 3]
            return pd.Series({self.ITEMS_FEAT: filtered_items, self.ACTIONS_FEAT: filtered_actions})
        df[[self.ITEMS_FEAT, self.ACTIONS_FEAT]] = df.apply(row_filter, axis=1)
        df = df[df[self.ITEMS_FEAT].str.len() > 0]
        return df

    def load_sequences(self) -> pd.DataFrame:
        sequences = self.read_sequences()
        sequences[self.ITEMS_FEAT] = sequences[self.ITEMS_FEAT].astype(str)
        sequences[self.ACTIONS_FEAT] = sequences[self.ACTIONS_FEAT].astype(str)

        sequences[self.ITEMS_FEAT] = sequences[self.ITEMS_FEAT].apply(lambda x: x.split(',') if x else [])
        sequences[self.ACTIONS_FEAT] = sequences[self.ACTIONS_FEAT].apply(
            lambda x: list(map(
                lambda v: int(float(v)), x.split(',')))
            if x else []
        )
        sequences = self.scenario_filter(sequences)

        # filter out sequences with less than 10 items
        sequences = sequences[sequences[self.ITEMS_FEAT].apply(lambda x: len(x) >= self.min_length)]
        sequences = sequences.reset_index(drop=True)
        sequences[self.HISTORY_FEAT] = sequences[self.ITEMS_FEAT].apply(lambda x: x[:-self.max_length])
        sequences[self.ITEMS_FEAT] = sequences[self.ITEMS_FEAT].apply(lambda x: x[-self.max_length:])
        sequences[self.ACTIONS_FEAT] = sequences[self.ACTIONS_FEAT].apply(lambda x: x[-self.max_length:])
        return sequences

    def tokenize_interactions(self, interactions: pd.DataFrame, item_vocab: Vocab, user_vocab: Vocab, dir_name):
        with UniTok() as ut:
            ut.add_index_feature()
            ut.add_feature(tokenizer=EntityTokenizer(vocab=user_vocab), column=self.USER_COL, name=self.USER_FEAT)
            ut.add_feature(tokenizer=EntitiesTokenizer(vocab=item_vocab), column=self.ITEMS_FEAT, name=self.ITEMS_FEAT)
            ut.add_feature(tokenizer=EntitiesTokenizer(vocab=item_vocab), column=self.HISTORY_FEAT, name=self.HISTORY_FEAT)
            ut.add_feature(tokenizer=DigitsTokenizer(vocab='action'), column=self.ACTIONS_FEAT, name=self.ACTIONS_FEAT)
            ut.add_feature(tokenizer=DigitsTokenizer(vocab='eval'), column=self.EVALS_FEAT, name=self.EVALS_FEAT)
            ut.tokenize(interactions).save(dir_name)

    def tokenize_items(self, items, item_vocab):
        with UniTok() as ut:
            ut.add_feature(tokenizer=EntityTokenizer(vocab=item_vocab), column=self.ITEM_COL, name=self.ITEM_FEAT, key=True)
            ut.add_feature(tokenizer=SplitTokenizer(vocab=self.GENRES, sep='|'))
            ut.tokenize(items).save(self.item_dir)

    def process(self):
        sequences = self.load_sequences()
        items = self.load_items(sequences)

        item_vocab = Vocab(name='item')
        user_vocab = Vocab(name='user')

        self.tokenize_items(items, item_vocab=item_vocab)

        item_vocab.deny_edit()

        train_data = sequences.copy()
        train_data[self.EVALS_FEAT] = train_data[self.ITEMS_FEAT].apply(lambda x: [1] * (len(x) - 2) + [0] * 2)

        test_data = sequences.copy()
        test_data[self.EVALS_FEAT] = test_data[self.ITEMS_FEAT].apply(lambda x: [0] * (len(x) - 2) + [1] * 2)

        self.tokenize_interactions(train_data, item_vocab, user_vocab, self.train_dir)
        self.tokenize_interactions(test_data, item_vocab, user_vocab, self.test_dir)

    def analyse(self):
        sequences = self.load_sequences()
        items = self.load_items(sequences)

        print(f'Dataset Statistics for {self.get_classname()}')
        print(f'Sequences: {len(sequences)}')
        print(f'Items: {len(items)}')
        print()

        max_time, min_time = None, None
        for times in tqdm(sequences[self.TIMES_FEAT], total=len(sequences[self.TIMES_FEAT])):
            times = list(map(int, times.split(',')))
            if not times:
                continue
            max_time_ = max(times)
            min_time_ = min(times)
            if max_time is None:
                max_time = max_time_
            if min_time is None:
                min_time = min_time_
            max_time = max(max_time, max_time_)
            min_time = min(min_time, min_time_)

        print(f'Max time: {max_time}, Min time: {min_time}')
