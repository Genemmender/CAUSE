import os
import pandas as pd
from pigmento import pnt

from processors.kuairand import KuairandProcessor


class KuairandBaseReader:
    """
    Kuairand 数据读取器基类，主要负责 item 和 interaction 数据的加载。
    子类通过设置 `item_path` 和实现 `read_interactions` 来定制加载逻辑。
    """

    item_path: str  # item 文件的相对路径，由子类指定

    def __init__(self, processor: KuairandProcessor):
        self.processor = processor

    def read_items(self):
        """
        加载视频 item 元数据（如 music_type、tag 等）。
        优先读取缓存的 pkl 文件，否则从 csv 中读取并缓存。
        """
        pnt(f'Reading items ...')
        pickle_path = self.item_path.replace('csv', 'pkl')
        if os.path.exists(pickle_path):
            return pd.read_pickle(pickle_path)

        items = pd.read_csv(
            filepath_or_buffer=os.path.join(self.processor.data_dir, 'data', self.item_path),
            sep=',',
            names=[
                self.processor.ITEM_COL, 'author_id', 'video_type', 'upload_at', 'upload_type',
                'visible_status', 'video_duration', 'server_width', 'server_height', 'music_id',
                self.processor.MUSIC_TYPE, self.processor.TAG
            ],
            usecols=[
                self.processor.ITEM_COL, self.processor.MUSIC_TYPE, self.processor.TAG,
            ],
        )

        # 将 video_id 强制转为字符串，保证一致性
        pnt(f'Stringify {self.processor.ITEM_COL} column ...')
        items[self.processor.ITEM_COL] = items[self.processor.ITEM_COL].apply(str)

        items.to_pickle(pickle_path)
        return items

    def _load_interactions(self, path: str):
        """
        加载用户交互日志，支持缓存（pkl 文件）。
        读取列包括用户、视频、时间戳和各种行为标签。
        """
        pickle_path = path.replace('.csv', '.pkl')
        if os.path.exists(pickle_path):
            return pd.read_pickle(pickle_path)

        interactions = pd.read_csv(
            filepath_or_buffer=path,
            sep=',',
            names=[
                self.processor.USER_COL, self.processor.ITEM_COL, self.processor.DATE_COL, 'hourmin',
                self.processor.TIME_COL, self.processor.IS_CLICK, self.processor.IS_LIKE, self.processor.IS_FOLLOW,
                self.processor.IS_COMMENT, self.processor.IS_FORWARD, self.processor.IS_HATE, self.processor.LONG_VIEW,
                'play_time_ms', 'duration_ms', 'profile_stay_time',
                'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab',
            ],
            usecols=[
                self.processor.USER_COL, self.processor.ITEM_COL, self.processor.TIME_COL, self.processor.DATE_COL,
                self.processor.IS_CLICK, self.processor.IS_LIKE, self.processor.IS_FOLLOW, self.processor.IS_COMMENT,
                self.processor.IS_FORWARD, self.processor.IS_HATE, self.processor.LONG_VIEW,
            ],
        )

        # 将必要字段强制转为字符串，避免 index 错误或 vocab 错误
        pnt(f'Stringify {self.processor.DATE_COL} column ...')
        interactions[self.processor.DATE_COL] = interactions[self.processor.DATE_COL].astype(str)
        pnt(f'Stringify {self.processor.USER_COL} column ...')
        interactions[self.processor.USER_COL] = interactions[self.processor.USER_COL].astype(str)
        pnt(f'Stringify {self.processor.ITEM_COL} column ...')
        interactions[self.processor.ITEM_COL] = interactions[self.processor.ITEM_COL].astype(str)

        interactions.to_pickle(pickle_path)
        return interactions

    def read_interactions(self):
        """
        抽象接口：由子类实现，负责加载全部交互数据。
        """
        raise NotImplementedError


class Kuairand1KReader(KuairandBaseReader):
    """
    加载 1K 数据集的 reader，适用于小规模样本。
    """

    item_path = 'video_features_basic_1k.csv'

    def read_interactions(self):
        pnt('Reading interactions for 1k dataset: 1/2')
        interactions_a = self._load_interactions(
            os.path.join(self.processor.data_dir, 'data', 'log_standard_4_08_to_4_21_1k.csv'))

        pnt('Reading interactions for 1k dataset: 2/2')
        interactions_b = self._load_interactions(
            os.path.join(self.processor.data_dir, 'data', 'log_standard_4_22_to_5_08_1k.csv'))

        return pd.concat([interactions_a, interactions_b])


class Kuairand27KReader(KuairandBaseReader):
    """
    加载 27K 数据集的 reader，适用于大规模样本。
    """

    item_path = 'video_features_basic_27k.csv'

    def read_interactions(self):
        pnt('Reading interactions for 27k dataset: 1/4')
        interactions_a1 = self._load_interactions(
            os.path.join(self.processor.data_dir, 'data', 'log_standard_4_08_to_4_21_27k_part1.csv'))

        pnt('Reading interactions for 27k dataset: 2/4')
        interactions_a2 = self._load_interactions(
            os.path.join(self.processor.data_dir, 'data', 'log_standard_4_08_to_4_21_27k_part2.csv'))

        pnt('Reading interactions for 27k dataset: 3/4')
        interactions_b1 = self._load_interactions(
            os.path.join(self.processor.data_dir, 'data', 'log_standard_4_22_to_5_08_27k_part1.csv'))

        pnt('Reading interactions for 27k dataset: 4/4')
        interactions_b2 = self._load_interactions(
            os.path.join(self.processor.data_dir, 'data', 'log_standard_4_22_to_5_08_27k_part2.csv'))

        return pd.concat([interactions_a1, interactions_a2, interactions_b1, interactions_b2])
