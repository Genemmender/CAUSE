from unitok import Vocab, UniTok


class ColumnMap:
    """
    ColumnMap 类用于映射和管理训练过程中使用的各列字段（如用户列、物品列、行为列等），
    并提供对 UniTok 编码信息的访问接口。
    """

    # UniTok 实例：主数据（interactions）和 item 数据
    ut: UniTok = None
    item_ut: UniTok = None

    def __init__(self, user_col, items_col, actions_col, evals_col, histories_col):
        """
        初始化 ColumnMap，定义并注册类型列。

        :param user_col: 用户列名
        :param items_col: 物品序列列名
        :param actions_col: 行为序列列名
        :param evals_col: 标签列名（eval）
        :param histories_col: 历史物品列名（通常是 item history）
        """
        self.types = Vocab(name='type')  # 管理字段类型的词表

        self.user_col = user_col
        self.items_col = items_col
        self.actions_col = actions_col
        self.evals_col = evals_col
        self.histories_col = histories_col

        self.history_group_col = None  # 可选字段，表示历史聚合用的列（暂未使用）

        # 注册基础列（用于主数据）
        self.types.append(self.user_col)
        self.types.append(self.items_col)
        self.types.append(self.actions_col)

    def load_ut(self, ut, item_ut):
        """
        加载主 UniTok 和 item UniTok（用于支持统一字段管理）。

        :param ut: 主 UniTok（interactions）
        :param item_ut: 物品 UniTok（items）
        """
        self.ut = ut
        self.item_ut = item_ut

    def get_features(self, item=False):
        """
        获取 UniTok 的字段元信息 features。

        :param item: 是否获取 item 的 UniTok features
        :return: features 字典
        """
        ut = self.item_ut if item else self.ut
        assert ut, ValueError(f'UniTok (item={item}) is not loaded yet.')
        return ut.meta.features

    def get_size(self, col, item=False):
        """
        获取某列的 vocab 大小。

        :param col: 列名
        :param item: 是否使用 item 的 UniTok
        :return: vocab size
        """
        features = self.get_features(item)
        return features[col].tokenizer.vocab.size

    def get_max_length(self, col, item=False):
        """
        获取某列的最大序列长度。

        :param col: 列名
        :param item: 是否使用 item 的 UniTok
        :return: max_len 值
        """
        features = self.get_features(item)
        return features[col].max_len
