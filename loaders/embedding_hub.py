from torch import nn

from loaders.column_map import ColumnMap


class EmbeddingHub:
    """
    EmbeddingHub 负责统一管理多个列的嵌入（embedding）表，支持通过 ColumnMap
    根据列名自动构建对应维度的 embedding 层。
    """

    def __init__(self, embedding_dim, cm: ColumnMap):
        """
        初始化 EmbeddingHub。

        :param embedding_dim: 嵌入维度
        :param cm: ColumnMap 实例，用于获取列信息及其 vocab 大小
        """
        self.embedding_dim = embedding_dim
        self.cm = cm

        self._table = nn.ModuleDict()  # 存放嵌入层的字典

    def get_table(self):
        """
        获取 embedding 层的 ModuleDict（可用于 nn.Module 的 forward 传递）。

        :return: ModuleDict of embeddings
        """
        return self._table

    def get_key(self, key):
        """
        获取嵌入层在 table 中的键名。

        :param key: 原始列名
        :return: 加前缀后的 key，例如 '@user_id'
        """
        return '@' + key

    def register(self, col, item=False):
        """
        为指定列注册 embedding 层（如果尚未存在），自动从 ColumnMap 获取 vocab size。

        :param col: 列名
        :param item: 是否为 item 表中的列
        """
        vocab_size = self.cm.get_size(col, item=item)

        self.add_table(col, vocab_size)

    def add_table(self, name, vocab_size):
        name = self.get_key(name)

        if name in self._table:
            return  # 已注册，跳过

        self._table.add_module(name, nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim
        ))
