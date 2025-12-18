from processors.kuairand import KuairandProcessor


class KuairandLargeProcessor(KuairandProcessor):
    """
    Kuairand 大规模数据处理器，继承自 KuairandProcessor，
    主要区别在于使用适配 27K 数据集的 Reader。
    """

    def load_reader(self):
        """
        加载用于读取 27K 数据集的 reader（Kuairand27KReader）。
        该 reader 会加载四个分片的交互日志文件。
        """
        from processors import kuairand_readers as readers
        return readers.Kuairand27KReader(self)
