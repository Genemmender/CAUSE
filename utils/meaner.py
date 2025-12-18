class Meaner:
    def __init__(self):
        """
        初始化均值器。
        - self.mean: 当前的平均值（初始为0）
        - self.count: 已添加的数值个数
        """
        self.mean = 0
        self.count = 0

    def add(self, value):
        """
        向均值器中添加一个新的数值，更新当前的平均值。
        使用在线算法公式更新，不需要保存所有历史数据。
        """
        self.mean = (self.mean * self.count + value) / (self.count + 1)
        self.count += 1

    def __call__(self, value):
        """
        允许像函数一样调用实例：meaner(x) 等价于 meaner.add(x)，并返回当前均值。
        """
        self.add(value)
        return self.mean
