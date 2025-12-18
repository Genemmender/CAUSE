import collections
import math

import termplot


class Plot:
    """
    用于绘制列表元素长度分布的类，支持在终端以字符图的形式展示数据分布。
    """

    @classmethod
    def draw_counts(cls, counts: list, height: int = 10):
        """
        根据数值列表 counts 绘制垂直条形图，条形符号为 '|', 图高为 height。
        :param counts: 数值列表，表示每个区间/类别的计数。
        :param height: 图形的高度，默认10行。
        """
        termplot.plot(
            x=counts,
            plot_char='|',
            plot_height=height,
        )

    @classmethod
    def draw_groups(cls, groups: dict, sort_func=None, height: int = 10):
        """
        根据字典 groups（key->计数）绘制分组计数的分布图。
        :param groups: dict，key为类别，value为该类别计数。
        :param sort_func: 自定义排序函数，用于排序key。
        :param height: 图高，默认10。
        """
        keys = list(groups.keys())
        keys = sorted(keys, key=sort_func)
        counts = [groups[k] for k in keys]
        print(f'[ X-LEFT: {keys[0]} ]')   # 显示x轴最左侧标签
        print(f'[ X-RIGHT: {keys[-1]} ]') # 显示x轴最右侧标签
        cls.draw_counts(counts, height)

    @classmethod
    def draw_distribution(cls, values: list, height: int = 10, groups: int = 50):
        """
        将数值列表values分桶统计后绘制直方图，自动计算分组间隔和个数。
        :param values: 输入数值列表。
        :param height: 图高，默认10。
        :param groups: 期望分组数，默认50。
        """
        min_value, max_value = min(values), max(values)
        range_ = max_value - min_value + 1  # 值域范围，含边界
        if range_ < groups:
            groups = range_  # 如果范围小于分组数，则分组数调整为范围大小
        interval = math.ceil(range_ / groups)  # 每个桶的宽度（区间长度）
        groups = math.ceil(range_ / interval)  # 重新计算桶数，确保覆盖所有数据

        counts = [0] * groups  # 初始化各桶计数为0
        for v in values:
            bucket = (v - min_value) // interval  # 计算值所在桶的索引
            counts[bucket] += 1  # 桶计数加1

        print(f'[ X-INT: {interval} ]')        # 打印每个桶的区间长度
        print(f'[ Y-INT: {max(counts) // height} ]')  # 打印Y轴刻度间隔估计
        cls.draw_counts(counts, height)  # 绘制条形图


if __name__ == '__main__':
    a = [1, 2, 3, 4, -5, 5, -4, -1, 0, -10, -4, -2, 3, 5, 8, 10, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
    c = collections.Counter(a)  # 统计各元素出现次数
    Plot.draw_distribution(a)   # 绘制数值分布直方图

    Plot.draw_groups(c)         # 根据计数绘制分组分布图