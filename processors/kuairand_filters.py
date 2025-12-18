import pandas as pd

from processors.backbone import Sequence
from processors.kuairand import KuairandProcessor


class KuairandBaseFilter:
    """
    Kuairand 数据过滤器的基类，定义三个核心处理函数的接口：
    - 场景过滤 scenario_filter
    - 构建动作列 build_action
    - 历史筛选 filter_history
    子类需实现不同过滤逻辑。
    """

    def __init__(self, processor: KuairandProcessor):
        self.processor = processor  # 上层 Processor，包含字段常量等

    def scenario_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        用于过滤特定行为的数据场景，例如点击行为。
        子类实现具体过滤策略。
        """
        raise NotImplementedError

    def build_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        用于构建 `action` 列，可根据不同行为组合。
        """
        raise NotImplementedError

    def filter_history(self, sequence: Sequence):
        """
        构建当前行为序列时，如何筛选历史序列作为辅助。
        """
        raise NotImplementedError


class KuairandMAFilter(KuairandBaseFilter):
    """
    MA 过滤器：
    - 仅保留点击行为（IS_CLICK == 1）
    - 构建的 action 为 IS_CLICK + LONG_VIEW（是否为长时观看）
    - 过滤历史时保留所有出现的 item
    """

    def scenario_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        # 保留点击行为样本
        return df[df[self.processor.IS_CLICK] == 1]

    def build_action(self, df: pd.DataFrame) -> pd.DataFrame:
        # 点击 + 长观看行为合并成 action（可表示偏好强度）
        df[self.processor.ACTION_COL] = df[self.processor.IS_CLICK] + df[self.processor.LONG_VIEW]
        return df

    def filter_history(self, sequence: Sequence):
        # 保留所有历史项
        return sequence.items


class KuairandSortFilter(KuairandBaseFilter):
    """
    Sort 过滤器：
    - 不过滤行为，保留正负样本混合场景
    - action 仅使用点击行为
    - 历史筛选时仅保留点击过的 item
    """

    def scenario_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        # 不做行为过滤，允许存在未点击行为
        return df

    def build_action(self, df: pd.DataFrame) -> pd.DataFrame:
        # 使用点击行为作为 action 标签
        df[self.processor.ACTION_COL] = df[self.processor.IS_CLICK]
        return df

    def filter_history(self, sequence: Sequence):
        # 只保留点击过的 item，action==1
        return [item for item, action in zip(sequence.items, sequence.actions) if action == 1]
