from unitok import Vocab

from loaders.symbols import Symbols as Syb
from utils.path_hub import PathHub


class Env:
    """
    Env 是全局环境配置类，保存运行时的状态信息、设备信息、路径配置等。
    常用于模型训练和评估阶段中的全局共享参数，如 device/null/type 词表等。
    """

    null = -1  # 缺失值标记
    num_items: int  # 物品总数（由外部赋值）

    debug = False  # 调试模式开关
    device = None  # 当前运行设备（如 'cuda' 或 'cpu'）
    ph: PathHub = None  # 路径管理类，由外部初始化

    types = Vocab(name='types')  # 全局 token 类型词表（如 user/item/action）

    status = Syb.is_initializing  # 当前运行状态，初始化为 initializing

    global_validation: bool

    @classmethod
    def is_training(cls):
        """
        判断当前是否处于训练状态。
        """
        return cls.status is Syb.is_training

    @classmethod
    def is_evaluating(cls):
        """
        判断当前是否处于评估状态。
        """
        return cls.status is Syb.is_evaluating

    @classmethod
    def start_training(cls):
        """
        设置当前状态为训练。
        """
        cls.status = Syb.is_training

    @classmethod
    def start_evaluating(cls):
        """
        设置当前状态为评估。
        """
        cls.status = Syb.is_evaluating
