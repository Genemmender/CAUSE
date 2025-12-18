from tqdm import tqdm


class Unset:
    """
    用于标记尚未设置的属性。
    """
    pass


class Bar:
    """
    通用进度条构造类，用于封装 tqdm 的配置与调用逻辑。
    可通过子类进一步定制训练、验证等阶段的进度条样式。
    """
    def __init__(self):
        self.iterable = Unset     # 要遍历的数据（如 DataLoader）
        self.bar_format = Unset   # 进度条显示格式
        self.leave = Unset        # 迭代结束后是否保留进度条
        self.kwargs = dict()      # 额外的 tqdm 参数

    def is_filled(self):
        """
        判断所有基础字段是否都已被设置。
        """
        for key in self.__dict__:
            if self.__dict__[key] is Unset:
                return False
        return True

    def get_config(self):
        """
        获取 tqdm 所需的全部参数配置，整合基础字段和额外参数。
        """
        kwargs = dict()
        kwargs.update(self.kwargs)
        for key in self.__dict__:
            if key != "kwargs":
                kwargs[key] = self.__dict__[key]
        return kwargs

    def __call__(self, iterable=None, **kwargs):
        """
        可调用对象，传入 iterable 和 tqdm 参数后生成 tqdm 进度条。
        支持动态设置属性与配置。
        """
        if iterable is not None:
            kwargs["iterable"] = iterable

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)  # 设置基础字段
            else:
                self.kwargs[key] = value   # 设置额外参数

        if self.is_filled():
            return tqdm(**self.get_config())  # 返回 tqdm 对象


class TrainBar(Bar):
    """
    训练阶段的进度条，显示当前 epoch 和进度。
    """
    def __init__(self, epoch):
        super().__init__()
        self.bar_format = "Training Epoch {epoch} [{percentage:.0f}% < {remaining}] {postfix}"
        self.bar_format = self.bar_format.replace("{epoch}", str(epoch))  # 替换 epoch 字段
        self.leave = False  # 不保留进度条


class ValBar(Bar):
    """
    验证阶段进度条，展示训练损失与当前 epoch。
    """
    def __init__(self, epoch, train_loss):
        super().__init__()
        self.bar_format = "Train loss: {train_loss} | Validating Epoch {epoch} [{percentage:.0f}% < {remaining}] {postfix}"
        self.bar_format = self.bar_format.replace("{epoch}", str(epoch))
        self.bar_format = self.bar_format.replace("{train_loss}", f"{train_loss:.4f}")
        self.leave = False


class TestBar(Bar):
    """
    测试阶段的进度条，仅展示测试进度。
    """
    def __init__(self):
        super().__init__()
        self.bar_format = "Testing [{percentage:.0f}% < {remaining}] {postfix}"
        self.leave = False


class DescBar(Bar):
    """
    通用描述信息进度条，用于任意任务阶段。
    """
    def __init__(self, desc):
        super().__init__()
        self.bar_format = "{desc} [{percentage:.0f}% < {remaining}]"
        self.bar_format = self.bar_format.replace("{desc}", desc)
        self.leave = False
