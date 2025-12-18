import refconfig
from oba import Obj
from refconfig import RefConfig

from utils import function


# 检查并确保依赖包版本满足要求，否则抛出异常提示安装或升级
function.package_require('refconfig', '0.1.2')
function.package_require('smartdict', '0.2.1')
function.package_require('unitok', '4.4.2')
function.package_require('oba', '0.3.1')
function.package_require('pigmento', '0.2.3')


class CommandInit:
    """
    用于命令行参数初始化的辅助类，支持检查必填参数和默认参数填充。
    """
    def __init__(self, required_args, default_args=None):
        self.required_args = required_args  # 必填参数列表
        self.default_args = default_args or {}  # 默认参数字典

    def parse(self, kwargs=None):
        # 解析传入的参数字典，如果为空则从命令行参数解析
        kwargs = kwargs or function.argparse()

        # 检查是否缺少必填参数
        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f'miss argument {arg}')

        # 填充缺省参数
        for arg in self.default_args:
            if arg not in kwargs:
                kwargs[arg] = self.default_args[arg]

        try:
            # 利用 RefConfig 解析参数，返回包装好的Obj对象
            config = RefConfig().add(refconfig.CType.SMART, **kwargs).parse()
            config = Obj(config)
        except Exception as e:
            print('! config parse error')
            print(e)
            exit(0)

        return config

    @classmethod
    def update(cls, old: dict, new: dict):
        for key in old:
            if key not in new:
                del old[key]

        for key, value in new.items():
            if key not in old:
                old[key] = value
                continue

            if isinstance(old[key], dict) ^ isinstance(value, dict):
                raise ValueError(f"Key {key} is a {type(old[key])} and cannot be updated with a {type(value)}")

            if isinstance(old[key], dict) and isinstance(value, dict):
                cls.update(old[key], value)

        return old
