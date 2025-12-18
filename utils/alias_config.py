import os.path
import string
from typing import Dict, Set

from pigmento import pnt
from utils import function, io
from utils.meta import Meta


class AliasConfig(Meta):
    """
    配置别名注册与管理器，基于 Meta 继承实现，用于将配置文件绑定到别名，并支持唯一签名生成。
    """

    VERSION = '1'  # 当前配置版本号

    def __init__(self, save_dir):
        # 初始化并加载已有 meta 配置（包含 aliases 字典）
        super().__init__(meta_name='.alias', save_dir=save_dir, extension='yaml')
        self.load_meta(default=dict(version=self.VERSION, aliases={}))

        self.aliases = self.meta.aliases()  # 别名 -> 签名 映射
        self.signs = self.get_signs()       # 签名 -> set(别名) 映射

    def get_signs(self):
        """
        构造签名到别名的反向映射字典。
        """
        signs = {}
        for alias, sign in self.aliases.items():
            if sign not in signs:
                signs[sign] = set()
            signs[sign].add(alias)
        return signs

    def save_config(self, sign, config):
        """
        将某一签名对应的配置保存到本地文件。
        """
        sign_path = os.path.join(self.save_dir, f'{sign}.{self.extension}')
        return self.save_handler(config, sign_path)

    def load_config(self, sign):
        """
        加载某一签名的配置文件。
        """
        sign_path = os.path.join(self.save_dir, f'{sign}.{self.extension}')
        if not os.path.exists(sign_path):
            raise FileNotFoundError(f'Config file {sign_path} does not exist.')
        return self.load_handler(sign_path)

    @staticmethod
    def validate_alias(alias: str):
        """
        检查别名是否合法，仅允许小写字母、数字、下划线、点号。
        """
        allow_chars = string.ascii_lowercase + string.digits + '_.'
        if not all(c in allow_chars for c in alias):
            raise ValueError(f'Alias {alias} contains invalid characters. '
                             f'Only letters, digits, "_" and "." are allowed.')

    def register(self, config: dict, alias=None, force=False):
        """
        注册配置并为其绑定别名，返回配置的唯一签名（hash 值）。
        如果配置已存在（签名已存在），则只更新别名。
        """
        sign = function.get_signature(config)  # 根据配置生成唯一签名
        config_str = io.json_dumps(config)     # 转为字符串展示

        # 获取当前签名的别名（如果存在）
        aliases = []
        if sign in self.signs:
            aliases = list(map(lambda x: '#' + x, self.signs[sign]))
        else:
            self.signs[sign] = set()
        alias_str = ', '.join(aliases)

        # 打印当前配置与别名绑定情况
        display_sentence = f'Following configuration is signatured by @{sign}'
        if alias_str:
            display_sentence += f', also known as {alias_str}'
        display_sentence += ':'
        pnt(display_sentence)
        pnt(config_str)

        # 保存配置到文件
        self.save_config(sign, config)

        if not alias:
            return sign  # 不设置别名则直接返回签名

        self.validate_alias(alias)  # 检查别名是否合法

        # 如果当前 alias 是新别名，添加到映射中
        if alias not in aliases:
            pnt(f'Alias #{alias} is registered for signature @{sign}')
            self.signs[sign].add(alias)

        # 如果 alias 已存在但指向不同的签名，处理冲突
        if alias in self.aliases and self.aliases[alias] != sign:
            if force:
                self.signs[self.aliases[alias]].remove(alias)  # 强制更新绑定
            else:
                raise ValueError(f'Alias #{alias} is already used by signature @{self.aliases[alias]}.')

        self.aliases[alias] = sign  # 更新别名映射
        self.save_meta()  # 保存 meta 数据

        return sign


if __name__ == '__main__':
    # 示例代码，注册不同配置与别名
    ac = AliasConfig(save_dir='test')
    ac.register(dict(a=1, b=2), 'hello')
    ac.register(dict(a=1, b=2), 'world')
    ac.register(dict(a=1, b=3), 'hello.')
