import os
from typing import Literal

from oba import Obj  # 可将字典包装为对象支持 obj.key 形式访问
from pigmento import pnt  # 美化打印函数
from utils import io  # 自定义的 JSON/YAML 读写工具模块


class Meta:
    VERSION: str  # 类变量，需在子类中定义版本号
    meta: Obj     # 用于保存当前加载的 meta 对象（支持属性式访问）

    def __init__(
        self,
        save_dir,
        meta_name,
        extension: Literal['json', 'yaml'] = 'yaml',
    ):
        """
        初始化 Meta 管理器。
        参数:
            - save_dir: 元信息文件保存目录
            - meta_name: 文件名（不带后缀）
            - extension: 文件扩展名，支持 'json' 或 'yaml'
        """
        self.save_dir = save_dir
        self.meta_name = meta_name
        self.extension = extension
        self.meta_path = os.path.join(save_dir, f'{meta_name}.{extension}')  # 拼接完整路径

        os.makedirs(self.save_dir, exist_ok=True)  # 确保保存目录存在
        self.load_handler, self.save_handler = self._parse_handler(extension)  # 根据扩展名绑定 I/O 函数

    @staticmethod
    def _parse_handler(extension):
        """
        根据扩展名返回对应的加载器和保存器函数。
        """
        if extension == 'json':
            return io.json_load, io.json_save
        if extension == 'yaml':
            return io.yaml_load, io.yaml_save
        raise ValueError(f'Invalid file extension: {extension}')  # 不支持的扩展名

    def load_meta(self, default=None):
        """
        加载 meta 文件内容为 self.meta。
        如果文件不存在，则使用 default 或空字典。
        如果版本不一致，发出警告，并更新为当前版本。
        """
        meta = default or {}
        if os.path.exists(self.meta_path):
            meta = self.load_handler(self.meta_path)
        self.meta = Obj(meta)  # 用 oba.Obj 包装字典，支持属性访问

        if self.meta.version != self.VERSION:
            pnt(f'Current {self.__class__.__name__.lower()} version ({self.VERSION}) may be not able to '
                f'parse the meta config of version {self.meta.version}')
            self.meta.version = self.VERSION  # 将旧版本更新为当前类定义的版本

    def save_meta(self):
        """
        保存当前 meta 对象到文件。
        self.meta 是 Obj 类型，需转为字典再写入。
        """
        return self.save_handler(self.meta(), self.meta_path)  # self.meta() 会返回原始 dict
