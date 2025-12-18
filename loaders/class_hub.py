import glob
import importlib
import os.path
from pathlib import Path


class ClassHub:
    """
    ClassHub 是一个通用的类加载器，用于动态查找、加载并管理指定目录下继承自某个基类的所有类。
    可用于模型（Model）、处理器（Processor）、数据集（Dataset）等模块的自动注册与调用。
    """

    @staticmethod
    def models():
        """
        加载 models 目录下继承自 BackboneModel 的所有模型类。
        """
        from models.backbone import BackboneModel
        return ClassHub(BackboneModel, 'models', 'Model')

    @staticmethod
    def processors():
        """
        加载 processors 目录下继承自 BackboneProcessor 的所有处理器类。
        """
        from processors.backbone import BackboneProcessor
        return ClassHub(BackboneProcessor, 'processors', 'Processor')

    @staticmethod
    def datasets():
        """
        加载 loaders/datasets 目录下继承自 BackboneDataset 的所有数据集类。
        """
        from loaders.datasets.backbone import BackboneDataset
        return ClassHub(BackboneDataset, os.path.join('loaders', 'datasets'), 'Dataset')

    @staticmethod
    def visualizers():
        """
        加载 visualizers 目录下继承自 BackboneVisualizer 的所有可视化类
        """
        from visualizers.backbone import BackboneVisualizer
        return ClassHub(BackboneVisualizer, 'visualizers', 'Visualizer')

    def __init__(self, base_class, module_dir: str, module_type: str):
        """
        初始化 ClassHub。

        :param base_class: 基类，例如 BackboneModel
        :param module_dir: 模块所在目录，例如 'models'
        :param module_type: 模块类型（用于命名裁剪），例如 'Model'
        """
        self.base_class = base_class
        self.module_dir = module_dir
        self.module_type = module_type.lower()
        self.upper_module_type = self.module_type.capitalize()

        self.class_list = self.get_class_list()  # 获取所有符合条件的类
        self.class_dict = dict()
        for class_ in self.class_list:
            name = class_.__name__
            # 裁剪掉类名中的后缀，如 DNNModel -> DNN
            name = name.replace(self.upper_module_type, '')
            self.class_dict[name.lower()] = class_

    def get_class_list(self):
        """
        获取目录下所有继承自 base_class 的类。

        :return: List[Type]
        """
        filepaths = glob.glob(str(Path(self.module_dir) / '*.py'))
        class_list = []

        for filepath in filepaths:
            path_obj = Path(filepath)
            file_name = path_obj.stem  # 文件名（不带扩展名）
            # 构造 Python 可导入的模块路径，例如 processors.kuairand
            module_path = ".".join(Path(self.module_dir).parts + (file_name,))

            module = importlib.import_module(module_path)

            for name, obj in module.__dict__.items():
                # 筛选符合要求的类：继承自 base_class 且不是 base_class 本身
                if isinstance(obj, type) and issubclass(obj, self.base_class) and obj is not self.base_class:
                    class_list.append(obj)

        return class_list

    def __call__(self, name):
        """
        通过类名（不区分大小写）获取类。
        """
        return self.class_dict[name.lower()]

    def __getitem__(self, name):
        """
        通过下标方式访问类。
        """
        return self.class_dict[name.lower()]

    def __contains__(self, name):
        """
        判断类名是否存在于当前 ClassHub。
        """
        return name.lower() in self.class_dict

    def list(self):
        """
        获取当前目录下所有可用的类名（已剪裁）。
        """
        return list(self.class_dict.keys())
