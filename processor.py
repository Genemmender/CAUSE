from typing import Type

import pigmento
from pigmento import pnt

from oba import Obj  # 用于将字典封装成对象，可用 obj.key 方式访问

from loaders.class_hub import ClassHub  # 注册所有 Processor 类的工厂类
from processors.backbone import BackboneProcessor  # 所有 processor 的基类
from utils import io  # 包含 yaml_load 方法
from utils.config_init import CommandInit  # 命令行配置解析类


class MetaProcessor:
    """
    任务入口类：根据配置动态加载 Processor，并运行其 process 或 analyse 流程。
    """

    config_path = '.data.yaml'         # 数据路径配置文件
    data_base_dir = 'data'             # 默认数据目录根路径

    paths = Obj(io.yaml_load(config_path))  # 加载 .data.yaml，转为可对象访问的路径字典

    def __init__(self, configuration):
        """
        初始化 MetaProcessor，负责选择 processor 并传递参数。
        Args:
            configuration: 命令行解析出来的配置对象，包含 config, force, analyse 等字段
        """
        self.config = configuration.config         # 实际的数据配置对象
        self.force = configuration.force           # 是否强制重建数据（通常用于覆盖缓存）
        self.analyse = configuration.analyse       # 是否只运行分析（analyse），跳过主流程

        self.init_pigmento()                       # 初始化 pigmento 打印样式

        self.data_name = self.config.data          # 当前数据集名
        self.processor_name = self.config.processor  # 指定的预处理器名称

        self.processors = ClassHub.processors()    # 从 ClassHub 获取所有注册的 Processor 类
        self.processor_class: Type[BackboneProcessor] = self.processors[self.processor_name]  # 动态获取类

    @classmethod
    def get_data_dir(cls, data_name):
        """
        获取数据名称对应的路径（从 .data.yaml 中查找）。
        """
        return cls.paths[data_name]

    @staticmethod
    def init_pigmento():
        """
        初始化 pigmento 日志样式（带时间戳、彩色、类名显示）。
        """
        pigmento.add_time_prefix()
        pigmento.add_dynamic_color_plugin()
        pnt.set_display_mode(
            display_method_name=False,
            display_class_name=True,
            use_instance_class=True,
        )

    def run(self):
        """
        执行 processor 的流程。
        - 若配置中 `analyse=True`，则只调用 analyse 方法。
        - 否则运行完整 process 流程。
        """
        data_dir = self.paths[self.config.data]  # 获取数据集路径

        processor: BackboneProcessor = self.processor_class(
            name=self.config.name,               # 实验/数据构建任务名称
            config=self.config.config,           # 实际配置参数
            data_dir=data_dir,                   # 数据存储路径
            force=self.force                     # 是否覆盖已有文件
        )

        if self.analyse:
            processor.analyse()  # 仅分析
            return
        processor.process()  # 正常预处理流程


if __name__ == '__main__':
    # -------------------- 入口 --------------------
    # 解析命令行参数：config 是必须参数，其余为默认值
    configurations = CommandInit(
        required_args=('config',),
        default_args=dict(
            force=False,
            analyse=False,
        )
    ).parse()

    # 创建并运行主处理器
    mp = MetaProcessor(configurations)
    mp.run()
