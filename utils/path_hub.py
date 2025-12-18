import psutil
from pigmento import pnt

from utils import io

import os


class BasePathHub:
    CHECKPOINT_BASE_DIR = 'checkpoints'  # 基础检查点目录

    def __init__(self, data_name, model_name, signature):
        # 初始化时传入数据集名、模型名和实验签名
        self.data_name = data_name
        self.model_name = model_name
        self.signature = signature

    def is_alive(self):
        # 判断当前实验进程是否存活
        if not os.path.exists(self.pid_path):
            return False
        pid = int(io.file_load(self.pid_path))  # 读取存储的PID
        return psutil.pid_exists(pid)  # 利用psutil判断进程是否存在

    def _get_path(self, extension):
        return os.path.join(self.checkpoint_base_dir, f'{self.signature}.{extension}')

    @property
    def checkpoint_base_dir(self):
        # 组装基础路径：checkpoints/{data_name}/{model_name}
        return os.path.join(self.CHECKPOINT_BASE_DIR, self.data_name, self.model_name)

    @property
    def log_path(self):
        # 日志文件路径，例如 checkpoints/data/model/signature.log
        # return os.path.join(self.checkpoint_base_dir, f'{self.signature}.log')
        return self._get_path('log')

    @property
    def cfg_path(self):
        # 配置文件路径，json格式
        # return os.path.join(self.checkpoint_base_dir, f'{self.signature}.json')
        return self._get_path('json')

    @property
    def ckpt_path(self):
        # 模型检查点文件路径，pt格式（PyTorch模型）
        # return os.path.join(self.checkpoint_base_dir, f'{self.signature}.pt')
        return self._get_path('pt')

    @property
    def result_path(self):
        # 结果文件路径，csv格式
        # return os.path.join(self.checkpoint_base_dir, f'{self.signature}.csv')
        return self._get_path('csv')

    @property
    def progress_path(self):
        # 训练进度文件路径，jsonl格式（逐行json）
        # return os.path.join(self.checkpoint_base_dir, f'{self.signature}.jsonl')
        return self._get_path('jsonl')

    @property
    def pid_path(self):
        # 存储当前实验进程PID的文件路径
        # return os.path.join(self.checkpoint_base_dir, f'{self.signature}.pid')
        return self._get_path('pid')

    @property
    def neg_pool_path(self):
        return self._get_path('pool')


class PathHub(BasePathHub):
    def __init__(self, **kwargs):
        # 继承BasePathHub构造函数，参数通过kwargs传递
        super().__init__(**kwargs)

        # 确保基础检查点目录存在
        os.makedirs(self.checkpoint_base_dir, exist_ok=True)

        # 注册实验，写入当前进程PID，防止重复运行
        self.register_experiment()

        # 初始化日志文件，清空内容（如果已存在）
        with open(self.log_path, 'w') as f:
            pass

        # 初始化训练进度文件，清空内容（如果已存在）
        with open(self.progress_path, 'w') as f:
            pass

    def register_experiment(self):
        # 检查当前实验是否已经在运行
        if self.is_alive():
            pnt(f"Experiment {self.signature} is already running, the live experiment will be terminated.")
            raise ValueError("Experiment is already running")

        # 写入当前进程ID到pid文件，标记该实验正在运行
        pid = os.getpid()
        io.file_save(self.pid_path, str(pid))
