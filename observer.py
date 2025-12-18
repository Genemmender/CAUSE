import glob
import os.path
from pathlib import Path
from typing import Dict, Type

import pigmento
from pigmento import pnt  # 彩色日志打印
from rich.console import Console  # rich 库：命令行打印表格
from rich.table import Table
from rich.text import Text

from loaders.class_hub import ClassHub
from loaders.symbols import Symbols as Syb  # 枚举类型标志
from utils import io  # JSON/JSONL/YAML 等文件读写工具
from utils.config_init import CommandInit
from utils.file_time import FileTime
from utils.path_hub import PathHub  # 路径管理类
from visualizers.backbone import BackboneVisualizer
from visualizers.experiment import Experiment


class Ob:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.name = self.filepath.stem
        self.ft = FileTime(filepath)
        self.num_experiments = len(io.json_load(filepath))


class Observer:
    """
    实验观察者（实验追踪与版本管理器）
    管理所有注册的实验，支持 push（注册）/pop（删除）/summarize（打印）操作。
    """
    OBSERVER_BASE_DIR = 'observers'  # 存储所有 observer 的目录
    VERSION = '1.0'

    SORT_CREATE_TIME = 'ct'
    SORT_NAME = 'name'

    def __init__(self, config):
        self.action = config.action
        self.ob = config.ob
        self.config = config

        self.init_pigmento()

        os.makedirs(self.OBSERVER_BASE_DIR, exist_ok=True)

        if self.ob:
            self.meta: Dict[str, Experiment] = self.load_meta()  # 加载所有注册的实验（key: name）

        # self.summarize()  # 启动时打印摘要

    def get_visualizer(self):
        visualizers = ClassHub.visualizers()
        visualizer_class = visualizers(self.config.v)  # type: Type[BackboneVisualizer]
        return visualizer_class(experiments=list(self.meta.values()), ob=self.ob)

    @staticmethod
    def init_pigmento():
        pigmento.add_time_prefix()
        pigmento.add_dynamic_color_plugin()
        pnt.set_display_mode(
            display_method_name=False,
            display_class_name=True,
            use_instance_class=True,
        )

    @classmethod
    def load_observers(cls):
        filepaths = glob.glob(str(Path(cls.OBSERVER_BASE_DIR) / '*.json'))
        obs = []
        for filepath in filepaths:
            obs.append(Ob(filepath))
        return obs

    @classmethod
    def get_meta_path(cls, ob):
        return os.path.join(cls.OBSERVER_BASE_DIR, f'{ob}.json')

    @property
    def meta_path(self):
        return self.get_meta_path(self.ob)

    @classmethod
    def load_meta_config(cls, ob):
        """
        从 observer JSON 文件加载所有实验的结构。
        """
        pnt(f'Observer {ob} is loading ...')
        meta: dict = io.json_load(cls.get_meta_path(ob))

        for name, config in meta.items():
            meta[name] = Experiment(
                name,
                data_name=config['data'],
                model_name=config['model'],
                signature=config['signature']
            )

        return meta

    def load_meta(self):
        """
        从 observer JSON 文件加载所有实验的结构。
        """
        if not os.path.exists(self.meta_path):
            pnt(f'Observer {self.ob} is successfully created ...')
            return {}

        return self.load_meta_config(self.ob)

    def save_meta(self):
        """
        将所有实验的 meta 信息存储到 JSON 文件中。
        """
        meta = {k: v.json() for k, v in self.meta.items()}
        io.json_save(meta, self.meta_path)

    @classmethod
    def parse_checkpoint_path(cls, filepath: Path):
        """
        从 checkpoint 路径解析出：data, model, signature
        例如：checkpoints/ml-1m/DotRec/<signature>.json
        """
        parts = filepath.parts
        assert len(parts) == 4, f'Invalid checkpoint path: {filepath}'
        _, data, model, _ = parts
        signature = filepath.stem
        return data, model, signature

    @classmethod
    def collect_experiments(cls):
        """
        遍历所有 checkpoint，收集 signature → (data, model) 映射。
        """
        filepaths = list(Path(PathHub.CHECKPOINT_BASE_DIR).rglob('*.json'))
        pnt(f'Searching among {len(filepaths)} experiments ...')

        experiments = dict()
        for filepath in filepaths:
            data, model, signature = cls.parse_checkpoint_path(filepath)
            experiments[signature] = data, model

        return experiments

    def push(self):
        """
        注册一个新的实验别名到当前 observer。
        - 检查重复
        - 验证 checkpoint 是否存在
        - 创建 Experiment 并添加到 meta
        """
        sign, name = self.config.sign, self.config.name
        if name in self.meta:
            if self.meta[name].signature == sign:
                raise ValueError(f'Experiment @{sign} is already registered as #{name}.')
            raise ValueError(f'#{name} is already registered for experiment @{self.meta[name].signature}')

        experiments = self.collect_experiments()
        if sign not in experiments:
            raise ValueError(f'Experiment @{sign} not found.')

        data, model = experiments[sign]
        self.meta[name] = Experiment(name, data_name=data, model_name=model, signature=sign)
        pnt(f'Experiment @{self.meta[name].signature} will be registered as #{name} ...')

        self.save_meta()

    def pop(self):
        """
        注销一个已注册的实验。
        """
        name = self.config.name
        if name not in self.meta:
            raise ValueError(f'#{name} is not registered.')

        pnt(f'Experiment @{self.meta[name].signature} will be popped ...')
        del self.meta[name]
        self.save_meta()

    def summarize(self):
        """
        打印所有注册实验的状态（以 rich 表格形式呈现）。
        包括当前 epoch、input 类型、是否结束、耐心值等。
        """
        console = Console()

        introduction_header = Text.assemble(
            (
                f"Observer (v{self.VERSION}), "
                f"Meta filepath: {self.meta_path}\n",
                "bold cyan"),
            (f"Registered Experiments: {len(self.meta)}\n", "green"),
            style="dim"
        )

        table = Table(title="Experiments", expand=True, title_style="bold yellow", show_lines=True)

        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Signature", justify="center", style="green")
        table.add_column("Data", justify="left", style="blue")
        table.add_column("Model", justify="left", style="magenta")
        table.add_column('Inputs', justify="left", style="white")
        table.add_column("Finished", justify="center")
        table.add_column("Progress", justify="center", style="yellow")

        for name, experiment in self.meta.items():
            sign, data, model = experiment.signature, experiment.data_name, experiment.model_name
            finished = Text('No', style="red") if experiment.is_alive() else Text('Yes', style="green")
            total_epoches = experiment.epoch or 'N/A'
            current_epoch = experiment.current_epoch
            patience = experiment.patience or 'N/A'
            input_type = experiment.input_type
            progress_string = f'{current_epoch} of {total_epoches} (P: {patience})'
            table.add_row(name, sign, experiment.data_full, model, input_type, finished, progress_string)

        console.print(introduction_header)
        console.print(table)

    def help(self):
        help_lines = [
            "Observer Actions",
            "   push(ob, sign, name) add experiment <sign> (alias <name>) into <ob> observer",
            "   pop(ob, sign/name) remove experiment <sign> (or alias <name>) from <ob> observer",
            "   summarize"
        ]

    def get_visualizer_config(self):
        visualizer_config = {}
        full_config = self.config()
        for k in full_config:
            if k.startswith('v.'):
                visualizer_config[k[2:]] = full_config[k]
        return visualizer_config

    def export(self):
        charts = self.config.charts
        allow_charts = ['dataset', 'loss.curve', 'eval.curve', 'loss.curve.time', 'eval.curve.time', 'test.bar', 'test.table']

        charts = charts or ''
        if charts == 'all':
            charts = '+'.join(allow_charts)
        charts = list(filter(lambda x: x.strip(), charts.split('+')))

        if not charts:
            raise ValueError('Please use `--charts` option to define charts.')

        for chart in charts:
            if chart not in allow_charts:
                raise ValueError(f'Chart {chart} is not allowed. Allowed charts: {allow_charts}')

        visualizer = self.get_visualizer()
        visualizer_config = self.get_visualizer_config()
        pnt('visualizer config:', visualizer_config)
        visualizer.init_config(**visualizer_config)

        if 'dataset' in charts:
            visualizer.generate_dataset_statistics()
        if 'loss.curve' in charts:
            visualizer.generate_loss_curve_by_epoch()
        if 'loss.curve.time' in charts:
            visualizer.generate_loss_curve_by_time()
        if 'eval.curve' in charts:
            visualizer.generate_evaluation_curve()
        if 'eval.curve.time' in charts:
            visualizer.generate_evaluation_curve_by_time()
        if 'test.bar' in charts:
            metrics = self.config.bar_metrics.split('+')
            visualizer.generate_test_bar(metrics=metrics)
        if 'test.table' in charts:
            metrics = self.config.table_metrics.split('+')
            visualizer.generate_test_table(metrics=metrics)

        visualizer.postact()

    def interact(self):
        pass

    def list(self):
        if self.ob:
            self.summarize()
            return

        obs = self.load_observers()
        pnt(f'Detecting {len(obs)} observers:')

        if self.config.sort:
            ascending = self.config.sort.startswith('-')
            if ascending:
                self.config.sort = self.config.sort[1:]
            if self.config.sort == Observer.SORT_CREATE_TIME:
                obs.sort(key=lambda x: x.ft.create_time, reverse=ascending)
            elif self.config.sort == Observer.SORT_NAME:
                obs.sort(key=lambda x: x.name, reverse=ascending)

        console = Console()
        introduction_header = Text.assemble(
            (
                f"Observer List (v{self.VERSION}), "
                f"Meta directory: {self.OBSERVER_BASE_DIR}\n",
                "bold cyan"),
            (f"Registered Observers: {len(obs)}\n", "green"),
            style="dim"
        )

        table = Table(title="Observers", expand=True, title_style="bold yellow", show_lines=True)

        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column('# Experiments', justify="center", style="cyan", no_wrap=True)
        table.add_column("Interval", justify="left", style="magenta")
        table.add_column("Create", justify="center", style="green")

        for ob in obs:  # type: Ob
            table.add_row(
                ob.name,
                str(ob.num_experiments),
                ob.ft.readable_interval,
                ob.ft.readable_create_time,
            )

        console.print(introduction_header)
        console.print(table)

    def rename(self):
        name = self.config.name

        if name:
            self.meta[self.config.to] = self.meta.pop(name)
        else:
            self.delete()
            self.ob = self.config.to
        self.save_meta()

    def delete(self):
        if self.config.name:
            raise AttributeError(f'please use `pop` action to delete experiment or unset `name` argument to delete observer.')
        os.remove(self.meta_path)

    def run(self):
        """
        根据 action 执行 push/pop 等操作，并展示摘要。
        """
        if self.action == Syb.observer_list.name:
            self.list()
            return

        assert self.ob, 'observer name should be provided using ``--ob <ob>``'

        if self.action == Syb.observer_interact.name:
            self.interact()
        if self.action == Syb.observer_push.name:
            self.push()
            self.summarize()
        if self.action == Syb.observer_pop.name:
            self.pop()
            self.summarize()
        if self.action == Syb.observer_rename.name:
            self.rename()
            self.summarize()
        elif self.action == Syb.observer_delete.name:
            self.delete()
        if self.action == Syb.observer_export.name:
            self.export()


if __name__ == "__main__":
    configurations = CommandInit(
        required_args=(),  # 指定 observer 名称
        default_args=dict(
            ob=None,
            sign=None,
            name=None,
            action='list',
            i=True,  # interactive mode
            v='latex',
            charts=None,
            bar_metrics='NDCG',
            table_metrics='GAUC+NDCG+MRR',
            sort=Observer.SORT_NAME,  # sort by name
        )
    ).parse()

    observer = Observer(configurations)
    observer.run()
