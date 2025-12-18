import abc
import os
from typing import Dict, List

from pigmento import pnt

from utils import io
from visualizers.experiment import Experiment


class BackboneVisualizer(abc.ABC):
    BASE_EXPORT_DIR = 'exports'
    BASE_TEMPLATE_DIR = os.path.join('visualizers', 'templates')

    def __init__(self, experiments: List[Experiment], ob: str):
        self.experiments = experiments
        self.ob = ob

        os.makedirs(self.export_dir, exist_ok=True)

    def init_config(self, **kwargs):
        raise NotImplementedError

    @property
    def export_dir(self):
        return os.path.join(self.BASE_EXPORT_DIR, self.ob, self.classname)

    @property
    def classname(self):
        return self.__class__.__name__.replace('Visualizer', '')

    def export_save(self, filename, content):
        filepath = os.path.join(self.export_dir, filename)
        io.file_save(filepath, content)
        pnt(f'Chart has been saved to {filepath} ...')

    def template_load(self, filename):
        filepath = os.path.join(self.BASE_TEMPLATE_DIR, filename)
        return io.file_load(filepath)

    def generate_dataset_statistics(self):
        raise NotImplementedError

    def generate_loss_curve_by_epoch(self):
        raise NotImplementedError

    def generate_loss_curve_by_time(self):
        raise NotImplementedError

    def generate_evaluation_curve(self):
        raise NotImplementedError

    def generate_evaluation_curve_by_time(self):
        raise NotImplementedError

    def generate_test_bar(self, metrics: List[str]):
        raise NotImplementedError

    def generate_test_table(self, metrics: List[str]):
        raise NotImplementedError

    def postact(self):
        raise NotImplementedError
