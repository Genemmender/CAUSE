from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import torch
from pigmento import pnt
from sklearn.cluster import KMeans, Birch, DBSCAN
from unitok import UniTok, EntityTokenizer

from loaders.class_hub import ClassHub
from processor import MetaProcessor
from processors.backbone import BackboneProcessor
from utils import io
from utils.config_init import CommandInit
from utils.path_hub import PathHub
from visualizers.experiment import Experiment


class Clusterer:
    def __init__(self, config):
        self.config = config
        self.sign = config.sign
        self.algo = config.algo
        self.name = 'cluster@' + config.name

        experiments = self.collect_experiments()
        if self.sign not in experiments:
            raise ValueError(f'Experiment with signature {self.sign} not found.')

        self.data, self.model = experiments[self.sign]
        self.experiment = Experiment(
            data_name=self.data,
            model_name=self.model,
            signature=self.sign,
            name=None,
        )

        self.data_full, _, _, _ = self.experiment.get_exp_config()
        self.processor = self.get_processor()

        self.used_items = io.pkl_load(self.experiment.neg_pool_path)

    def get_processor(self) -> BackboneProcessor:
        assert '.' in self.data_full, f'data should be in <dataset>.<name> format, but got {self.data_full}'
        dataset, alias = self.data_full.split('.', maxsplit=1)
        data_dir = MetaProcessor.get_data_dir(dataset)

        processors = ClassHub.processors()
        processor_class: Type[BackboneProcessor] = processors[dataset]
        return processor_class(name=alias, config=None, data_dir=data_dir)

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

    def load_item_embeddings(self):
        state_dict = torch.load(self.experiment.ckpt_path, map_location=torch.device('cpu'))
        return state_dict['model']['el.table.@items.weight'].numpy()

    def pca(self, embeddings):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.config.pca)
        pca.fit(embeddings)
        embeddings = pca.transform(embeddings)
        return embeddings

    def norm(self, embeddings):
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms

    def kmeans(self, embeddings):
        if not self.config.k:
            raise ValueError('kmeans needs config.k')
        kmeans = KMeans(n_clusters=self.config.k, random_state=42)
        kmeans.fit(embeddings)
        return kmeans.labels_

    def birch(self, embeddings):
        """
        Birch 默认欧氏。想做球面近似可先 L2 归一化（严格余弦需自定义度量，sklearn 未直接支持）。
        """
        if not self.config.k:
            raise ValueError('Birch needs config.k')
        brc = Birch(n_clusters=self.config.k)
        return brc.fit_predict(embeddings)

    def dbscan(self, embeddings):
        params = {}
        if self.config.eps:
            params['eps'] = self.config.eps
        if self.config.min_samples:
            params['min_samples'] = self.config.min_samples
        pnt(f'DBSCAN params: {params}')
        db = DBSCAN(**params)
        return db.fit_predict(embeddings)

    def run(self):
        ut = UniTok.load(self.processor.item_dir)
        if self.name in ut.meta.features:
            raise ValueError(
                f'Feature {self.name} already exists in {self.processor.item_dir}, please choose a different name.')

        all_embeddings = self.load_item_embeddings()
        pnt(f'Loaded {all_embeddings.shape[0]} item embeddings with shape {all_embeddings.shape[1:]}')

        pnt(f'Used Items: {len(self.used_items)}')
        embeddings = all_embeddings[self.used_items]

        if self.config.pca:
            embeddings = self.pca(embeddings)
            pnt(f'PCA of {embeddings.shape[0]} item embeddings with shape {embeddings.shape[1:]}')

        if self.config.norm:
            embeddings = self.norm(embeddings)
            pnt(f'Normalized item embeddings')

        if self.algo == 'kmeans':
            labels = self.kmeans(embeddings)
        elif self.algo == 'birch':
            labels = self.birch(embeddings)
        elif self.algo == 'dbscan':
            labels = self.dbscan(embeddings)
        else:
            raise ValueError(f'Unknown algorithm: {self.algo}, expect kmeans or birch.')
        pnt(f'Kmeans labels: {min(labels)} to {max(labels)}')

        assert min(labels) == 0, f'Labels should start from 0, but got min label {min(labels)}'

        all_labels = np.ones(len(all_embeddings), dtype=np.int32) * -1
        all_labels[self.used_items] = labels
        all_labels = all_labels.tolist()

        df = pd.DataFrame(data={
            self.name: all_labels,
        })

        with ut:
            ut.add_feature(tokenizer=EntityTokenizer(vocab=self.name))
            ut.tokenize(df).save(self.processor.item_dir)

        pnt(f'Saved to {self.processor.item_dir}')


if __name__ == '__main__':
    configurations = CommandInit(
        required_args=('sign', 'name'),  # 指定 observer 名称
        default_args=dict(
            pca=64,
            k=None,
            algo='kmeans',
            norm=False,
        )
    ).parse()

    clusterer = Clusterer(configurations)
    clusterer.run()
