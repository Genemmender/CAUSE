import os
from typing import Type, Dict

from pigmento import pnt
from torch.utils.data import DataLoader
from unitok import UniTok, Symbol, Job

from loaders.class_hub import ClassHub
from loaders.column_map import ColumnMap
from loaders.datasets.backbone import BackboneDataset
from loaders.embedding_hub import EmbeddingHub
from loaders.embedding_layer import EmbeddingLayer
from loaders.env import Env
from loaders.symbols import Symbols as Syb
from models.backbone import BackboneModel
from models.classifier import VanillaClassifier, SamplingClassifier
from models.genemmender import Genemmender
from processors.backbone import BackboneProcessor
from utils import io


class Manager:
    def __init__(self, processor, model, exp, mode):
        self.processor: BackboneProcessor = processor
        self.model = model  # 对应config/model/xxx.yaml
        self.exp = exp  # 对应config/exp/default.yaml
        self.mode = mode

        self.cm: ColumnMap = self.processor.build_column_map()
        self.seq_type = self.model.config.seq_type.lower()

        self.model_class: Type[BackboneModel] = ClassHub.models()(self.model.name)
        self.dataset_class: Type[BackboneDataset] = ClassHub.datasets()(self.seq_type)

        self.uts = self.get_uts()
        self.item_ut = UniTok.load(self.processor.item_dir)

        self.selector = self.get_item_selector()
        self.cm.load_ut(self.uts[Syb.train], self.item_ut)

        self.datasets = self.get_datasets()  # type: Dict[Symbol, BackboneDataset]
        self.model_max_length = self.datasets[Syb.test].get_max_length()
        pnt(f'Model Max Sequence Length: {self.model_max_length}')

        backbone_config = self.model_class.config_class(**self.model.config(), max_length=self.model_max_length)
        self.backbone: BackboneModel = self.model_class(backbone_config)

        # 给user, item vocab分配 embedding table
        self.eh = EmbeddingHub(self.exp.policy.embedding_dim, cm=self.cm)
        self.eh.register(self.cm.user_col)
        self.eh.register(self.cm.items_col)
        self.eh.register(self.cm.actions_col)
        self.eh.add_table(BackboneDataset.SEG_VOCAB.name, BackboneDataset.SEG_VOCAB.size)
        self.register_item_features()

        if self.cm.history_group_col:
            self.eh.register(self.cm.history_group_col, item=True)

        self.el = EmbeddingLayer(
            eh=self.eh,
            decay_rate=self.exp.policy.decay,
            fusion=self.exp.policy.fusion,
        )

        Env.global_validation = self.exp.validation.global_validation
        if not Env.global_validation:
            assert not self.exp.validation.mask_seen, 'global_validation must be True when mask_seen is True'

        self.action_classifier = VanillaClassifier(
            num_classes=self.cm.get_size(self.cm.actions_col),
            embedding_dim=self.eh.embedding_dim
        )

        self.item_classifier = SamplingClassifier(
            neg_num=self.exp.policy.sampling.neg_num or 10,
            temperature=self.exp.policy.sampling.temperature or 1.0,
            neg_pool=self.get_item_neg_pools(),
            embedding=self.el[self.cm.items_col],
        )
        io.pkl_save(self.item_classifier.neg_pool.cpu().tolist(), Env.ph.neg_pool_path)

        self.genemmender = Genemmender(
            el=self.el,
            backbone=self.backbone,
            action_classifier=self.action_classifier,
            item_classifier=self.item_classifier,
            use_mask_seen=self.exp.validation.mask_seen,
            use_segment_ids=self.exp.policy.use_segment_ids,
            use_separators=self.exp.policy.use_separators,
        )

    def get_item_selector(self):
        use_item_content = self.model.config.use_item_content
        if use_item_content is False:
            return False
        if use_item_content is True:
            return None
        return tuple(use_item_content.split('+'))

    def register_item_features(self):
        use_item_content = self.model.config.use_item_content
        if use_item_content is False:
            return

        if use_item_content is True:
            for job in self.item_ut.meta.jobs:  # type: Job
                if job.key:
                    continue
                self.eh.register(job.name, item=True)
            return

        jobs = use_item_content.split('+')
        for j in jobs:
            job = self.item_ut.meta.jobs[j]
            if job.key:
                continue
            self.eh.register(job.name, item=True)

    def get_model_max_length(self):
        length_ratio = 2 if self.seq_type == 'flatten' else 1
        return self.cm.get_max_length(self.cm.items_col) * length_ratio + 1

    def get_item_neg_pools(self):
        if self.exp.policy.disable_neg_pool:
            return None

        item_pool = set()
        item_count = 0
        for sample in self.uts[Syb.train]:
            item_pool.update(set(sample[self.cm.items_col]))
            item_count += len(sample[self.cm.items_col])
        pnt(f'Total items: {item_count}')
        pnt(f'Total sequences: {len(self.uts[Syb.train])}')
        pnt(f'Individual items: {len(item_pool)}')

        if self.exp.load.sign:
            path = os.path.join(Env.ph.checkpoint_base_dir, f'{self.exp.load.sign}.pool')
            if os.path.exists(path):
                pnt(f'Integrate Past Items from {path}')
                past_pool = io.pkl_load(path)  # type: list
                item_pool.update(set(past_pool))
                pnt(f'Individual items Update to: {len(item_pool)}')

        return item_pool

    def get_uts(self):
        return {
            Syb.train: UniTok.load(self.processor.train_dir),
            Syb.val: UniTok.load(self.processor.val_dir),
            Syb.test: UniTok.load(self.processor.test_dir),
        }

    def get_datasets(self):
        datasets = {}
        modes = [Syb.train, Syb.val, Syb.test] if 'train' in self.mode else [Syb.test]
        for mode in modes:
            datasets[mode] = self.dataset_class(
                self.uts[mode],
                mode=mode,
                cm=self.cm,
                item_ut=self.item_ut,
                selector=self.selector,
                history_config=self.model.config.history,
                use_separators=self.exp.policy.use_separators,
            )
        return datasets

    def get_dataloader(self, mode):
        return DataLoader(
            dataset=self.datasets[mode],
            batch_size=self.exp.policy.batch_size,
            shuffle=mode is Syb.train
        )
