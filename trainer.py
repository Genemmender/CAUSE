import multiprocessing
import os
import sys
from typing import Optional, Type

import pigmento
import torch
import unitok
from pigmento import pnt
from unitok import Symbol

from loaders.class_hub import ClassHub
from loaders.manager import Manager
from loaders.env import Env
from loaders.symbols import Symbols as Syb
from models.genemmender import GenemmenderOutput, Genemmender
from processor import MetaProcessor
from processors.backbone import BackboneProcessor
from utils import bars, io
from utils.config_init import CommandInit
from utils import function
from utils.fast_metrics import MetricPool
from utils.gpu import GPU
from utils.meaner import Meaner
from utils.monitor import Monitor
from utils.path_hub import PathHub
from utils.timer import Timer, StatusTimer


class Trainer:
    def __init__(self, config):
        self.config = config
        self.data = self.config.data
        self.exp = self.config.exp
        self.model = self.config.model

        multiprocessing.set_start_method('fork')
        self.init_pigmento()
        self.init_unitok()

        self.processor = self.get_processor()
        self.data_sign = f'{self.processor.get_classname()}.{self.processor.signature}'

        self.mode = self.config.mode

        config.seed = int(config.seed or 2025)
        function.seeding(config.seed)

        Env.debug = self.config.debug
        Env.device = self.get_device()
        self.signature = function.get_exp_signature(self.data_sign, self.model(), self.exp())
        Env.ph = PathHub(
            data_name=self.processor.get_classname(),
            model_name=self.model.name,
            signature=self.signature,
        )

        pigmento.add_log_plugin(Env.ph.log_path)

        pnt('python', ' '.join(sys.argv))
        pnt(f'Exp signature: @{Env.ph.signature}')
        pnt(f'Exp config: {io.json_dumps(self.config())}')
        io.json_save(self.config(), Env.ph.cfg_path)

        self.manager = Manager(
            processor=self.processor,
            model=self.config.model,
            exp=self.config.exp,
            mode=self.config.mode
        )
        self.genemmender: Genemmender = self.manager.genemmender.to(Env.device)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.timer = Timer()

    def get_processor(self) -> BackboneProcessor:
        assert '.' in self.data, f'data should be in <dataset>.<name> format, but got {self.data}'
        dataset, alias = self.data.split('.', maxsplit=1)
        data_dir = MetaProcessor.get_data_dir(dataset)

        processors = ClassHub.processors()
        processor_class: Type[BackboneProcessor] = processors[dataset]
        return processor_class(name=alias, config=None, data_dir=data_dir)

    def get_device(self):
        cuda = self.config.cuda
        if cuda in ['-1', -1] or cuda is False:
            pnt('choose cpu')
            return 'cpu'
        if isinstance(cuda, int) or isinstance(cuda, str):
            pnt(f'User select cuda {cuda}')
            # return f"cuda:{cuda}"
            cuda = eval(f'[{cuda}]') if isinstance(cuda, str) else cuda
            # return torch.cuda.device(cuda)
            return f'cuda:{cuda}'
        return GPU.auto_choose(torch_format=True)

    @staticmethod
    def init_pigmento():
        pigmento.add_time_prefix()
        pigmento.add_dynamic_color_plugin()
        pnt.set_display_mode(
            display_method_name=False,
            display_class_name=True,
            use_instance_class=True,
        )

    @staticmethod
    def init_unitok():
        unitok.Verbose.set_level(unitok.Verbose.ERROR)

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.genemmender.parameters()),
            lr=self.exp.policy.lr
        )

        for name, p in self.genemmender.named_parameters():  # type: str, torch.Tensor
            if p.requires_grad:
                pnt(name, p.data.shape)

    @staticmethod
    def log_interval(epoch, step, loss):
        io.file_save(Env.ph.log_path, f'[epoch {epoch}] step {step}, loss {loss:.4f}\n', append=True)
        jsonl = io.json_dumps(dict(epoch=epoch, step=step, loss=loss, type=Syb.interval.name), indent=None) + '\n'
        io.file_save(Env.ph.progress_path, jsonl, append=True)

    @staticmethod
    def log_epoch(epoch, results: dict, data: Symbol):
        line = ', '.join([f'{metric} {results[metric]:.4f}' for metric in results])
        pnt(f'[epoch {epoch}] ({data.name}) {line}')
        data = dict(epoch=epoch, data=data.name, type=Syb.epoch.name)
        data.update(results)
        jsonl = io.json_dumps(data, indent=None) + '\n'
        io.file_save(Env.ph.progress_path, jsonl, append=True)

    def load(self, sign=None):
        if not sign:
            sign = self.exp.load.sign
        if not sign:
            return
        # sign = sign.replace('@', '')

        ckpt_path = os.path.join(Env.ph.checkpoint_base_dir, f'{sign}.pt')
        state_dict = torch.load(ckpt_path, map_location=Env.device)
        # compatible to old version where each operator are wrapped with an encoder
        self.genemmender.load_state_dict(state_dict['model'], strict=self.exp.load.strict)
        if not self.exp.load.model_only:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        pnt(f"load model from {ckpt_path}")

    def save(self):
        state_dict = dict(
            model=self.genemmender.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        torch.save(state_dict, Env.ph.ckpt_path)
        pnt(f'save model to {Env.ph.ckpt_path}')

    def train(self):
        train_steps = len(self.manager.datasets[Syb.train]) // self.exp.policy.batch_size

        check_interval = self.exp.policy.check_interval
        if check_interval and check_interval < 0:
            check_interval = max(train_steps // (-check_interval), 1)

        eval_interval = self.exp.policy.eval_interval or 1
        patience = self.exp.save.patience and self.exp.save.patience * eval_interval
        monitor = Monitor(
            patience=patience,
            epoch=self.exp.save.epoch,
            minimize=MetricPool.is_minimize(self.exp.save.metric)
        )

        meaner = Meaner()
        self.optimizer.zero_grad()

        accumulate_step = 0
        accumulate_batch = self.exp.policy.acc_batch or 1

        aw = self.exp.policy.action_weight
        iw = self.exp.policy.item_weight
        epoch = 0

        while True:
            epoch += 1
            loader = self.manager.get_dataloader(Syb.train)
            Env.start_training()

            # print(self.genemmender.item_classifier.embedding.weight[28878].sum())
            for step, batch in enumerate(bar := bars.TrainBar(epoch=epoch)(loader)):
                self.optimizer.zero_grad()
                self.timer('train')
                go: GenemmenderOutput = self.genemmender(batch=batch)
                loss = go.get_loss(aw, iw)
                loss.backward()
                self.timer('train')

                accumulate_step += 1
                if accumulate_step == accumulate_batch:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulate_step = 0

                bar.set_postfix_str(f'loss: {meaner(loss.item()):.4f}')

                if check_interval and (step + 1) % check_interval == 0:
                    self.log_interval(epoch, step, meaner.mean)

            if epoch % eval_interval:
                continue

            val_bar = bars.ValBar(epoch=epoch, train_loss=meaner.mean)
            val_results, monitor_metric = self.validate(val_bar)
            self.log_epoch(epoch, val_results, data=Syb.val)

            test_bar = bars.ValBar(epoch=epoch, train_loss=meaner.mean)
            test_results, _ = self.validate(test_bar, loader=self.manager.get_dataloader(Syb.test))
            self.log_epoch(epoch, test_results, data=Syb.test)

            action = monitor.push(monitor_metric, epoch)
            if action is Syb.stop:
                pnt('Early stop')
                break
            elif action is Syb.best:
                self.save()

    def evaluate(self, loader, metrics, bar: bars.Bar):
        pool = MetricPool.parse(metrics)
        aw, iw = self.exp.policy.action_weight, self.exp.policy.item_weight

        self.genemmender.eval()
        Env.start_evaluating()
        meaner = Meaner()

        # flags_series, labels_series, logits_series = [], [], []
        flags_series, ranks_series = [], []
        for step, batch in enumerate(bar := bar(loader)):
            with torch.no_grad():
                go: GenemmenderOutput = self.genemmender(batch=batch)

                # labels, logits, flags = go.prepare_evaluation()
                ranks, flags = go.prepare_evaluation()

                if flags is None:
                    flags_series.extend([0] * len(ranks))
                else:
                    flags_series.extend(flags)
                ranks_series.extend(ranks)
                loss = go.get_loss(aw, iw)

            bar.set_postfix_str(f'score: {meaner(loss.item()):.4f}')

        # flag_to_indices = defaultdict(list)
        # for idx, flag in enumerate(flags_series):
        #     flag_to_indices[flag].append(idx)
        #
        # for flag, indices in flag_to_indices.items():
        #     print(f'Flag: {flag}, Size: {len(indices)}')
        #     sub_labels = [labels_series[i] for i in indices]
        #     sub_logits = [logits_series[i] for i in indices]
        #
        #     results = pool.calculate(sub_labels, sub_logits)
        #     print(f'Results: {results}')

        return pool.calculate(ranks_series)

    def validate(self, bar: bars.Bar, loader=None):
        loader = loader or self.manager.get_dataloader(Syb.val)
        results = self.evaluate(loader, metrics=[self.exp.save.metric], bar=bar)
        return results, results[self.exp.save.metric]

    def test(self):
        loader = self.manager.get_dataloader(Syb.test)
        self.genemmender.eval()

        results = self.evaluate(loader, metrics=self.exp.metrics, bar=bars.TestBar())

        lines = []
        for metric in results:
            pnt(f'{metric}: {results[metric]:.4f}')
            lines.append(f'{metric},{results[metric]:.4f}')
        io.file_save(Env.ph.result_path, '\n'.join(lines))

        return results

    def run(self):
        self.init_optimizer()
        self.load()

        if self.mode == 'train':
            self.train()
            self.load(Env.ph.signature)

        if self.mode == 'train_latency':
            self.timer.clear()
            self.timer.activate()
            try:
                self.train()
            except KeyboardInterrupt:
                pass
            finally:
                st: StatusTimer = self.timer.status_dict['train']
                print(st.count)
                print(st.avgms())
                pnt(f'Total {st.count} steps, avg ms {st.avgms():.4f}')

        if self.mode == 'test':
            self.test()


if __name__ == '__main__':
    configurations = CommandInit(
        required_args=('data',),
        default_args=dict(
            model='config/model/trm.yaml',
            exp='config/exp/default.yaml',
            cuda=None,
            load=None,
            mode='train',
            debug=False,
        )
    ).parse()
    torch.cuda.empty_cache()
    trainer = Trainer(configurations)
    trainer.run()
