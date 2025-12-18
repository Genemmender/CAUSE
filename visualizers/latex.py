import os.path
import zipfile
from typing import List, cast

import requests
from pigmento import pnt

from visualizers.backbone import BackboneVisualizer
from visualizers.experiment import Experiment


class LaTeXVisualizer(BackboneVisualizer):
    MAX_POINTS_PER_CURVE = 200

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_points = self.MAX_POINTS_PER_CURVE
        self.avg_window = 0

        self.latex_ob = self.ob.replace('_', '\\_')

        self.latex_condition_template = self.template_load('latex_condition.template')
        self.latex_condition_name_template = self.template_load('latex_condition_name.template')
        self.latex_curve_experiment_template = self.template_load('latex_curve_experiment.template')
        self.latex_curve_wrapper_template = self.template_load('latex_curve_wrapper.template')
        self.latex_bar_experiment_template = self.template_load('latex_bar_experiment.template')
        self.latex_bar_wrapper_template = self.template_load('latex_bar_wrapper.template')
        self.latex_table_experiment_template = self.template_load('latex_table_experiment.template')
        self.latex_table_wrapper_template = self.template_load('latex_table_wrapper.template')
        self.latex_table_statistics_template = self.template_load('latex_table_statistics.template')

        self.export_save('acmart.cls', self.template_load( 'latex_acmart.cls'))

        self.main_tex_filename = 'main.tex'
        self.main_tex_filepath = os.path.join(self.export_dir, self.main_tex_filename)
        self.max_tex_code = self.template_load('latex_main.template')
        # if os.path.exists(self.main_tex_filepath):
        #     self.max_tex_code = io.file_load(self.main_tex_filepath)
        self.max_tex_code = self.replace(self.max_tex_code, ob=self.latex_ob)

        self.colors = [
            'brightblue',
            'brightorange',
            'brightgreen',
            'brightred',
            'brightpurple',
            'brightteal',
            'brightyellow',
            'brightpink',
            'brightbrown',
            'brightcyan',
            'brightlime',
            'brightcrimson',
            'brightaqua',
            'brightolive',
            'brightlavender',
            'brightturquoise',
        ]
        self.marks = ['none', '*', 'o', 'x']
        self.plot_anchor = '% <plot>'

        self.display_triggers = set()
        self.display_charts = []

    def init_config(
            self,
            max_points=None,
            avg_window=None
    ):
        if max_points is not None:
            self.max_points = max_points
            pnt(f'max_points: {self.max_points}')
        if avg_window is not None:
            self.avg_window = avg_window
            pnt(f'avg_window: {self.avg_window}')

    @classmethod
    def replace(cls, text, **kwargs):
        if 'ob' in kwargs:
            kwargs['label_ob'] = kwargs['ob'].replace('\\_', '-').replace(' ', '-').lower()
        if 'title' in kwargs:
            kwargs['label_title'] = kwargs['title'].replace('\\_', '-').replace(' ', '-').lower()

        for k, v in kwargs.items():
            text = text.replace(f'<{k}>', str(v))
        return text

    def update_main(self, filename):
        # if filename not in self.max_tex_code:
        #     self.max_tex_code = self.max_tex_code.replace(self.plot_anchor, f'\input{{{filename}}}\n' + self.plot_anchor)
        # self.export_save(self.main_tex_filename, self.max_tex_code)
        self.display_charts.append(filename)

    def display_trigger_name(self, name):
        name = name.replace('_', '').replace('-', '').replace('.', '').lower()
        name = 'display' + name[0].upper() + name[1:]
        if name not in self.display_triggers:
            self.display_triggers.add(name)
        return name

    def _get_average_value(self, datalist, index):
        real_count = 1
        value = datalist[index][1]
        for i in range(self.avg_window):
            if index - i >= 0:
                real_count += 1
                value += datalist[index - i][1]
            if index + i < len(datalist):
                real_count += 1
                value += datalist[index + i][1]
        return value / real_count

    def _generate_loss_curve_code(self, point_dict, wrapper_replace_dict):
        experiment_latex_code = []
        experiment_index = 0
        for name, data in point_dict.items():
            color_id = experiment_index % len(self.colors)
            mark_id = experiment_index // len(self.colors)

            coordinates = []
            num_points = min(len(data), self.max_points)
            flags = [False] * len(data)

            if data:
                interval = len(data) * 1.0 / num_points
                for idx in range(num_points):
                    index = int(idx * interval)
                    flags[index] = True

            for index, (key, loss) in enumerate(data):
                if flags[index]:
                    coordinates.append(f'    ({key}, {loss})')

            experiment_replace_dict = dict(
                legend=name,
                color=self.colors[color_id],
                mark=self.marks[mark_id],
                coordinates='\n'.join(coordinates),
                best_coordinate='',
                type='loss',
            )
            core_latex_code = self.replace(self.latex_curve_experiment_template, **experiment_replace_dict)
            condition_replace_dict = dict(
                condition=core_latex_code,
                name=self.display_trigger_name(name),
            )
            condition_code = self.replace(self.latex_condition_template, **condition_replace_dict)
            experiment_latex_code.append(condition_code)
            experiment_index += 1

        wrapper_replace_dict['experiments'] = '\n'.join(experiment_latex_code)
        code = self.replace(self.latex_curve_wrapper_template, **wrapper_replace_dict)

        return code

    def generate_loss_curve_by_epoch(self):
        point_dict = dict()
        for experiment in self.experiments:
            point_dict[experiment.name] = experiment.get_loss_curve_by_epoch()
        if len(self.experiments) > len(self.colors):
            raise ValueError(f'The number of experiments ({len(self.experiments)}) is greater than the number of colors ({len(self.colors)})')

        wrapper_replace_dict = dict(
            star='*',
            width='20cm',
            height='10cm',
            xlabel='Epoch',
            ylabel='Loss',
            position='north east',
            ob=self.latex_ob,
            title='Training loss curve',
            num=len(point_dict),
        )

        code = self._generate_loss_curve_code(point_dict, wrapper_replace_dict)
        filename = 'training_loss_curve.tex'

        self.export_save(filename, code)
        self.update_main(filename)

    def generate_loss_curve_by_time(self):
        point_dict = dict()
        for experiment in self.experiments:  # type: Experiment
            point_dict[experiment.name] = experiment.get_loss_curve_by_time()
        if len(self.experiments) > len(self.colors):
            raise ValueError(f'The number of experiments ({len(self.experiments)}) is greater than the number of colors ({len(self.colors)})')

        wrapper_replace_dict = dict(
            star='*',
            width='20cm',
            height='10cm',
            xlabel='Time (seconds)',
            ylabel='Loss',
            position='north east',
            ob=self.latex_ob,
            title='Training loss curve',
            num=len(point_dict),
        )

        code = self._generate_loss_curve_code(point_dict, wrapper_replace_dict)
        filename = 'training_loss_curve_by_time.tex'

        self.export_save(filename, code)
        self.update_main(filename)

    def get_metric(self, d: dict):
        known_keys = ['data', 'epoch', 'type']

        for key in d.keys():
            if key not in known_keys:
                return key

        raise ValueError(f'No metric found in the experiment {self.experiments[0].name}')

    def _generate_evaluation_curve_code(self, point_dict, wrapper_replace_dict):
        mark_dict = dict(validate=1, test=2)

        experiment_latex_code = []
        experiment_index = 0
        for name, subdata in point_dict.items():
            color_id = experiment_index % len(self.colors)

            for eval_type, data in subdata.items():
                assert eval_type in mark_dict, f'Invalid evaluation type: {eval_type}'
                mark_id = mark_dict[eval_type]

                coordinates = []
                best_coordinate = ''
                num_points = min(len(data), self.max_points)
                flags = [False] * len(data)

                if data:
                    interval = len(data) * 1.0 / num_points
                    for idx in range(num_points):
                        index = int(idx * interval)
                        flags[index] = True

                    index = 0
                    for idx, (epoch, value) in enumerate(data):
                        if value > data[index][1]:
                            index = idx
                    flags[index] = True
                    best_coordinate = f'({data[index][0]}, {data[index][1]})'

                for index, (epoch, _) in enumerate(data):
                    if flags[index]:
                        value = self._get_average_value(data, index)
                        coordinates.append(f'    ({epoch}, {value})')

                experiment_replace_dict = dict(
                    legend=f'{name} ({eval_type})',
                    color=self.colors[color_id],
                    mark=self.marks[mark_id],
                    coordinates='\n'.join(coordinates),
                    best_coordinate=best_coordinate,
                    type='eval',
                )
                core_latex_code = self.replace(self.latex_curve_experiment_template, **experiment_replace_dict)
                inner_condition_replace_dict = dict(
                    condition=core_latex_code,
                    name=self.display_trigger_name(name),
                )
                inner_condition_code = self.replace(self.latex_condition_template, **inner_condition_replace_dict)
                outer_condition_replace_dict = dict(
                    condition=inner_condition_code,
                    name=self.display_trigger_name(eval_type),
                )
                outer_condition_code = self.replace(self.latex_condition_template, **outer_condition_replace_dict)
                experiment_latex_code.append(outer_condition_code)

            experiment_index += 1

        code = self.replace(
            self.latex_curve_wrapper_template,
            experiments='\n'.join(experiment_latex_code),
            **wrapper_replace_dict
        )
        return code

    def generate_evaluation_curve(self):
        point_dict = dict()
        metric = None

        for experiment in self.experiments:  # type: Experiment
            metric, points = experiment.get_evaluation_curve_by_epoch()
            point_dict[experiment.name] = points
        if len(self.experiments) > len(self.colors):
            raise ValueError(f'The number of experiments ({len(self.experiments)}) is greater than the number of colors ({len(self.colors)})')

        wrapper_replace_dict = dict(
            star='*',
            width='20cm',
            height='10cm',
            xlabel='Epoch',
            ylabel=metric,
            position='north west',
            ob=self.latex_ob,
            title='Evaluation curve',
            num=len(point_dict),
        )

        code = self._generate_evaluation_curve_code(point_dict, wrapper_replace_dict)
        filename = 'evaluation_curve.tex'

        self.export_save(filename, code)
        self.update_main(filename)

    def generate_evaluation_curve_by_time(self):
        point_dict = dict()
        metric = None

        for experiment in self.experiments:  # type: Experiment
            metric, points = experiment.get_evaluation_curve_by_time()
            point_dict[experiment.name] = points
        if len(self.experiments) > len(self.colors):
            raise ValueError(f'The number of experiments ({len(self.experiments)}) is greater than the number of colors ({len(self.colors)})')

        wrapper_replace_dict = dict(
            star='*',
            width='20cm',
            height='10cm',
            xlabel='Time (seconds)',
            ylabel=metric,
            position='north west',
            ob=self.latex_ob,
            title='Evaluation curve',
            num=len(point_dict),
        )

        code = self._generate_evaluation_curve_code(point_dict, wrapper_replace_dict)
        filename = 'evaluation_curve_by_time.tex'

        self.export_save(filename, code)
        self.update_main(filename)

    def generate_test_bar(self, metrics: List[str]):
        point_dict = dict()
        metric_list = []
        for experiment in self.experiments:  # type: Experiment
            test_results = experiment.get_test_results(metrics)
            if test_results:
                point_dict[experiment.name] = test_results
                metric_list = list(test_results.keys())

        if len(self.experiments) > len(self.colors):
            raise ValueError(f'The number of experiments ({len(self.experiments)}) is greater than the number of colors ({len(self.colors)})')

        wrapper_replace_dict = dict(
            star='*',
            width='20cm',
            height='10cm',
            position='north west',
            ob=self.latex_ob,
            title='Performance comparison bar',
            num=len(point_dict),
            metrics=','.join(metric_list)
        )

        experiment_latex_code = []
        experiment_index = 0
        for name, data in point_dict.items():
            color_id = experiment_index % len(self.colors)

            coordinates = []
            for metric, value in data.items():
                coordinates.append(f'    ({metric}, {value})')

            experiment_replace_dict = dict(
                legend=name,
                color=self.colors[color_id],
                coordinates='\n'.join(coordinates),
            )
            experiment_latex_code.append(self.replace(self.latex_bar_experiment_template, **experiment_replace_dict))
            experiment_index += 1

        wrapper_replace_dict['experiments'] = '\n'.join(experiment_latex_code)
        code = self.replace(self.latex_bar_wrapper_template, **wrapper_replace_dict)
        filename = 'performance_comparison_bar.tex'

        self.export_save(filename, code)
        self.update_main(filename)

    def generate_test_table(self, metrics: List[str]):
        data_dict = dict()
        metric_list = []
        for experiment in self.experiments:  # type: Experiment
            test_results = experiment.get_test_results(metrics)
            if test_results:
                data_dict[experiment.name] = test_results
                metric_list = list(test_results.keys())

        wrapper_replace_dict = dict(
            title='Performance comparison',
            num=len(data_dict),
            ob=self.latex_ob,
            center='c' * len(metric_list),
            head=' & '.join([f'\\textbf{{{m}}}' for m in metric_list]),
        )

        experiment_latex_code = []
        for name, data in data_dict.items():
            values = []
            for metric in metric_list:
                values.append(data[metric])
            values = ' & '.join(values)

            experiment_replace_dict = dict(
                name=name,
                values=values,
            )
            experiment_latex_code.append(self.replace(self.latex_table_experiment_template, **experiment_replace_dict))

        wrapper_replace_dict['experiments'] = '\n'.join(experiment_latex_code)
        code = self.replace(self.latex_table_wrapper_template, **wrapper_replace_dict)
        filename = 'performance_comparison_table.tex'

        self.export_save(filename, code)
        self.update_main(filename)

    def generate_dataset_statistics(self):
        group_dict = dict()
        for experiment in self.experiments:
            if experiment.data_full not in group_dict:
                group_dict[experiment.data_full] = []
            group_dict[experiment.data_full].append(experiment)

        keys = None
        datasets = []
        for data_name in group_dict:
            dataset_info = group_dict[data_name][0].get_dataset_info()
            exps = []
            for experiment in group_dict[data_name]:  # type: Experiment
                exps.append(experiment.name)
            dataset_info['exp'] = ', '.join(exps)

            new_dataset_info = dict()
            for key, value in dataset_info.items():
                value = value.replace('_', '\\_') if isinstance(value, str) else value
                key = key.replace('_', '\\_')
                new_dataset_info[key] = value
            dataset_info = new_dataset_info

            keys = list(dataset_info.keys())
            datasets.append(dataset_info)

        keys = list(filter(lambda x: x != 'name', keys))

        wrapper_replace_dict = dict(
            title='Data',
            ob=self.latex_ob,
            left='l' * len(datasets),
            head=' & '.join([f'\\textbf{{{dataset["name"]}}}' for dataset in datasets]),
        )

        statistics = []
        for key in keys:
            line = [key]
            for dataset in datasets:
                line.append(str(dataset[key]))
            line = ' & '.join(line)
            statistics.append(line + ' \\\\')
        statistics = '\n'.join(statistics)

        wrapper_replace_dict['statistics'] = statistics
        code = self.replace(self.latex_table_statistics_template, **wrapper_replace_dict)
        filename = 'dataset_statistics.tex'

        self.export_save(filename, code)
        self.update_main(filename)

    def postact(self):
        # step1. zip the self.export_dir to a zip file

        charts = '\n'.join([f'\input{{{filename}}}' for filename in self.display_charts])
        conditions = []
        for name in self.display_triggers:
            conditions.append(self.replace(self.latex_condition_name_template, name=name))
        conditions = '\n'.join(conditions)
        self.max_tex_code = self.replace(self.max_tex_code, conditions=conditions, charts=charts)

        self.export_save(self.main_tex_filename, self.max_tex_code)

        zip_filepath = self.export_dir + '.zip'
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.export_dir):
                for file in files:
                    abs_path = cast(str, os.path.join(root, file))
                    relative_path = os.path.relpath(abs_path, start=self.export_dir)
                    zipf.write(abs_path, relative_path)

        pnt('Zip file created:', zip_filepath)

        # step2. upload the zip file to 0x0.st
        with open(zip_filepath, "rb") as f:
            res = requests.post(
                url="https://0x0.st",
                files={"file": f},
                headers={
                    "User-Agent": "Genemmender/github-dot-com-slash-jyonn-slash-genemmender"
                }
            )

        link = 'https://www.overleaf.com/docs?snip_uri=' + res.text.strip()
        pnt('Click to get your visualization results:', link)
