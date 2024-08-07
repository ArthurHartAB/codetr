# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterator, List, Optional, Sequence, Union

from mmengine.dataset import pseudo_collate
from mmengine.registry import EVALUATOR, METRICS
from mmengine.structures import BaseDataElement
from mmengine.evaluator.metric import BaseMetric
from mmengine.evaluator.evaluator import Evaluator


@EVALUATOR.register_module()
class CropEvaluator(Evaluator):

    def process(self,
                data_samples: Sequence[BaseDataElement],
                crop_data_samples: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None,
                crop_data_batch: Optional[Any] = None):

        _data_samples = []
        _crop_data_samples = []
        for data_sample in data_samples:
            if isinstance(data_sample, BaseDataElement):
                _data_samples.append(data_sample.to_dict())
            else:
                _data_samples.append(data_sample)

        for crop_data_sample in crop_data_samples:
            if isinstance(crop_data_sample, BaseDataElement):
                _crop_data_samples.append(crop_data_sample.to_dict())
            else:
                _crop_data_samples.append(crop_data_sample)

        for metric in self.metrics:
            metric.process(_data_samples, _crop_data_samples,
                           data_batch, crop_data_batch)

    def evaluate(self, size: int) -> dict:

        print("evaluating")

        metrics = {}
        for metric in self.metrics:
            _results = metric.evaluate(size)

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)
        return metrics
