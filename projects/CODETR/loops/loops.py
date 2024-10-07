# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals


@LOOPS.register_module()
class ABValLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 crop_dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):

        self._runner = runner
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader = runner.build_dataloader(
                dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader

        if isinstance(crop_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.crop_dataloader = runner.build_dataloader(
                crop_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.crop_dataloader = crop_dataloader

        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore

        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""

        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        crop_dataloader_iterator = iter(self.crop_dataloader)

        for idx, data_batch in enumerate(self.dataloader):

            try:
                crop_data_batch = next(crop_dataloader_iterator)
            except StopIteration:
                crop_dataloader_iterator = iter(self.crop_dataloader)
                crop_data_batch = next(crop_dataloader_iterator)

            self.run_iter(idx, data_batch, crop_data_batch)

            # if idx >= 3:
            #    break

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset)*2 + 1)
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')

        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], crop_data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """

        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
            outputs_crop = self.runner.model.test_step(crop_data_batch)

        self.evaluator.process(
            outputs, outputs_crop, data_batch, crop_data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class ABTestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 crop_dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):

        self._runner = runner
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader = runner.build_dataloader(
                dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader

        if isinstance(crop_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.crop_dataloader = runner.build_dataloader(
                crop_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.crop_dataloader = crop_dataloader

        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore

        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""

        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        crop_dataloader_iterator = iter(self.crop_dataloader)

        for idx, data_batch in enumerate(self.dataloader):

            try:
                crop_data_batch = next(crop_dataloader_iterator)
            except StopIteration:
                crop_dataloader_iterator = iter(self.crop_dataloader)
                crop_data_batch = next(crop_dataloader_iterator)

            self.run_iter(idx, data_batch, crop_data_batch)

            # if idx >= 3:
            #    break

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset)*2 + 1)
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')

        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], crop_data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """

        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
            outputs_crop = self.runner.model.test_step(crop_data_batch)

        self.evaluator.process(
            outputs, outputs_crop, data_batch, crop_data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
