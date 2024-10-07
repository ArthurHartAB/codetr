# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.fileio import dump
from mmengine.logging import print_log
from mmengine.registry import METRICS
from mmengine.structures import BaseDataElement
from mmengine.evaluator import BaseMetric

import os

from projects.CODETR.inference.utils import parse_results, clip_bboxes
from projects.CODETR.inference.co_detr import CoDetr
import projects.CODETR.inference.consts as consts


from projects.CODETR.inference.thresholds import MIN_THRESHOLD


import pandas as pd


@METRICS.register_module()
class TSVSaver(BaseMetric):

    def __init__(self,
                 crop_h: int = 300, crop_h_pos: int = 960,
                 collect_device: str = 'cpu',
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(
            collect_device=collect_device, collect_dir=collect_dir)

        consts.CROP_H = crop_h
        consts.CROP_H_POS = crop_h_pos

        self.crop_results = []

    def process(self, data_samples, crop_data_samples,
                data_batch, crop_data_batch) -> None:
        """transfer tensors in predictions to CPU."""

        cpu_data_samples = _to_cpu(data_samples)
        cpu_crop_data_samples = _to_cpu(crop_data_samples)

        for el in cpu_crop_data_samples:
            el['from'] = 'crop'
        for el in cpu_data_samples:
            el['from'] = 'resize'

        self.results.extend(cpu_data_samples)
        self.results.extend(cpu_crop_data_samples)

    def compute_metrics(self, results: list) -> dict:

        resize_dfs = []
        crop_dfs = []

        for result in results:
            img_name = os.path.basename(result['img_path'])
            result['pred_instances']['bboxes'] = result['pred_instances']['bboxes'].tolist()

            if result['from'] == 'crop':
                crop_dfs.append(parse_results(
                    img_name, result['pred_instances']))
            else:
                resize_dfs.append(parse_results(
                    img_name, result['pred_instances']))

        crop_df = pd.concat(crop_dfs)
        resize_df = pd.concat(resize_dfs)

        res_dfs = []

        for img_name in crop_df['name'].unique():
            img_crop_df = crop_df[crop_df['name'] == img_name]
            img_resize_df = resize_df[resize_df['name'] == img_name]

            image_res_df = CoDetr.merge_results(img_crop_df, img_resize_df)
            image_res_df[consts.HANDLING_HINT_COLUMN_NAME] = "keep"
            image_res_df = CoDetr.nms_results_intersection(image_res_df)
            image_res_df = CoDetr.apply_thresholds(image_res_df)

            image_res_df = image_res_df[image_res_df[consts.SCORE_COLUMN_NAME]
                                        > MIN_THRESHOLD]
            image_res_df = clip_bboxes(image_res_df)
            image_res_df = CoDetr.mark_truncated(image_res_df)

            res_dfs.append(image_res_df)

        res_df = pd.concat(res_dfs)

        # res_df.to_csv(os.path.join(
        #    self.out_folder_path, "merge.tsv"), sep='\t')
        # crop_df.to_csv(os.path.join(
        #    self.out_folder_path, "crop.tsv"), sep='\t')
        # resize_df.to_csv(os.path.join(
        #    self.out_folder_path, "resize.tsv"), sep='\t')

        return {"res_df": res_df, "crop_df": crop_df, "resize_df": resize_df}


def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data
