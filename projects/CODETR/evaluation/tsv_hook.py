from mmcv.cnn import VGG
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

from typing import Optional
from torch.utils.data import DataLoader
import os.path as osp

import pandas as pd


@HOOKS.register_module()
class TSVHook(Hook):
    """Check whether the `num_classes` in head matches the length of `classes`
    in `dataset.metainfo`."""

    def __init__(self) -> None:
        pass

    def after_test_epoch(self, runner: Runner, metrics):

        if runner.rank != 0:
            return

        res_df, crop_df, resize_df = metrics["res_df"], metrics["crop_df"], metrics["resize_df"]

        print("saving to : ", osp.join(runner.work_dir,
                                       f"res_df_{runner.epoch}.tsv"))

        res_df.to_csv(osp.join(runner.work_dir,
                      f"res_df_ep{runner.epoch}.tsv"), sep='\t')

        metrics.pop("res_df")
        metrics.pop("crop_df")
        metrics.pop("resize_df")

    def after_val_epoch(self, runner: Runner, metrics):

        if runner.rank != 0:
            return

        res_df, crop_df, resize_df = metrics["res_df"], metrics["crop_df"], metrics["resize_df"]

        print("saving to : ", osp.join(runner.work_dir,
                                       f"res_df_{runner.epoch}.tsv"))

        res_df.to_csv(osp.join(runner.work_dir,
                      f"res_df_ep{runner.epoch}.tsv"), sep='\t')

        metrics.pop("res_df")
        metrics.pop("crop_df")
        metrics.pop("resize_df")
