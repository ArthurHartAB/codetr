from clearml import Task, Logger
from mmcv.cnn import VGG
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

from typing import Optional
from torch.utils.data import DataLoader
import os.path as osp

import pandas as pd
import matplotlib.pyplot as plt

# from eval_bin import run_eval_bin
# from services.logging_utils import setup_logger
# from run_multi_eval import _update_config

import os
import numpy as np
import json
# from projects.CODETR.evaluation.log_parser import NEWLogParser


@HOOKS.register_module()
class ABEvalHook(Hook):

    def __init__(self, eval_path, gt_path, python_path, eval_name='FastEval') -> None:
        self.eval_path = eval_path
        self.gt_path = gt_path
        self.eval_name = eval_name
        self.python_path = python_path

    def after_test_epoch(self, runner: Runner, metrics):

        if runner.rank != 0:
            return

        det_path = osp.join(runner.work_dir, f"res_df_ep{runner.epoch}.tsv")

        output_dir = osp.join(
            runner.work_dir, f"{self.eval_name}_ep{runner.epoch}")

        os.makedirs(output_dir, exist_ok=True)

        gt_df = pd.read_csv(self.gt_path, sep="\t")

        det_df = pd.read_csv(det_path, sep="\t")

        cut_gt_df = gt_df[gt_df["name"].isin(det_df["name"])]

        cut_gt_path = osp.join(output_dir, "cut_gt.tsv")

        cut_gt_df.to_csv(cut_gt_path, sep="\t", index=False)

        os.system(f"export PYTHONPATH=$PYTONPATH:{self.eval_path}")

        command = f"cd {self.eval_path} && {self.python_path} runner.py --gt_path {cut_gt_path} --det_path {det_path} --output_path {output_dir}"

        os.system(command)

    def after_val_epoch(self, runner: Runner, metrics):
        self.after_test_epoch(runner, metrics)


@HOOKS.register_module()
class ABClearMLHook(Hook):

    def __init__(self, project_name, task_name) -> None:
        # Initialize the ClearML task
        self.task = Task.init(
            project_name=project_name, task_name=task_name)
        self.logger = self.task.get_logger()

    def after_test_epoch(self, runner: Runner, metrics):

        if runner.rank != 0:
            return

        eval_dir = osp.join(runner.work_dir, f"FastEval_ep{runner.epoch}")

        metrics_incl_path = osp.join(eval_dir, "metrics_incl_ignores.tsv")
        metrics_excl_path = osp.join(eval_dir, "metrics_excl_ignores.tsv")

        metrics_incl_df = pd.read_csv(metrics_incl_path, sep='\t')
        metrics_excl_df = pd.read_csv(metrics_excl_path, sep='\t')

        for category in metrics_incl_df.category.unique():
            mectics_incl_df_category = metrics_incl_df[metrics_incl_df.category == category]
            for bin in mectics_incl_df_category.bin.unique():
                metrics_incl_df_category_bin = metrics_incl_df[metrics_incl_df.bin == bin]
                precisions = metrics_incl_df_category_bin.precision_strict.values()
                recalls = metrics_incl_df_category_bin.recall_strict.values()

                scatter2d = [precisions, recalls]

                self.logger.report_scatter2d(
                    title="PR curve",
                    series=f"{category} {bin}",
                    iteration=runner.epoch,
                    scatter=scatter2d,
                    xaxis="precision",
                    yaxis="recall",
                    mode='lines+markers'
                )

        print(f"Metrics have been sent to ClearML.")

    def after_val_epoch(self, runner: Runner, metrics):
        self.after_test_epoch(runner, metrics)
