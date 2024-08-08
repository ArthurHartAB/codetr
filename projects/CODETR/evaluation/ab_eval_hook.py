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


def update_config(config_path, output_dir, ground_truth_path, det_path, save_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    config['output_dir'] = output_dir
    config['ground_truth_path'] = ground_truth_path
    config['det_path'] = det_path

    with open(save_path, 'w') as file:
        json.dump(config, file, indent=4)


@HOOKS.register_module()
class ABEvalHook(Hook):

    def __init__(self, eval_path, gt_path, config_path) -> None:
        self.eval_path = eval_path
        self.gt_path = gt_path
        self.config_path = config_path

    def after_test_epoch(self, runner: Runner, metrics):

        if runner.rank != 0:
            return

        det_path = osp.join(runner.work_dir, f"res_df_ep{runner.epoch}.tsv")

        output_dir = osp.join(runner.work_dir, f"NewEval_ep{runner.epoch}")

        os.makedirs(output_dir, exist_ok=True)

        gt_df = pd.read_csv(self.gt_path, sep="\t")

        det_df = pd.read_csv(det_path, sep="\t")

        cut_gt_df = gt_df[gt_df["name"].isin(det_df["name"])]

        cut_gt_path = osp.join(output_dir, "cut_gt.tsv")

        cut_gt_df.to_csv(cut_gt_path, sep="\t", index=False)

        os.system(f"export PYTHONPATH=$PYTONPATH:{self.eval_path}")

        config_new_path = osp.join(output_dir, "wrapper.json")

        update_config(self.config_path, output_dir,
                      cut_gt_path, det_path, config_new_path)

        # --output_dir {output_dir} --ground_truth_path {cut_gt_path} --det_path {det_path}"
        # --ground_truth_path {cut_gt_path} --det_path {det_path}"
        command = f"cd {self.eval_path} && /home/b2b/anaconda3/envs/od_arthur/bin/python3 worker.py --config {config_new_path}"

        os.system(command)
        print("EVAL COMMAND EXECUTED!")

        # run eval here

    def after_val_epoch(self, runner: Runner, metrics):
        self.after_test_epoch(runner, metrics)


def recall_from_pr_df(pr_df, precision_value=0.75):
    recalls_dict = {}
    classes = pr_df.family.unique()

    for cls in classes:
        cls_pr = pr_df[pr_df.family == cls]
        bins = cls_pr.range.unique()
        recalls_dict[cls] = {}

        for bin in bins:
            cls_bin_pr = cls_pr[cls_pr.range == bin]
            precisions = cls_bin_pr.precision.values[2:-2]
            recalls = cls_bin_pr.recall.values[2:-2]

            # Sort the arrays to ensure monotonic increase
            sorted_indices = np.argsort(precisions)
            precisions = precisions[sorted_indices]
            recalls = recalls[sorted_indices]

            # Remove duplicate precision values by keeping the max recall for each precision
            unique_precisions, indices = np.unique(
                precisions, return_index=True)
            recalls = recalls[indices]

            if len(unique_precisions) == 0 or len(recalls) == 0:
                recall = 0.0
            elif precision_value < unique_precisions.min():
                recall = recalls[unique_precisions.argmin()]
            elif precision_value > unique_precisions.max():
                recall = recalls[unique_precisions.argmax()]
            else:
                recall = np.interp(precision_value, unique_precisions, recalls)

            recalls_dict[cls][bin] = recall

    return recalls_dict


@HOOKS.register_module()
class ABKPIHook(Hook):

    def __init__(self) -> None:
        pass

    def plot_recall(self, cls, bins_data, plot_dir, epoch):
        bins = list(bins_data.keys())
        recalls = list(bins_data.values())

        plt.figure(figsize=(10, 5))
        plt.plot(bins, recalls, marker='o', linestyle='-')
        plt.xlabel('Bin')
        plt.ylabel('Recall')
        plt.title(f'Recall vs Bin for class {cls} at epoch {epoch}')
        plt.ylim(0, 1)
        plt.grid(True)

        plot_path = osp.join(plot_dir, f"{cls}_recall_ep{epoch}.png")
        # Ensure the directory exists
        os.makedirs(osp.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

    def after_test_epoch(self, runner: Runner, metrics):

        if runner.rank != 0:
            return

        eval_dir = osp.join(runner.work_dir, f"NewEval_ep{runner.epoch}")
        plot_dir = osp.join(eval_dir, "plots")

        pr_df = pd.read_csv(
            osp.join(eval_dir, "outputs/all_kpis.tsv"), sep='\t')

        recalls_dict = recall_from_pr_df(pr_df, precision_value=0.75)

        for cls, bins_data in recalls_dict.items():
            for bin, recall in bins_data.items():
                metrics[f"recall_{cls}_{bin}"] = recall

            # Generate plot for each class
            self.plot_recall(cls, bins_data, plot_dir, runner.epoch)

        print(metrics)

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

        eval_dir = osp.join(runner.work_dir, f"NewEval_ep{runner.epoch}")
        plot_dir = osp.join(eval_dir, "plots")

        if not osp.exists(plot_dir):
            print(f"Plot directory {plot_dir} does not exist.")
            return

        # Log all plots in the directory to ClearML
        for plot_file in os.listdir(plot_dir):
            plot_path = osp.join(plot_dir, plot_file)
            if plot_path.endswith(".png"):
                cls = plot_file.split('_')[0]
                self.logger.report_image(
                    title=f'Recall vs Bin for class {cls}',
                    series=f'epoch_{runner.epoch}',
                    iteration=runner.epoch,
                    local_path=plot_path
                )

        print(f"Plots from {plot_dir} have been sent to ClearML.")

    def after_val_epoch(self, runner: Runner, metrics):
        self.after_test_epoch(runner, metrics)
