from mmcv.cnn import VGG
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

from typing import Optional
from torch.utils.data import DataLoader
import os.path as osp

import pandas as pd

#from eval_bin import run_eval_bin
#from services.logging_utils import setup_logger
#from run_multi_eval import _update_config

import os

from projects.CODETR.evaluation.log_parser import NEWLogParser


@HOOKS.register_module()
class ABEvalHook(Hook):

    def __init__(self, gt_path, config_path) -> None:
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

        #logger = setup_logger(0, f"{output_dir}/log.txt")
        #logger.info(f"Running eval for {det_path} and {self.gt_path}")
        #config_path = _update_config(
        #    self.config_path, output_dir, cut_gt_path, det_path)
        #run_eval_bin(logger, config_path)


@HOOKS.register_module()
class ABKPIHook(Hook):

    def __init__(self) -> None:
        pass

    def after_test_epoch(self, runner: Runner, metrics):

        if runner.rank != 0:
            return

        eval_dir = osp.join(runner.work_dir, f"NewEval_ep{runner.epoch}")

        parser = NEWLogParser(eval_dir, precision_val=50)

        recalls_dict = parser.recalls_dict

        for cls, bins_data in recalls_dict.items():
            for bin, recall in bins_data.items():
                runner.metrics[f"recall_{cls}_{bin}"] = recall
