# From https://github.com/cortica-iei/AB_AutoTagging_Inference

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from mmdet.apis import DetInferencer

from projects.CODETR.inference.consts import (CROP_H, CROP_H_POS, BORDER_DELTA, INTERSECTION_DELTA, SCORE_COLUMN_NAME,
                                              LABEL_COLUMN_NAME, HANDLING_HINT_COLUMN_NAME, BATCH_SIZE, IMAGE_SHAPE,
                                              TRUNCATED_COLUMN_NAME, NMS_WAS_APPLIED_COLUMN_NAME, ORIGIN_COLUMN_NAME)
from projects.CODETR.inference.utils import add_bbox_columns, remove_bbox_columns, parse_results, clip_bboxes

from projects.CODETR.inference.thresholds import thresholds, MIN_THRESHOLD

import torch
import torchvision


class CoDetr:
    def __init__(self, target_config_path, weights_path, device: str):
        self.weights_path = weights_path
        self.config_path = target_config_path
        self.infer = DetInferencer(
            model=self.config_path, weights=self.weights_path, device=device)

    @staticmethod
    def apply_nms(df: pd.DataFrame, iou_threshold: float = 0.9, handling_hint: str = 'nms'):

        def nms_per_group(frame):
            bboxes = frame[['x1', 'y1', 'x2', 'y2']].values
            scores = frame[SCORE_COLUMN_NAME].values

            nms_indices_in_frame = torchvision.ops.nms(
                torch.tensor(bboxes, dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32) / 100,
                iou_threshold
            ).tolist()

            frame_indices = frame.index.to_numpy()
            nms_indices_set = set(frame_indices[nms_indices_in_frame])

            frame.loc[~frame.index.isin(
                nms_indices_set), HANDLING_HINT_COLUMN_NAME] = handling_hint
            return frame

        df = df.reset_index(drop=True)
        add_bbox_columns(df)
        df = df.groupby(['name', LABEL_COLUMN_NAME]).apply(
            nms_per_group).reset_index(drop=True)
        remove_bbox_columns(df)

        return df

    @staticmethod
    def apply_thresholds(df):
        for label in thresholds.keys():

            min_height = thresholds[label]['bins'][-1][0]
            is_in_bin = (df[LABEL_COLUMN_NAME] == label) & (
                df.height <= min_height)
            df.loc[is_in_bin, HANDLING_HINT_COLUMN_NAME] = "small"

            max_height = thresholds[label]['bins'][0][1]
            is_in_bin = (df[LABEL_COLUMN_NAME] == label) & (
                df.height > max_height)
            df.loc[is_in_bin, HANDLING_HINT_COLUMN_NAME] = "large"

            for i in range(len(thresholds[label]['bins'])):
                bin_min_height, bin_max_height = thresholds[label]['bins'][i]
                ignore_thresh = thresholds[label]['ignore_threshs'][i]
                remove_thresh = thresholds[label]['remove_threshs'][i]

                is_in_bin = (df[LABEL_COLUMN_NAME] == label) & (
                    df.height > bin_min_height) & (df.height <= bin_max_height)

                should_handle = is_in_bin & (
                    df[HANDLING_HINT_COLUMN_NAME] == "keep")

                is_ignore = should_handle & (df[SCORE_COLUMN_NAME] < ignore_thresh) & (
                    df[SCORE_COLUMN_NAME] >= remove_thresh)
                is_remove = should_handle & (
                    df[SCORE_COLUMN_NAME] < remove_thresh)

                df.loc[is_ignore, HANDLING_HINT_COLUMN_NAME] = "ignore"
                df.loc[is_remove, HANDLING_HINT_COLUMN_NAME] = "remove"

        return df

    @staticmethod
    def merge_results(crop_df, image_df):
        crop_df['y_center'] = crop_df['y_center'] + CROP_H_POS - (CROP_H / 2)

        boundary_bottom = CROP_H_POS + (CROP_H / 2) - BORDER_DELTA
        boundary_top = CROP_H_POS - (CROP_H / 2) + BORDER_DELTA

        crop_det_bottom = crop_df.y_center + (crop_df.height / 2)
        crop_det_top = crop_df.y_center - (crop_df.height / 2)

        is_crop_det_in_boundary = (crop_det_bottom < boundary_bottom) & (
            crop_det_top > boundary_top)
        crop_df = crop_df[is_crop_det_in_boundary]

        image_det_bottom = image_df.y_center + (image_df.height / 2)
        image_det_top = image_df.y_center - (image_df.height / 2)

        is_image_det_outside_boundary = (image_det_bottom > boundary_bottom - INTERSECTION_DELTA) | (
            image_det_top < boundary_top + INTERSECTION_DELTA)

        image_df = image_df[is_image_det_outside_boundary]

        crop_df[ORIGIN_COLUMN_NAME] = "crop"
        image_df[ORIGIN_COLUMN_NAME] = "image"

        return pd.concat([crop_df, image_df])

    @staticmethod
    def nms_results_intersection(df):

        df[NMS_WAS_APPLIED_COLUMN_NAME] = 0

        boundary_bottom = CROP_H_POS + (CROP_H / 2) - BORDER_DELTA
        boundary_top = CROP_H_POS - (CROP_H / 2) + BORDER_DELTA

        det_bottom = df.y_center + (df.height / 2)
        det_top = df.y_center - (df.height / 2)

        is_det_in_boundary = (det_bottom < boundary_bottom + INTERSECTION_DELTA) & (
            det_top > boundary_top - INTERSECTION_DELTA)

        is_det_outside_boundary = (det_bottom > boundary_bottom - 2*INTERSECTION_DELTA) | (
            det_top < boundary_top + 2*INTERSECTION_DELTA)

        is_det_in_intersection_delta = is_det_in_boundary & is_det_outside_boundary

        in_intersection_df = df[is_det_in_intersection_delta]
        if in_intersection_df.empty:
            return df

        in_intersection_df_nms = CoDetr.apply_nms(
            in_intersection_df, iou_threshold=0.75, handling_hint='nms_75')

        in_intersection_df_nms = CoDetr.apply_nms(
            in_intersection_df_nms, iou_threshold=0.9, handling_hint='nms_90')

        in_intersection_df_nms.loc[in_intersection_df_nms[HANDLING_HINT_COLUMN_NAME] == 'nms_75',
                                   SCORE_COLUMN_NAME] = in_intersection_df_nms[in_intersection_df_nms[HANDLING_HINT_COLUMN_NAME] == 'nms_75'][SCORE_COLUMN_NAME]/2

        in_intersection_df_nms.loc[in_intersection_df_nms[HANDLING_HINT_COLUMN_NAME]
                                   == 'nms_75', HANDLING_HINT_COLUMN_NAME] = 'ignore'
        in_intersection_df_nms.loc[in_intersection_df_nms[HANDLING_HINT_COLUMN_NAME]
                                   == 'nms_90', HANDLING_HINT_COLUMN_NAME] = 'nms'

        in_intersection_df_nms[NMS_WAS_APPLIED_COLUMN_NAME] = 1

        not_in_intersection_df = df[~is_det_in_intersection_delta]

        nms_df = pd.concat([not_in_intersection_df, in_intersection_df_nms])

        return nms_df

    @staticmethod
    def mark_truncated(detections: pd.DataFrame) -> pd.DataFrame:
        truncate_margin = np.maximum(10, detections.width * 0.1)

        detections[TRUNCATED_COLUMN_NAME] = (
            ((detections.x_center - detections.width / 2) <= truncate_margin) |
            ((IMAGE_SHAPE[1] - (detections.x_center +
                                detections.width / 2)) <= truncate_margin)
        ).astype(int)

        return detections
