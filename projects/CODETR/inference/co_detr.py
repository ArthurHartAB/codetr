# From https://github.com/cortica-iei/AB_AutoTagging_Inference

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from mmdet.apis import DetInferencer

from projects.CODETR.inference.consts import CROP_H, CROP_H_POS, BORDER_DELTA, INTERSECTION_DELTA
from projects.CODETR.inference.utils import parse_results

from projects.CODETR.inference.thresholds import thresholds

import torch
import torchvision

from mmengine.dataset import Compose


class Inferencer:
    def __init__(self, model, pipeline_cfg, device: str):
        self.model = model
        self.device = device
        self.pipeline_cfg = pipeline_cfg
        self.pipeline = Compose(pipeline_cfg[1:])

    def forward(self, img_arr):
        results = {"predictions": []}
        with torch.no_grad():
            for img in img_arr:
                inputs = self.pipeline({"img": np.array(img)})

                results['predictions'].append(self.model.test_step(inputs))
        return results


class CoDetr:
    def __init__(self, model, test_pipeline_cfg, device: str):

        self.infer = Inferencer(model, test_pipeline_cfg, device)

    @staticmethod
    def get_image_array(image_uri):
        image = Image.open(image_uri)
        # image.resize((3840, 1980))
        return image

    @staticmethod
    def get_image_id(image_uri):
        return Path(image_uri).stem

    @staticmethod
    def get_crop(image_array):
        return image_array.crop((0, CROP_H_POS - CROP_H / 2, 3840, CROP_H_POS + CROP_H / 2))

    @staticmethod
    def get_bbox(row):
        x, y, h, w = row.height, row.width, row.x_center, row.y_center
        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
        return [x1, y1, x2, y2]

    @staticmethod
    def apply_nms(df, iou_threshold=0.9):
        df = df.reset_index(drop=True)

        for name in set(df.name):
            for label in set(df.label):
                frame = df[(df.name == name) & (df.label == label)]
                bboxes = frame.apply(CoDetr.get_bbox, axis=1).tolist()
                scores = frame['score'].tolist()

                keep = torchvision.ops.nms(
                    torch.tensor(bboxes, dtype=torch.float), torch.tensor(scores, dtype=torch.float)/100, iou_threshold)
                remove = list(set(range(len(bboxes))) - set(keep.tolist()))

                if len(remove) > 0:
                    for idx in remove:

                        df.loc[df.index == (frame.index[idx]), 'label'] = \
                            df[df.index == (frame.index[idx])].label + '_nms'
        return df

    @staticmethod
    def apply_thresholds(df):
        for label in thresholds.keys():

            min_height = thresholds[label]['bins'][-1][0]
            is_in_bin = (df.label == label) & (df.height <= min_height) & (
                df.score < thresholds[label]['ignore_threshs'][-1])
            df.loc[is_in_bin, 'label'] = df[is_in_bin].label + "_ignore"

            max_height = thresholds[label]['bins'][0][1]
            is_in_bin = (df.label == label) & (df.height > max_height) & (
                df.score < thresholds[label]['ignore_threshs'][0])
            df.loc[is_in_bin, 'label'] = df[is_in_bin].label + "_ignore"

            for i in range(len(thresholds[label]['bins'])):

                bin_min_height, bin_max_height = thresholds[label]['bins'][i]
                ignore_thresh = thresholds[label]['ignore_threshs'][i]
                remove_thresh = thresholds[label]['remove_threshs'][i]

                is_in_bin = (df.label == label) & (
                    df.height > bin_min_height) & (df.height <= bin_max_height)

                is_ignore = is_in_bin & (df.score < ignore_thresh) & (
                    df.score >= remove_thresh)
                is_remove = is_in_bin & (df.score < remove_thresh)

                df.loc[is_ignore, 'label'] = df[is_ignore].label + "_ignore"
                df.loc[is_remove, 'label'] = df[is_remove].label + "_remove"

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

        return pd.concat([crop_df, image_df])

    @staticmethod
    def nms_results_intersection(df):
        boundary_bottom = CROP_H_POS + (CROP_H / 2) - BORDER_DELTA
        boundary_top = CROP_H_POS - (CROP_H / 2) + BORDER_DELTA

        det_bottom = df.y_center + (df.height / 2)
        det_top = df.y_center - (df.height / 2)

        is_det_in_boundary = (det_bottom < boundary_bottom) & (
            det_top > boundary_top)

        is_det_outside_boundary = (det_bottom > boundary_bottom - INTERSECTION_DELTA) | (
            det_top < boundary_top + INTERSECTION_DELTA)

        is_det_in_intersection_delta = is_det_in_boundary & is_det_outside_boundary

        in_intersection_df = df[is_det_in_intersection_delta]
        if in_intersection_df.empty:
            return df

        in_intersection_df_nms = CoDetr.apply_nms(
            in_intersection_df, iou_threshold=0.9)

        not_in_intersection_df = df[~is_det_in_intersection_delta]

        nms_df = pd.concat([not_in_intersection_df, in_intersection_df_nms])

        return nms_df

    def run(self, image_uri, *args, **kwargs):
        image_id = CoDetr.get_image_id(image_uri)
        image_array = CoDetr.get_image_array(image_uri)
        crop = CoDetr.get_crop(image_array)

        results_dict = self.infer.forward(
            (np.array(crop), np.array(image_array)))

        results_crop_df = parse_results(
            image_id, results_dict['predictions'][0])
        results_image_df = parse_results(
            image_id, results_dict['predictions'][1])

        res_df = CoDetr.merge_results(results_crop_df, results_image_df)
        res_df = CoDetr.nms_results_intersection(res_df)
        res_df = CoDetr.apply_thresholds(res_df)

        return res_df
