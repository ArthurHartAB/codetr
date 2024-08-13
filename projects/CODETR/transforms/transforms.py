# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image import imresize
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

Number = Union[int, float]


@TRANSFORMS.register_module()
class ABRandomCrop(BaseTransform):

    def __init__(self,
                 crop_size: tuple,
                 crop_type: str = 'absolute',
                 allow_negative_crop: bool = False,
                 recompute_bbox: bool = False,
                 bbox_clip_border: bool = True) -> None:
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
            if crop_type == 'absolute_range':
                assert crop_size[0] <= crop_size[1]
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:

        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('loss_weight', None) is not None:
                results['loss_weight'] = \
                    results['loss_weight'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get('gt_instances_ids', None) is not None:
                results['gt_instances_ids'] = \
                    results['gt_instances_ids'][valid_inds]

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:

        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return offset_h, offset_w

    @cache_randomness
    def _get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:

        h, w = image_size
        if self.crop_type == 'absolute':
            return min(self.crop_size[1], h), min(self.crop_size[0], w)
        elif self.crop_type == 'absolute_range':
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_w, crop_h = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        else:
            # 'relative_range'
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:

        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'recompute_bbox={self.recompute_bbox}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@TRANSFORMS.register_module()
@avoid_cache_randomness
class ABAlbu(BaseTransform):

    def __init__(self,
                 transforms: List[dict],
                 bbox_params: Optional[dict] = None,
                 keymap: Optional[dict] = None,
                 skip_img_without_anno: bool = False) -> None:
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        if bbox_params is not None:
            bbox_params = copy.deepcopy(bbox_params)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            self.origin_label_fields.append('loss_weight')  # arthur
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes',
                'loss_weight': 'loss_weight',  # arthur
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()
        obj_type = args.pop('type')
        if is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d: dict, keymap: dict) -> dict:
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function of Albu."""
        # TODO: gt_seg_map is not currently supported
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        results, ori_masks = self._preprocess_results(results)
        results = self.aug(**results)
        results = self._postprocess_results(results, ori_masks)
        if results is None:
            return None
        # back to the original format
        results = self.mapper(results, self.keymap_back)
        results['img_shape'] = results['img'].shape[:2]
        return results

    def _preprocess_results(self, results: dict) -> tuple:
        """Pre-processing results to facilitate the use of Albu."""
        if 'bboxes' in results:
            # to list of boxes
            if not isinstance(results['bboxes'], HorizontalBoxes):
                raise NotImplementedError(
                    'Albu only supports horizontal boxes now')
            bboxes = results['bboxes'].numpy()
            results['bboxes'] = [x for x in bboxes]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        if 'loss_weight' in results:  # arthur
            if self.filter_lost_elements:
                results['loss_weight'] = np.array(
                    [results['loss_weight'][i] for i in results['idx_mapper']])

        # TODO: Support mask structure in albu
        ori_masks = None
        if 'masks' in results:
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            if albumentations.__version__ < '0.5':
                results['masks'] = results['masks'].masks
            else:
                results['masks'] = [mask for mask in results['masks'].masks]

        return results, ori_masks

    def _postprocess_results(
            self,
            results: dict,
            ori_masks: Optional[Union[BitmapMasks,
                                      PolygonMasks]] = None) -> dict:
        """Post-processing Albu output."""
        # albumentations may return np.array or list on different versions
        if 'gt_bboxes_labels' in results and isinstance(
                results['gt_bboxes_labels'], list):
            results['gt_bboxes_labels'] = np.array(
                results['gt_bboxes_labels'], dtype=np.int64)
        if 'gt_ignore_flags' in results and isinstance(
                results['gt_ignore_flags'], list):
            results['gt_ignore_flags'] = np.array(
                results['gt_ignore_flags'], dtype=bool)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)
            results['bboxes'] = HorizontalBoxes(results['bboxes'])

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    assert ori_masks is not None
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'],
                        results['masks'][0].shape[0],
                        results['masks'][0].shape[1],
                    )
                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None
            elif 'masks' in results:
                results['masks'] = ori_masks.__class__(results['masks'],
                                                       ori_masks.height,
                                                       ori_masks.width)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str
