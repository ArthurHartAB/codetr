from projects.CODETR.codetr.codetr import CoDETR

from mmdet.registry import MODELS

from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union

import copy


@MODELS.register_module()
class CoDETRML(CoDETR):

    def __init__(
            self,
            backbone,
            remapping=None,
            neck=None,
            query_head=None,  # detr head
            rpn_head=None,  # two-stage rpn
            roi_head=[None],  # two-stage
            bbox_head=[None],  # one-stage
            train_cfg=[None, None],
            test_cfg=[None, None],
            # Control whether to consider positive samples
            # from the auxiliary head as additional positive queries.
            with_pos_coord=True,
            use_lsj=True,
            eval_module='detr',
            # Evaluate the Nth head.
            eval_index=0,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None):
        super(CoDETR, self).__init__(
            backbone,
            neck,
            query_head,
            rpn_head,
            roi_head,
            bbox_head,
            train_cfg,
            test_cfg,
            with_pos_coord,
            use_lsj,
            eval_module,
            eval_index,
            data_preprocessor,
            init_cfg)

        self.remapping = remapping

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        batch_input_shape = batch_data_samples[0].batch_input_shape
        if self.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = batch_input_shape
                img_metas['img_shape'] = [input_img_h, input_img_w]

        x = self.extract_feat(batch_inputs)

        losses_vanilla = self._loss(self, x,  batch_data_samples)
        losses_remapping = self._loss(
            self, x,  batch_data_samples, remapping=self.remapping)

        return losses_vanilla

    def _loss(self, x,  batch_data_samples, remapping=None) -> Union[dict, list]:

        losses = dict()

        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses

        # DETR encoder and decoder forward
        if self.with_query_head:
            bbox_losses, x = self.query_head.loss(
                x, batch_data_samples)  # bbox_losses
            losses.update(bbox_losses)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get(
                'rpn_proposal', self.test_cfg[self.head_idx].rpn)

            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, proposal_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)  # rpn_losses

            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)

            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            proposal_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].loss(x, proposal_list,
                                               batch_data_samples)  # roi_losses
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else:
                if 'pos_coords' in roi_losses.keys():
                    roi_losses.pop('pos_coords')
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)

        for i in range(len(self.bbox_head)):
            bbox_losses = self.bbox_head[i].loss(
                x, batch_data_samples)  # bbox_losses
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    bbox_losses.pop('pos_coords')
            bbox_losses = upd_loss(bbox_losses, idx=i + len(self.roi_head))
            losses.update(bbox_losses)

        if self.with_pos_coord and len(positive_coords) > 0:
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.loss_aux(x, positive_coords[i],
                                                       i, batch_data_samples)  # query_head_losses
                bbox_losses = upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)

        return losses
