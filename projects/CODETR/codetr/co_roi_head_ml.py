from projects.CODETR.codetr.co_roi_head import CoStandardRoIHead

from mmdet.registry import MODELS


@MODELS.register_module()
class CoStandardRoIHeadML(CoStandardRoIHead):
    pass
