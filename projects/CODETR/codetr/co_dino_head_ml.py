from projects.CODETR.codetr.co_dino_head import CoDINOHead

from mmdet.registry import MODELS


@MODELS.register_module()
class CoDINOHeadML(CoDINOHead):
    pass
