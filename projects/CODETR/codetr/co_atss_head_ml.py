from projects.CODETR.codetr.co_atss_head import CoATSSHead

from mmdet.registry import MODELS


@MODELS.register_module()
class CoATSSHeadML(CoATSSHead):
    pass
