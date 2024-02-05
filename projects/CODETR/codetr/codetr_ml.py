from projects.CODETR.codetr.codetr import CoDETR

from mmdet.registry import MODELS


@MODELS.register_module()
class CoDETRML(CoDETR):
    pass
