from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class ABDataset(CocoDataset):
    # METAINFO = {
    #    'classes':
    #    ("4w", "nlv", "cipv"),
    #    'palette':
    #    [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
    # }

    # METAINFO = {
    #    'classes':
    #    ("rider", "ped", "2w", "car", "van", "truck"),
    #    'palette':
    #    [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    #     (0, 60, 100)]
    # }

    METAINFO = {
        'classes':
        ("2w", "4w", "ped", "rider"),
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
    }

    # def evaluate(self, results):
    #    print("GOT RESULTS")
