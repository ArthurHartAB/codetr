_base_ = [
    './_base_/default_300.py'
]

custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.CODETR.codetr',
        'projects.CODETR.datasets',
        'projects.CODETR.loops',
        'projects.CODETR.runners',
        'projects.CODETR.evaluation',
    ])


test_ann_file = '/home/b2b/arthur/data/10_hard_images/coco_labels.json'
train_ann_file = test_ann_file

test_dataloader = dict(dataset=dict(ann_file=test_ann_file))
test_crop_dataloader = dict(dataset=dict(ann_file=test_ann_file))

val_dataloader = test_dataloader
val_crop_dataloader = test_crop_dataloader


train_dataloader = dict(dataset=dict(ann_file=train_ann_file))

load_from = '/home/b2b/arthur/git/codetr/work_dirs/default_600_crop_01_lr_01_wd_long_train/epoch_13.pth'
