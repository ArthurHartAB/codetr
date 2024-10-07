_base_ = [
    './_base_/model.py',
    './_base_/dataset_600.py',
    './_base_/schedule.py',
    './_base_/runtime.py'
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

custom_hooks = [
    dict(type='TSVHook'),
    dict(type="ABEvalHook",
         eval_path="/home/b2b/arthur/git/Eval",
         gt_path="/home/b2b/arthur/data/10_hard_images/gt.tsv",
         python_path='/home/b2b/anaconda3/envs/od_arthur/bin/python3',
         eval_name='FastEval'),
]

test_evaluator = dict(
    type='CropEvaluator',
    metrics=dict(type="TSVSaver",
                 crop_h=600,
                 crop_h_pos=960)
)

load_from = '/home/b2b/arthur/git/codetr/work_dirs/default_600_crop_01_lr_01_wd_long_train/epoch_13.pth'
