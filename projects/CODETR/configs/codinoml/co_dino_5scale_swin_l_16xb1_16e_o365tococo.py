_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048),  (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048),  (640, 2048), (672, 2048), (704, 2048),
                            (736, 2048),  (768, 2048), (800, 2048), (832, 2048),
                            (864, 2048),  (896, 2048), (928, 2048), (960, 2048),
                            (992, 2048),  (1024, 2048), (1056, 2048),
                            (1088, 2048), (1120, 2048), (1152, 2048),
                            (1184, 2048), (1216, 2048), (1248, 2048),
                            (1280, 2048), (1312, 2048), (1344, 2048),
                            (1376, 2048), (1408, 2048), (1440, 2048),
                            (1472, 2048), (1504, 2048), (1536, 2048)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048), (512, 2048),
                            (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                            (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                            (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                            (992, 2048), (1024, 2048), (1056, 2048),
                            (1088, 2048), (1120, 2048), (1152, 2048),
                            (1184, 2048), (1216, 2048), (1248, 2048),
                            (1280, 2048), (1312, 2048), (1344, 2048),
                            (1376, 2048), (1408, 2048), (1440, 2048),
                            (1472, 2048), (1504, 2048), (1536, 2048)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(_delete_=True,
                        batch_size=1, num_workers=1, dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048/2, 1280/2), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=_base_.dataset_type,
        data_root='/media/DATADISK/bdd_dataset/images/100k/',
        # ann_file='annotations/instances_train2017.json',
        ann_file='/media/DATADISK/bdd_dataset/labels_coco/bdd100k_labels_images_train_coco.json',
        # data_prefix=dict(img='train2017/'),
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        # backend_args=_base_.backend_args
    ))

val_dataloader = dict(dataset=dict(
                      _delete_=True,
                      # ann_file='annotations/instances_train2017.json',
                      type=_base_.dataset_type,
                      data_root='/media/DATADISK/bdd_dataset/',
                      ann_file='/media/DATADISK/bdd_dataset/labels_coco/bdd100k_labels_images_val_coco.json',
                      # data_prefix=dict(img='train2017/'),
                      data_prefix=dict(img='images/100k/val/'),
                      test_mode=True,
                      pipeline=test_pipeline,
                      # backend_args=_base_.backend_args
                      ))


test_evaluator = dict(
    ann_file='/media/DATADISK/bdd_dataset/labels_coco/bdd100k_labels_images_val_coco.json')

val_evaluator = dict(
    ann_file='/media/DATADISK/bdd_dataset/labels_coco/bdd100k_labels_images_val_coco.json')


test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 16
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]
