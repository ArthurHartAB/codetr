
dataset_type = 'ABDataset'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        608,
        3840,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(scale=(
        3840,
        1920,
    ), type='Resize'),
    dict(
        transforms=[
            [
                dict(
                    bbox_params=dict(
                        filter_lost_elements=True,
                        format='pascal_voc',
                        label_fields=[
                            'gt_bboxes_labels',
                            'gt_ignore_flags',
                        ],
                        min_visibility=0.0,
                        type='BboxParams'),
                    keymap=dict(
                        gt_bboxes='bboxes', gt_masks='masks', img='image'),
                    skip_img_without_anno=True,
                    transforms=[
                        dict(
                            always_apply=True,
                            height=300,
                            type='CenterCrop',
                            width=3840),
                    ],
                    type='Albu'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            2048,
                        ),
                        (
                            512,
                            2048,
                        ),
                        (
                            544,
                            2048,
                        ),
                        (
                            576,
                            2048,
                        ),
                        (
                            608,
                            2048,
                        )
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            2048,
                        ),
                        (
                            512,
                            2048,
                        ),
                        (
                            544,
                            2048,
                        ),
                        (
                            576,
                            2048,
                        ),
                        (
                            608,
                            2048,
                        )

                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/home/b2b/arthur/data/921_hard_images/coco_labels.json',
        data_prefix=dict(img='/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                608,
                3840,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))

test_crop_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/home/b2b/arthur/data/921_hard_images/coco_labels.json',
        data_prefix=dict(img='/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1920,
                3840,
            ), type='Resize'),

            dict(
                transforms=[
                    dict(
                        always_apply=True,
                        height=300,
                        type='CenterCrop',
                        width=3840),
                ],
                type='Albu'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        #
        ann_file='/home/b2b/arthur/data/all_data_coco.json',
        data_prefix=dict(img='/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        type=dataset_type),
    num_workers=10,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))

val_dataloader = test_dataloader

val_crop_dataloader = test_crop_dataloader
