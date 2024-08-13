import os
os.environ["CLEARML_CONFIG_FILE"] = '/home/b2b/arthur/git/codetr/projects/CODETR/clearml/clearml.conf'


auto_scale_lr = dict(base_batch_size=16)
backend_args = None

custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.CODETR.codetr',
        'projects.CODETR.datasets',
        'projects.CODETR.loops',
        'projects.CODETR.runners',
        'projects.CODETR.evaluation',
        'projects.CODETR.transforms',
        'projects.CODETR.visualizer'
    ])

dataset_type = 'ABDataset'

runner_type = 'ABRunner'

num_classes = 4


clear_ml_init = dict(
    project_name="CoDETR",
    task_name="test",
    task_type="training",
    reuse_last_task_id=False,
    continue_last_task=False,
    output_uri=None,
    auto_connect_arg_parser=True,
    auto_connect_frameworks=True,
    auto_resource_monitoring=True,
    auto_connect_streams=True,
)


default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=10, type='LoggerHook'),
    # logger=dict(_scope_='mmdet', type="ClearMLLoggerHook", interval=10),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

load_from = '/home/b2b/arthur/git/codetr/weights/codetr_best.pth'

log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
loss_lambda = 2.0
max_epochs = 16
max_iters = 270000

model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=192,
        init_cfg=dict(
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=12,
        with_cp=True),
    bbox_head=[
        dict(
            anchor_generator=dict(
                octave_base_scale=8,
                ratios=[
                    1.0,
                ],
                scales_per_octave=1,
                strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                ],
                type='AnchorGenerator'),
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            loss_bbox=dict(loss_weight=24.0, type='GIoULoss'),
            loss_centerness=dict(
                loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=12.0,
                type='FocalLoss',
                use_sigmoid=True),
            num_classes=num_classes,
            stacked_convs=1,
            type='CoATSSHead'),
    ],
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=False,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    eval_module='detr',
    neck=dict(
        act_cfg=None,
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=5,
        out_channels=256,
        type='ChannelMapper'),
    query_head=dict(
        as_two_stage=True,
        dn_cfg=dict(
            box_noise_scale=0.4,
            group_cfg=dict(dynamic=True, num_dn_queries=500, num_groups=None),
            label_noise_scale=0.5),
        in_channels=2048,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(  # arthur : Change this one for multilabeling
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=num_classes,
        num_query=900,
        positional_encoding=dict(
            normalize=True,
            num_feats=128,
            temperature=20,
            type='SinePositionalEncoding'),
        transformer=dict(
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_heads=8,
                            type='MultiheadAttention'),
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_levels=5,
                            type='MultiScaleDeformableAttention'),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='DetrTransformerDecoderLayer'),
                type='DinoTransformerDecoder'),
            encoder=dict(
                num_layers=6,
                transformerlayers=dict(
                    attn_cfgs=dict(
                        dropout=0.0,
                        embed_dims=256,
                        num_levels=5,
                        type='MultiScaleDeformableAttention'),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='DetrTransformerEncoder',
                with_cp=6),
            num_co_heads=2,
            num_feature_levels=5,
            type='CoDinoTransformer',
            with_coord_feat=False),
        type='CoDINOHead'),
    roi_head=[
        dict(
            bbox_head=dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=120.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=12.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=num_classes,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            bbox_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                ],
                finest_scale=56,
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            type='CoStandardRoIHead'),
    ],
    rpn_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                4,
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=12.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=[
        dict(max_per_img=300),
        dict(
            rcnn=dict(
                max_per_img=100,
                nms=dict(iou_threshold=0.5, type='nms'),
                score_thr=0.0),
            rpn=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=1000)),
        dict(
            max_per_img=100,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=1000,
            score_thr=0.0),
    ],
    train_cfg=[
        dict(
            assigner=dict(
                match_costs=[
                    # arthur : Change this one for multilabeling
                    dict(type='FocalLossCost', weight=2.0),
                    dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                    dict(iou_mode='giou', type='IoUCost', weight=2.0),
                ],
                type='HungarianAssigner')),
        dict(
            rcnn=dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            rpn=dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.3,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.5,
                    type='RandomSampler')),
            rpn_proposal=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=4000)),
        dict(
            allowed_border=-1,
            assigner=dict(topk=9, type='ATSSAssigner'),
            debug=False,
            pos_weight=-1),
    ],
    type='CoDETR',
    use_lsj=False)

num_dec_layer = 6
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')

param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=16,
        gamma=0.1,
        milestones=[
            8,
        ],
        type='MultiStepLR'),
]
pretrained = '/home/b2b/arthur/git/codetr/weights/codetr_best.pth'
resume = False
test_cfg = dict(type='ABTestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/home/b2b/arthur/data/921_hard_images/coco_labels.json',
        data_prefix=dict(img='/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                736,  # 832,
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

test_evaluator = dict(
    type='CropEvaluator',
    metrics=dict(type="TSVSaver",
                 out_folder_path="/home/b2b/arthur/git/codetr/tsvsaver/")
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        736,  # 832,
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
train_cfg = dict(max_epochs=16, type='EpochBasedTrainLoop', val_interval=1)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ABLoadAnnotations', with_bbox=True),
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
                    type='ABAlbu'),
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
                        ),
                        (
                            640,
                            2048,
                        ),
                        (
                            672,
                            2048,
                        ),
                        (
                            704,
                            2048,
                        ),
                        (
                            736,
                            2048,
                        ),
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
                    type='ABRandomCrop'),
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
                        ),
                        (
                            640,
                            2048,
                        ),
                        (
                            672,
                            2048,
                        ),
                        (
                            704,
                            2048,
                        ),
                        (
                            736,
                            2048,
                        )
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='ABPackDetInputs'),
]


train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        #
        ann_file='/home/b2b/arthur/data/all_data_coco_loss_weight.json',
        data_prefix=dict(img='/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        type=dataset_type),
    num_workers=10,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))

val_cfg = dict(type='ABValLoop')

val_dataloader = test_dataloader

val_crop_dataloader = test_crop_dataloader

val_evaluator = test_evaluator

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='ABDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        # dict(type="ClearMLVisBackend", init_kwargs=clear_ml_init)
    ])

custom_hooks = [
    dict(type='TSVHook'),
    dict(type="ABEvalHook",
         eval_path="/home/b2b/arthur/git/EvalBin",
         gt_path="/home/b2b/arthur/data/921_hard_images/gt.tsv",
         config_path="/home/b2b/arthur/git/EvalBin/wrappers/wrapper_config.json"),
    dict(type="ABKPIHook"),
    # dict(type="ABClearMLHook",
    #     project_name="CODETR",
    #     task_name="test")
]
