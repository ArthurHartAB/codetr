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

    dict(type="Resize", scale=(3840, 1920)),

    dict(type='RandomChoice',
         transforms=[
             [dict(
                 type='Albu',
                 transforms=[dict(type='CenterCrop', height=300,
                                  width=3840, always_apply=True)],
                 bbox_params=dict(
                     type='BboxParams',
                     format='pascal_voc',
                     label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
                     min_visibility=0.0,
                     filter_lost_elements=True),
                 keymap={
                     'img': 'image',
                     'gt_masks': 'masks',
                     'gt_bboxes': 'bboxes'
                 },
                 skip_img_without_anno=True)
              ],
             [
                 dict(
                     type='RandomChoiceResize',
                     scales=[(480, 2048), (512, 2048),
                             (544, 2048), (576, 2048),
                             (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                             (736, 2048), (768, 2048)],
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
                     crop_size=(384, 600),  # crop_size=(384, 600)
                     allow_negative_crop=True),
                 dict(
                     type='RandomChoiceResize',
                     scales=[(480, 2048), (512, 2048),
                             (544, 2048), (576, 2048),
                             (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                             (736, 2048), (768, 2048)],
                     keep_ratio=True)
             ]

         ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(_delete_=True,
                        batch_size=1, num_workers=1, dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(768, 3840), keep_ratio=True),
    dict(type='Resize', scale=(832, 3840), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# test_pipeline = train_pipeline

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=_base_.dataset_type,
        # data_root='/media/DATADISK/bdd_dataset/images/100k/',
        # ann_file='annotations/instances_train2017.json',
        ann_file='/media/DATADISK/coco_datasets/ab_train/coco_labels.json',
        # data_prefix=dict(img='train2017/'),
        data_prefix=dict(img='/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        # backend_args=_base_.backend_args
    ))

val_dataloader = dict(dataset=dict(
                      _delete_=True,
                      # ann_file='annotations/instances_train2017.json',
                      type=_base_.dataset_type,
                      # data_root='/media/DATADISK/bdd_dataset/',
                      ann_file='/media/DATADISK/coco_datasets/amba_taiwan_train_0208/coco_labels.json',
                      data_prefix=dict(img='/'),
                      # data_prefix=dict(img='images/100k/val/'),
                      test_mode=True,
                      pipeline=test_pipeline,
                      # backend_args=_base_.backend_args
                      ))


test_evaluator = dict(
    ann_file='/media/DATADISK/coco_datasets/amba_taiwan_train_0208/coco_labels.json')

val_evaluator = dict(
    ann_file='/media/DATADISK/coco_datasets/amba_taiwan_train_0208/coco_labels.json')


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

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3),
)
log_processor = dict(by_epoch=True)

# visualization = dict(  # user visualization of validation and test results
#    type='DetVisualizationHook',
#    draw=True,
#    interval=1,
#    show=True)

# visualization = _base_.default_hooks.visualization
# visualization.update(dict(draw=True, show=True))
