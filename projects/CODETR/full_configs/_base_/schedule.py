max_epochs = 16

auto_scale_lr = dict(base_batch_size=16)

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
