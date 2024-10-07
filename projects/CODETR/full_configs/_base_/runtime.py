runner_type = 'ABRunner'

log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)

test_evaluator = dict(
    type='CropEvaluator',
    metrics=dict(type="TSVSaver",
                 crop_h=600,
                 crop_h_pos=960)
)

train_cfg = dict(max_epochs=16, type='EpochBasedTrainLoop', val_interval=1)

test_cfg = dict(type='ABTestLoop')

val_cfg = dict(type='ABValLoop')

val_evaluator = test_evaluator

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

custom_hooks = [
    dict(type='TSVHook'),
    dict(type="ABEvalHook",
         eval_path="/home/b2b/arthur/git/Eval",
         gt_path="/home/b2b/arthur/data/921_hard_images/gt.tsv",
         python_path='/home/b2b/anaconda3/envs/od_arthur/bin/python3',
         eval_name='FastEval'),
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        # dict(type="ClearMLVisBackend", init_kwargs=clear_ml_init)
    ])
