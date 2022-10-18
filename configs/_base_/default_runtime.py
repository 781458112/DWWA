checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# load_from ='/home/zh/zhao/mmdetection-2.18.0-new/tools/work_dirs/50/epoch_10.pth'
# load_from ='/home/zh/zhao/mmdetection-2.18.0-new/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
# load_from = r'/home/zh/zhao/mmdetection-2.18/mmdetection-master/cascadercnn-xiaobo-resnet50.pth'
resume_from = None
workflow = [('train', 1)]
