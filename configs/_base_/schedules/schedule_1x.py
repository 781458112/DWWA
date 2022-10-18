# # optimizer
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)#faster rcnn
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     # gamma=0.5,
#     step=[7,10]) #8 11  #15，23
# runner = dict(type='EpochBasedRunner', max_epochs=18)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)#faster rcnn
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=22000, #21000 VOC
    warmup_ratio=0.001,
    # gamma=0.5,
    step=[5,7]) # 21 32  #15，23

runner = dict(type='EpochBasedRunner', max_epochs=8)
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)#faster rcnn
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     # gamma=0.5,
#     step=[7,10]) #8 11  #15，23
# runner = dict(type='EpochBasedRunner', max_epochs=18)