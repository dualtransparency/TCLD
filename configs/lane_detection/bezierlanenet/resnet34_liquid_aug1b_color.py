LIQUID_ROOT = '/mnt/ssd/liquid'

############################################################################################################
dataset = dict(
    name='LiquidAsBezier',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=LIQUID_ROOT,
    order=3,
    aux_segmentation=True
)
###########################################################################################################
# Aug level: strong-B (w.o. random lighting, more random affine like LaneATT)
train_augmentation = dict(
    name='Compose',
    transforms=[
        # dict(
        #     name='RandomAffine',
        #     degrees=(-10, 10),
        #     scale=(0.8, 1.2),
        #     translate=(50, 20),
        #     ignore_x=None
        # ),
        # dict(
        #     name='RandomHorizontalFlip',
        #     flip_prob=0.5,
        #     ignore_x=None
        # ),
        dict(
            name='Resize',
            size_image=(512, 512),
            size_label=(512, 512),
            ignore_x=None
        ),
        dict(
            name='ColorJitter',
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15
        ),
        dict(
            name='ToTensor'
        ),
        dict(
            name='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            normalize_target=True,
            ignore_x=None
        )
    ]
)
###########################################################################################################
test_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='Resize',
            size_image=(512, 512),
            size_label=(512, 512)
        ),
        dict(
            name='ToTensor'
        ),
        dict(
            name='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)
###########################################################################################################
loss = dict(
    name='HungarianBezierLoss',
    weight=[0.4, 1],
    weight_seg=[0.4, 1],
    curve_weight=1,
    label_weight=0.1,
    seg_weight=0.75,
    color_weight=1,
    alpha=0.8,
    num_sample_points=100,
    bezier_order=3,
    k=9,  # for the local maximum prior
    reduction='mean',
    ignore_index=255
)
###########################################################################################################
lr = 0.0006
optimizer = dict(
    name='torch_optimizer',
    torch_optim_class='Adam',
    lr=lr,
    parameters=[
        dict(
            params='conv_offset',
            lr=lr * 0.1  # 1/10 lr for DCNv2 offsets
        ),
        dict(
            params='__others__'
        )
    ]
)
###########################################################################################################
lr_scheduler = dict(
    name='CosineAnnealingLRWrapper',
    epochs=30
)
###########################################################################################################
import datetime
now = datetime.datetime.now().strftime('%Y%m%d_%H_clok')
train = dict(
    exp_name='resnet34_bezierlanenet_liquid_loss_ablation'+now,
    workers=10,
    batch_size=48,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./debug',

    input_size=(512, 512),
    original_size=(512, 512),
    num_classes=None,
    num_epochs=30,
    collate_fn='dict_collate_fn',  # 'dict_collate_fn' for LSTR
    seg=False,  # Seg-based method or not
)
assert lr_scheduler['epochs'] == train['num_epochs'], 'please make sure that epochs in <lr_scheduler> and <train> are same!'
###########################################################################################################
test = dict(
    exp_name='resnet34_bezierlanenet_liquid_for_release',
    workers=0,
    batch_size=1,
    checkpoint='checkpoints/resnet34_bezierlanenet_liquid_color-aug220231027_22_clok/model.pt',
    # Device args
    device='cuda',

    save_dir='./debug',

    seg=False,
    gap=20,
    ppl=18,
    thresh=None,
    collate_fn='dict_collate_fn',  # 'dict_collate_fn' for LSTR
    input_size=(512, 512),
    original_size=(512, 512),
    max_lane=1,
    dataset_name='liquid'
)

###########################################################################################################
model = dict(
    name='BezierLaneNet',
    image_height=512,
    num_regression_parameters=8,  # 3 x 2 + 2 = 8 (Cubic Bezier Curve)

    # Inference parameters
    thresh=0.95,
    local_maximum_window_size=9,

    # Backbone (3-stage resnet (no dilation) + 2 extra dilated blocks)
    backbone_cfg=dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet34',
        return_layer='layer3',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False]
    ),
    reducer_cfg=None,  # No need here
    dilated_blocks_cfg=dict(
        name='predefined_dilated_blocks',
        in_channels=256,
        mid_channels=64,
        dilations=[4, 8]
    ),

    # Head, Fusion module
    feature_fusion_cfg=dict(
        name='FeatureFlipFusion',
        channels=256
    ),
    head_cfg=dict(
        name='ConvProjection_1D',
        num_layers=2,
        in_channels=256,
        bias=True,
        k=3
    ),  # Just some transforms of feature, similar to FCOS heads, but shared between cls & reg branches

    # Auxiliary binary segmentation head (automatically discarded in eval() mode)
    aux_seg_head_cfg=dict(
        name='SimpleSegHead',
        in_channels=256,
        mid_channels=64,
        num_classes=1
    ),

    colornet_cfg = dict(
    name='ColorNet',
    backbone_cfg = dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet18',
        return_layer={'layer1':'layer1', 'layer2':'layer2', 'layer3':'layer3'},
        pretrained=True,
        replace_stride_with_dilation=[False, False, False]
    )
)
)
###########################################################################################################
