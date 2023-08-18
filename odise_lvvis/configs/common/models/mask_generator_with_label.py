from detectron2.config import LazyCall as L
from detectron2.data import MetadataCatalog

from odise_video.data.build import get_openseg_labels
from odise_video.modeling.meta_arch.odise_video import (
    CategoryODISE_VIDEO,
    ODISEMultiScaleMaskedTransformerDecoder_Video,
    PooledMaskEmbed,
    CategoryEmbed,
    PseudoClassEmbed,
    PoolingCLIPHead,
)
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher
model=L(CategoryODISE_VIDEO)(
    sem_seg_head=L(MaskFormerHead)(
        ignore_value=255,#
        num_classes=40,#change
        pixel_decoder=L(MSDeformAttnPixelDecoder)(
            conv_dim=256,#
            mask_dim=256,#
            norm="GN",#
            transformer_dropout=0.0,#
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            transformer_in_features=["s3","s4","s5"],#
            common_stride=4,
        ),
        loss_weight=1.0,
        transformer_in_features="multi_scale_pixel_decoder",
        transformer_predictor=L(ODISEMultiScaleMaskedTransformerDecoder_Video)(
            class_embed=L(PseudoClassEmbed)(num_classes="${..num_classes}"),
            hidden_dim=256,
            post_mask_embed=L(PooledMaskEmbed)(
                hidden_dim="${..hidden_dim}",
                mask_dim="${..mask_dim}",
                projection_dim="${..projection_dim}",
            ),
            in_channels="${..pixel_decoder.conv_dim}",
            mask_classification=True,
            num_classes="${..num_classes}",
            num_queries="${..num_queries}",
            nheads=8,
            dim_feedforward=2048,
            dec_layer=9,
            pre_norm=False,
            enforce_input_project=False,
            mask_dim=256,
        ),
    ),
    criterion=L(VideoSetCriterion)(
        num_layers="${..sem_seg_head.pixel_decoder.declayers}",
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        num_classes="${..sem_seg_head.num_classes}",
        matcher=L(VideoHungarianMatcher)(
            cost_class="${..class_weight}",
            mask_weight="${..mask_weight}",
            dice_weight="${..dice_weight}",
            num_points="${..num_points}",
        ),
        eos_coef=0.1,
        losses=["labels","masks"],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ),
    category_head=L(CategoryEmbed)(
        clip_model_name="ViT-L-14-336",
        labels=L(get_openseg_labels)(dataset="coco_panoptic",prompt_engineered=True),#需要改
        projection_dim="${..sem_seg_head.transformer_predictor.post_mask_embed.projection_dim}",
        
    ),
    clip_head=L(PoolingCLIPHead)(),
    num_queries=100,
    object_mask_threshold=0.0,
    overlap_threshold=0.8,
    metadata=L(MetadataCatalog.get)(name=""),#要改
    size_divisibility=64,
    sem_seg_postprocess_before_inference=True,
    pixel_mean=[0.0,0.0,0.0],
    pixel_std=[255.0,255.0,255.0],
    instance_on=True,
    semantic_on=False,
    panoptic_on=False,
    test_topk_per_image=100,
    
)    
