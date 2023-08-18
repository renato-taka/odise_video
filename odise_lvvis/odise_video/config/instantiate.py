# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

from detectron2.config import instantiate

#cfg 其实是cfg.model
def instantiate_odise(cfg):
    backbone = instantiate(cfg.backbone)
    cfg.sem_seg_head.input_shape = backbone.output_shape()
    #input shape为backbone的输出，格式为一个字典，key为s3.Value为shapespc
    cfg.sem_seg_head.pixel_decoder.input_shape = backbone.output_shape()
    cfg.backbone = backbone
    #这时候这个backbone为一个实例化的backbone
    model = instantiate(cfg)

    return model
