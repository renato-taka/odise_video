from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import DatasetMapper

from mask2former_video import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)


dataloader = OmegaConf.create()


#augment
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(
        dataset_names="lvvis_train", filter_empty=False,
        proposal_files=None
    ),
    mapper=L(YTVISDatasetMapper)(
        is_train=True,
        augmentations=[],
        image_format="RGB",
        use_instance_mask=True,
        sampling_frame_num=2,
        sampling_frame_range=20,
        sampling_frame_shuffle=False,
        num_classes=1212,
        # COCO LSJ aug
    ),
    filter_empyt=False,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    
    dataset=L(get_detection_dataset_dicts)(
        dataset_names="lvvis_val",
        filter_empty=False,
        proposal_files=None,
    ),
    mapper=L(YTVISDatasetMapper)(
        is_train=True,
        augmentations=[],
        image_format="RGB",
        use_instance_mask=True,
        sampling_frame_num=2,
        sampling_frame_range=20,
        sampling_frame_shuffle=False,
        num_classes=1212,
    ),
    num_works=4,
)

dataloader.evaluator = L(YTVISEvaluator)(
    dataset_name="lvvis_val",
    distributed=True,
    output_dir=None,
    )



    

