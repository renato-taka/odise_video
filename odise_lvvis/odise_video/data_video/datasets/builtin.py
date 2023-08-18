# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os

from .ytvis import register_ytvis_instances


# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_LVVIS = {
    "lvvis_train": ("LVVIS/train/JPEGImages", "LVVIS/train_instances.json"),
    "lvvis_val": ("LVVIS/val/JPEGImages", "LVVIS/val_instances.json"),
}





def register_all_lvvis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )





if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_lvvis(_root)
