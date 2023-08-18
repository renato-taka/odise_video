from detectron2.config import LazyConfig,instanctiate
import argparse
from odise_video.data_video import *


parser=argparse.ArgumentParser()
parser.add_argument("--config_file",default="configs/common/data/youtubevis2019")


args=parser.parse_args()

cfg=LazyConfig.load(args.config_file)
train_loader=instanctiate(cfg.dataloader.train)
data_loader_iter=iter(train_loader)
now_data=next(data_loader_iter)
print(len(now_data))