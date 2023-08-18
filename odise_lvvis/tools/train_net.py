import logging


from detectron2.utils.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from odise_video.utils.events import CommenMetricPrinter,WandbWriter,WriterStack


PathManager.register_handler(S3PathHandler())

logger=logging.getLogger("odise_video")

def default_writers(cfg):
    if "log_dir" in cfg.train:
        log_dir=cfg.train.log_dir
    else:
        log_dir=cfg.train.output_dir
    
    PathManager.mkdirs(log_dir)
    ret=[
        CommenMetricPrinter(
            cfg.train.max_iter,run_name=osp.join(cfg.train.run_name,cfg.train.run_tag)
        )
    ]