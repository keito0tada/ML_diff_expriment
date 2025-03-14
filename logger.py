import datetime
from logging import getLogger, config
import yaml
import jinja2
import torch
import os
from src.utils.misc import now

NOW = now()
TODAY = datetime.date.today()

with open("src/log/logging.yaml", "r", encoding="utf-8") as f:
    os.makedirs(f"log/{TODAY}", exist_ok=True)
    config.dictConfig(
        yaml.safe_load(
            jinja2.Template(f.read()).render(
                {
                    "filepath": f"log/{TODAY}/{NOW}.log",
                    "info_filepath": f"log/{TODAY}/{NOW}_info.log",
                }
            )
        )
    )
logger_regular = getLogger("regular")
logger_overwrite = getLogger("overwrite")


def cuda_memory_usage(index: int):
    logger_regular.debug(
        f"cuda:{index} | total: {torch.cuda.get_device_properties(index).total_memory/1024/1024/1024}GB, free: {torch.cuda.memory_reserved(index)/1024/1024/1024}GB, used: {torch.cuda.memory_allocated(index)/1024/1024/1024}GB"
    )
