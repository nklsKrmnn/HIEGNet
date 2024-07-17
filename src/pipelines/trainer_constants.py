from typing import Final

from src.pipelines.cnn_trainer import ImageTrainer
from src.pipelines.trainer import Trainer

TRAINER_MAPPING: Final[dict[str, any]] = {
    'graph_trainer': Trainer,
    "image_trainer": ImageTrainer
}
