from typing import Final

from src.models.gnn_models import TestGNN
from src.models.test_model import TestModel

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "test": TestModel,
    "test_gnn": TestGNN
}
