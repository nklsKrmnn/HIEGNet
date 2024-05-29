from typing import Final
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer

SCALER_OPTIONS: Final[dict] = {"MinMaxScaler": MinMaxScaler,
                               "StandardScaler": StandardScaler,
                               "QuantileTransformer": QuantileTransformer,
                               "PowerTransformer": PowerTransformer}