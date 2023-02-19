# Imports:

# PyTorch imports:
from torch.utils.data import Dataset

# Pandas imports:
import pandas as pd

# Other imports:
import os


class HousePricesDataset(Dataset):
    def __init__(self, data : pd.DataFrame, labels : pd.Series):
        pass