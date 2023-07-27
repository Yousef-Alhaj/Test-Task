

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


data=pd.read_csv('data\data.csv')
ozone_data = data['OZONE'].values.astype(float)
print(ozone_data)