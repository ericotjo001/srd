import os, argparse, joblib, json, shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.set_printoptions(precision=3)

def average_every_n(x, n=10):
    if not len(x)%n==0:
        x = list(x) + [x[-1] for i in range(n-len(x)%n)]
    x = np.array(x).reshape(-1,n)
    return np.mean(x, axis=1)


