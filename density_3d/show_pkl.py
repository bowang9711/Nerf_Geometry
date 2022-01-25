import os, sys
import pickle
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

os.chdir('./density_3d')
dr = os.getcwd()
print(dr)

path = 'archive/data.pkl'


f = open(path, 'rb')
data = pickle.load(f)
print(data)