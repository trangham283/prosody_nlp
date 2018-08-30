import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

np.random.seed(0)
seed_from_numpy = np.random.randint(2147483648)
torch.manual_seed(seed_from_numpy)

a = [0, 1, 2, 3, 2, 1]
b = torch.LongTensor(a)
emb = nn.Embedding(10, 7)
print(emb.weight)
c = emb(b)
print(c)

init.normal_(emb.weight)
print(emb.weight)

w = torch.empty(3, 5)
init.uniform_(w)
print(w)
