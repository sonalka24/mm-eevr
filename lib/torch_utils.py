import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


# ============================================================
# Device
# ============================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Dense layer with Keras defaults
# glorot uniform kernel, zero bias (torch Linear default init differs)
# ============================================================
def dense(in_features,out_features):
    layer = nn.Linear(in_features,out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


# ============================================================
# Diagonal Gaussian, same as tfd.MultivariateNormalDiag
# event dim = last dim, log_prob / kl sum over it
# ============================================================
def mvn_diag(loc,scale_diag):
    return Independent(Normal(loc=loc,scale=scale_diag),1)


# ============================================================
# Utility: numpy / dict input -> torch tensors on device
# float arrays -> float32, integer arrays (token ids) -> int64
# ============================================================
def to_tensor(x,device):
    if isinstance(x,dict):
        return {k:to_tensor(v,device) for k,v in x.items()}

    if isinstance(x,torch.Tensor):
        return x.to(device)

    x = np.asarray(x)

    if np.issubdtype(x.dtype,np.integer):
        return torch.as_tensor(x,dtype=torch.long,device=device)

    return torch.as_tensor(x,dtype=torch.float32,device=device)


def index_batch(x,idx):
    if isinstance(x,dict):
        return {k:index_batch(v,idx) for k,v in x.items()}

    return x[idx]


# ============================================================
# Mini-batch loader, replaces
# tf.data.Dataset.from_tensor_slices((X,y)).shuffle(len(y),seed=seed).batch(batch_size)
# reshuffled every epoch, reproducible for a given seed
# ============================================================
class BatchLoader:
    def __init__(self,X,y,batch_size,seed,device):
        self.X          = to_tensor(X,device)                       # dict of tensors, already on device
        self.y          = to_tensor(np.asarray(y).astype("int64"),device)
        self.batch_size = batch_size
        self.n          = len(self.y)
        self.generator  = torch.Generator().manual_seed(int(seed))  # shuffle order per epoch

    def __len__(self):
        return (self.n + self.batch_size - 1)//self.batch_size

    def __iter__(self):
        perm = torch.randperm(self.n,generator=self.generator).to(self.y.device)

        for start in range(0,self.n,self.batch_size):
            idx = perm[start:start+self.batch_size]
            yield index_batch(self.X,idx),self.y[idx]


# ============================================================
# Optimizer parameter groups
# BERT params get their own lr, everything else uses the optimizer lr
# ============================================================
def param_groups(params,bert_params,bert_lr):
    bert_ids  = set([id(p) for p in bert_params])
    non_bert  = [p for p in params if id(p) not in bert_ids]

    groups = [{"params":non_bert}]

    if len(bert_params)>0:
        groups.append({"params":list(bert_params),"lr":bert_lr})    # effective BERT lr = bert_lr

    return groups
