import torch, random
import os
import numpy as np
import argparse
from scipy.spatial.distance import pdist


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description="manual to this script")

parser.add_argument("--gamma", type=float, default=1.5)
parser.add_argument("--lambda2", type=float, default=0.3)
parser.add_argument("--pop", type=int, default=30)
parser.add_argument("--optset", type=int, default=150)

parser.add_argument("--func_dim", type=int, default="2")
parser.add_argument("--func_ins", type=int, default="0")
parser.add_argument("--func_id", type=str, default="Conformal_Bent_Cigar")
parser.add_argument("--func_alg", type=str, default="OPT-GAN")
parser.add_argument("--maxfes", type=int, default=50000)

args = parser.parse_args()

device = torch.device("cpu")

hyperparameter_defaults = dict(
    # para:function
    bestf_target=1e-8,
    MAXFes=args.maxfes,
    #
    # para: network
    gen_lr=0.0001,
    reg_lr=0.005,
    D_nodes=[50,],
    G_nodes=[50,],
    latent_dim=5, # deprecated
    encoder_num=0, # deprecated
    lambda1=[1], # deprecated
    lambda2=args.lambda2,
    batch_size=30,
    D_iter=4,
    epochs=150,
    pretrain=100,
    #
    # para: optimal set
    pop_size=args.pop,
    opt_size=args.optset,
    gamma=args.gamma,
)

# para:show
En_SCALE = False
ENABLE_LOGGING = True


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Convert(object):
    def __init__(self, samples):
        self.samples = samples

    def to_var(self):
        self.samples = torch.Tensor(self.samples)
        if torch.cuda.is_available():
            self.samples = self.samples.to(device)
        return self.samples

    def to_np(self):
        if self.samples.device == "cpu":
            return self.samples.numpy()
        else:
            return self.samples.data.cpu().numpy()

    def zscore(self):
        x_mean = np.mean(self.samples, 0)
        x_std = np.std(self.samples, 0)
        x_new = (self.samples - x_mean) / x_std
        return x_new, x_mean, x_std

    def reduction(self, para_dict):
        return self.samples * (para_dict.upper - para_dict.lower) + para_dict.lower


def init_history(problem, para_dict):
    fitness_history = np.zeros((para_dict.init_size, 1))
    data_history = np.random.rand(para_dict.init_size, para_dict.DIMENSION)
    data_history = Convert(data_history).reduction(para_dict)
    for i in range(para_dict.init_size):
        fitness_history[i, :] = problem(data_history[i, :])
    return data_history.astype(np.float32), fitness_history.astype(np.float32)


def fitnessf(problem, x):
    size = x.shape[0]
    fitness = np.zeros((size, 1))
    for i in range(size):
        fitness[i, :] = problem(x[i, :])
    return fitness


def min_distance(data, size):
    x = data
    x1 = pdist(x)
    x2 = np.sort(x1)[0:size]
    x3 = np.mean(x2)

    return x3


class Problem:
    def __init__(self, func):
        self.maxFes = 0
        self.solutions = []
        self.fitness = []
        self.func = func

    def __call__(self, x):
        self.maxFes += 1
        self.solutions.append(x)
        fit = self.func(x)
        self.fitness.append(fit)
        return fit
