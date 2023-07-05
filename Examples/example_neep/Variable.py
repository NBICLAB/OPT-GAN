import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch, random
import os, time
import numpy as np
from scipy.spatial.distance import pdist
from anytree import Node
import csv
import argparse


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
#
parser.add_argument("--func_ins", type=str, default="0")
parser.add_argument("--func_id", type=str, default="concrete")
parser.add_argument("--func_alg", type=str, default="OPT-GAN")
parser.add_argument("--maxfes", type=int, default=100000)
args = parser.parse_args()


# para: machine CPU/GPU
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
    latent_dim=5,
    encoder_num=0,
    lambda1=1,
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

datapath = "PUT_MY_BBOB_DATA_PATH"
opts = dict(
    algid="ALG_NAME", comments="PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC"
)


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


def loss(model, z, problem):
    x = Convert(model.decoder(Convert(z).to_var())).to_np()
    y, z = problem(x)
    return y, z


def rank_weights(properties, k_val):
    if np.isinf(k_val):
        return np.ones_like(properties)
    # s = np.argsort(-1 * properties, axis=0)
    s = np.argsort(properties, axis=0)
    ranks = np.argsort(s, axis=0)
    weights = 1.0 / (k_val * len(properties) + ranks)
    return weights


def reduce_weight_variance(weights, data, fitness, expression):
    weights_new = []
    data_new = []
    fitness_new = []
    expression_new = []
    for w, d, f, e in zip(weights, data, fitness, expression):
        if w == 0.0:
            continue
        while w > 1:
            weights_new.append(1.0)
            data_new.append(d)
            fitness_new.append(f)
            expression_new.append(e)
            w -= 1
        weights_new.append(w)
        data_new.append(d)
        fitness_new.append(f)
        expression_new.append(e)

    return (
        np.array(weights_new, dtype=float),
        np.array(data_new),
        np.array(fitness_new),
        np.array(expression_new),
    )


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


PROBLEM = args.my_func
path_all = "./"

with open(path_all + "datasets/train/" + PROBLEM + ".txt", "r") as f:
    lines = f.readlines()
    first_line = lines[0]
    size_x = len(first_line.split())
    print("x_size:", size_x)

# GEP parameter
GEP_TERMINAL_NODE = ["x" + str(i) for i in range(1, size_x)]
GEP_FUNCTION_NODE = {
    "+": {"args": 2, "f": lambda x, y: x + y},
    "-": {"args": 2, "f": lambda x, y: x - y},
    "*": {"args": 2, "f": lambda x, y: x * y},
    "/": {"args": 2, "f": lambda x, y: x / (y + 1e-2)},
    "s": {"args": 1, "f": lambda x: np.sin(x)},
    "c": {"args": 1, "f": lambda x: np.cos(x)},
    # "e": {"args": 1, "f": lambda x: np.exp(x)},
    "l": {"args": 1, "f": lambda x: np.log(np.abs(x + 1e-2))},
}
GEP_NODE = [node for node in GEP_FUNCTION_NODE] + GEP_TERMINAL_NODE
GEP_FUNCTION_NUMBER = len(GEP_FUNCTION_NODE)
GEP_TERMINAL_NUMBER = len(GEP_TERMINAL_NODE)
GEP_NODE_NUMBER = GEP_FUNCTION_NUMBER + GEP_TERMINAL_NUMBER
GEP_MAX_OPERATOR = 2
GEP_GENE_HEAD_LEN = 30
GEP_GENE_TAIL_LEN = GEP_GENE_HEAD_LEN * (GEP_MAX_OPERATOR - 1) + 1
GEP_GENE_LEN = GEP_GENE_HEAD_LEN + GEP_GENE_TAIL_LEN

fp = open(path_all + "datasets/train/" + PROBLEM + ".txt", "r")
TRAIN_FITNESS_NUMBER = len(fp.readlines())
print("train_length = ", TRAIN_FITNESS_NUMBER)
fp.close()

fp = open(path_all + "datasets/test/" + PROBLEM + "_test.txt", "r")
TEST_FITNESS_NUMBER = len(fp.readlines())
print("test_length = ", TEST_FITNESS_NUMBER)
fp.close()

# TRAIN_FITNESS_NUMBER = 20
# TEST_FITNESS_NUMBER = 20
# Neural Netqork Parameter
HIDDEN = 40
OUTPUT = GEP_NODE_NUMBER
DIMENSION = HIDDEN * OUTPUT + HIDDEN
INIT_MAX = 2
INIT_MIN = -2
NEURON_STEPS = 10
INIT_NEURON_STEPS = 10
print("x:", GEP_TERMINAL_NODE)
print("hidden:%d,output:%d" % (HIDDEN, OUTPUT))
print("dim:", DIMENSION)
print("func:", PROBLEM)

## Evaluation Algorithm Parameter
config_init = Dict(hyperparameter_defaults)
config_wandb = Dict(
    {
        "upper": INIT_MAX,
        "lower": INIT_MIN,
        "lambda2": config_init.lambda2,
        "opt_size": config_init.opt_size,
        "pop_size": config_init.pop_size,
        "gamma": config_init.gamma,
        "MAXFes": config_init.MAXFes,
        "dimension": DIMENSION,
    }
)
maxfun = int(config_wandb.MAXFes / config_init.pop_size)
para_dict = Dict(
    {
        "init_size": config_init.opt_size,
        "upper": config_wandb.upper,
        "lower": config_wandb.lower,
        "ADD_SIZE": config_init.pop_size,
        "Gnodes_in": int(DIMENSION * 2),
        "DIMENSION": DIMENSION,
        "D_iter": config_init.D_iter,
        "epoch": config_init.epochs,
        "LatentDim": config_init.latent_dim,
        "Lambda1": config_init.lambda1,
        "Lambda2": config_init.lambda2,
        "pretrain": config_init.pretrain,
    }
)


def load_train_data(problem):
    with open(path_all + "datasets/train/" + problem + ".txt", "r") as f:
        fitness_cases = []
        s = f.read().split()
        for i in range(0, int(len(s)), GEP_TERMINAL_NUMBER + 1):

            d = dict()
            k1 = i
            k2 = k1 + GEP_TERMINAL_NUMBER
            for j in range(GEP_TERMINAL_NUMBER):
                d[GEP_TERMINAL_NODE[j]] = eval(s[k1])
                k1 += 1
            case = (d, eval(s[k2]))
            k2 += GEP_TERMINAL_NUMBER + 1
            fitness_cases.append(case)
        f.close()
    return fitness_cases


def load_test_data(problem):
    with open(path_all + "datasets/test/" + problem + "_test.txt", "r") as f:
        fitness_cases = []
        s = f.read().split()
        for i in range(0, int(len(s)), GEP_TERMINAL_NUMBER + 1):

            d = dict()
            k1 = i
            k2 = k1 + GEP_TERMINAL_NUMBER
            for j in range(GEP_TERMINAL_NUMBER):
                d[GEP_TERMINAL_NODE[j]] = eval(s[k1])
                k1 += 1
            case = (d, eval(s[k2]))
            k2 += GEP_TERMINAL_NUMBER + 1
            fitness_cases.append(case)
        f.close()
    return fitness_cases


fitness_cases = load_train_data(PROBLEM)


def caculate(expression_tree, terminal_values):
    def inorder(start):
        nonlocal terminal_values
        if start.name in GEP_TERMINAL_NODE:
            return terminal_values[start.name]
        if start.name in GEP_FUNCTION_NODE:
            return GEP_FUNCTION_NODE[start.name]["f"](
                *[inorder(node) for node in start.children]
            )

    try:
        return inorder(expression_tree)
    except ZeroDivisionError:
        return 1e10


def evaluate(W, B, indiv):
    expression = []
    error = []
    i = 0
    if isinstance(indiv, list):
        indiv = np.array(indiv)
    if indiv.ndim == 1:
        indiv = indiv.reshape(1, -1)
    for ind in indiv:
        expression.append(encode(np.array(ind), W, B))
        error.append(decode(expression[i]))
        i += 1
    return np.array(error), np.array(expression)


def activation_func(x):
    return np.exp(-1 * (x ** 2))


def encode(indiv, W, B):
    y = np.zeros((1, HIDDEN))
    layers = NEURON_STEPS
    length = 0
    expression = list()
    for path in range(GEP_GENE_LEN + INIT_NEURON_STEPS):
        for s in range(layers):
            net = np.matmul(y, W)
            y = activation_func(net + B)
        Wo = indiv.reshape((HIDDEN, OUTPUT + 1))
        o = activation_func(np.matmul(y, Wo))
        output = o[0]
        if path >= INIT_NEURON_STEPS:
            if length < GEP_GENE_HEAD_LEN:
                max_index = np.argmax(output[0:GEP_NODE_NUMBER])
                ch = GEP_NODE[max_index]
                position = np.round(output[-1] * length + 1)
            else:
                max_index = np.argmax(
                    output[
                        GEP_FUNCTION_NUMBER : GEP_FUNCTION_NUMBER + GEP_TERMINAL_NUMBER
                    ]
                )
                ch = GEP_TERMINAL_NODE[max_index]
                position = np.round(
                    output[-1] * (length - GEP_GENE_HEAD_LEN + 1) + GEP_GENE_HEAD_LEN
                )
            if len(expression) == 0 or np.isnan(position):
                expression.append(ch)
                length += 1
            else:
                if position < 1:
                    expression = [ch] + expression[0:]
                    length += 1
                elif position > len(expression) - 1 or len(expression) == 1:
                    expression = expression[0:] + [ch]
                    length += 1
                else:
                    s1 = expression[0 : int(position)]
                    s2 = expression[int(position) :]
                    expression = s1 + [ch] + s2
                    length += 1
    return expression


def decode(gene, is_test=False):
    def get_args_num(elem):
        return GEP_FUNCTION_NODE[elem]["args"] if elem in GEP_FUNCTION_NODE else 0

    def generate(parent, current_level):
        nonlocal levels
        if current_level < len(levels):
            args_num = get_args_num(parent.name)
            for i in range(args_num):
                current_node = Node(levels[current_level][i], parent)
                generate(current_node, current_level + 1)
                if current_level < len(levels) - 1:
                    levels[current_level + 1] = levels[current_level + 1][
                        get_args_num(current_node.name) :
                    ]

    levels = [gene[0]]
    index = 0
    while index < len(gene) and sum([get_args_num(elem) for elem in levels[-1]]) != 0:
        args_num = sum([get_args_num(elem) for elem in levels[-1]])
        levels.append(gene[index + 1 : index + 1 + args_num])
        index += args_num
    tree = Node(gene[0])
    generate(tree, 1)
    expression_tree = tree
    fitness_sum = 0
    for case in fitness_cases:
        fitness_sum += np.power(caculate(expression_tree, case[0]) - case[1], 2)
    if np.isnan(fitness_sum) or np.isinf(fitness_sum):
        fitness = 1e10
    else:
        if is_test:
            fitness = fitness_sum / TEST_FITNESS_NUMBER
        else:
            fitness = fitness_sum / TRAIN_FITNESS_NUMBER
    if fitness > 1e6:
        fitness = 1e6
    return fitness


def test(
    gene, best_gene, best_x, curr_best_x, ins, algo, fes, start_time, is_train=False
):
    data = decode(gene, is_test=True)
    best_data = decode(best_gene, is_test=True)
    if best_data < data:
        save_data = best_data
        save_x = best_x
    else:
        save_data = data
        best_gene = gene
        save_x = curr_best_x
        beet_x = curr_best_x

    save_gene = [i for i in best_gene]
    if is_train:
        path = path_all + "result/converge/" + algo + "/" + PROBLEM + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + PROBLEM + "_ins" + str(ins) + "_d" + str(DIMENSION) + "_" + algo
        with open(path + "_weight.csv", "a", newline="") as file:
            mywriter = csv.writer(file, delimiter=",")
            mywriter.writerows([save_x])
        with open(path + "_fitness.txt", "a") as f:
            f.write(
                str(fes)
                + " "
                + str(save_data)
                + " "
                + str(time.time() - start_time)
                + " "
                + ",".join(save_gene)
                + "\n"
            )
            f.close()
    else:
        path = path_all + "result/error/"
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + PROBLEM + "_d" + str(DIMENSION) + "_" + algo
        with open(path + "_fitness.txt", "a") as f:
            f.write(
                str(ins)
                + " "
                + str(save_data)
                + " "
                + str(time.time() - start_time)
                + " "
                + ",".join(save_gene)
                + "\n"
            )
            f.close()
        with open(path + "_weight.csv", "a", newline="") as file:
            mywriter = csv.writer(file, delimiter=",")
            mywriter.writerows([save_x])

    return best_gene, best_data, best_x

