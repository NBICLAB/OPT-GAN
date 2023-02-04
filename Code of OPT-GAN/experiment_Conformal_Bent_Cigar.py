import numpy as np
from Variable import *
from OPTGAN import OPTGAN_min


def problem_baojiao(samples):
    xd, yd, ud, vd = [1 for _ in range(4)]
    condition = 1e6
    beta = 0.5
    Fopt = 134.5
    Fadd = Fopt
    Xopt = [-3.7504, 2.0728]
    L = 10
    # samples=np.ones((1,2))
    # samples=[2.5,2.5]
    scale_forw, shift_forw, scale_inv, shift_inv = [np.zeros((1, 2)) for _ in range(4)]

    scale_forw[0, 1], scale_forw[0, 0] = 2 * xd / L, -2 * yd / L
    scale_inv[0, 1], scale_inv[0, 0] = 2 * ud / L, -2 * ud / L
    shift_inv[0, 1], shift_inv[0, 0] = -ud, vd

    data = (samples * scale_forw) + shift_forw
    sum = data[:, 0] ** 2 + data[:, 1] ** 2
    data[:, 0] = -data[:, 0] / sum
    data[:, 1] = data[:, 1] / sum
    t2 = (data - shift_inv) / scale_inv

    sum1 = t2[:, 0] ** 2
    Ftrue = condition * sum1 + t2[:, 1] ** 2
    Fval = Ftrue
    Ftrue = Ftrue + Fadd
    Fval = Fval + Fadd
    m = shift_inv[:, 0]
    n = shift_inv[:, 1]
    xp = -m / (m ** 2 + n ** 2)
    yp = n / (m ** 2 + n ** 2)
    x0 = (xp - shift_forw[:, 0]) / scale_forw[:, 0]
    y0 = (yp - shift_forw[:, 1]) / scale_forw[:, 1]
    # 最优点(2.5,-2.5)
    z0 = condition * x0 ** 2 + y0 ** 2
    """fitness = np.zeros((t2.shape[0], 1))
    for i in range(samples.shape[0]):
        fitness[i, :] = problem(t2[i, :])"""
    F = np.squeeze(Fval)
    # F = F-min(F)+1
    F = np.log(F) - np.log(Fadd)
    return F


def run(func, ins, dimension, algo):
    config_init = Dict(hyperparameter_defaults)

    if func == "Conformal_Bent_Cigar":
        problem_now = Problem(problem_baojiao)
        upper = 5
        lower = -5

    config_wandb = Dict(
        {
            "upper": upper,
            "lower": lower,
            #
            # "ADD_SIZE_per": config_init.ADD_SIZE_per,
            "lambda2": config_init.lambda2,
            "opt_size": config_init.opt_size,
            "pop_size": config_init.pop_size,
            "gamma": config_init.gamma,
            "MAXFes": config_init.MAXFes,
            "dimension": dimension,
        }
    )

    fun_id = func + "_i" + str(ins)
    print("--> algorithm %s on function %s ***" % (algo, fun_id + "_" + str(dimension)))
    if algo == "OPT-GAN":  # add solver to investigate here
        fmin = OPTGAN_min(
            problem_now,
            config_init,
            config_wandb,
            fun_id + "_d" + str(dimension),
            setup_seed(int(ins)),
        )


if __name__ == "__main__":
    print(args.maxfes)
    func = args.my_func
    ins = args.my_ins
    dim = args.my_dim
    algo = args.my_alg
    run(func, ins, dim, algo)

