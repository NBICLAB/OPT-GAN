import torch
from OPTGAN_model import *
from Variable import *
import matplotlib
import csv
import os
import numpy as np
import time

matplotlib.use("Agg")


def InitLOGGING(save_csv, id):
    result_csv = open(save_csv + "/" + id + ".csv", "w", newline="")
    result_csv_writer = csv.writer(result_csv)
    result_csv_writer.writerow(
        (
            "iter",
            "fes",
            "curr_bestf",
            "bestf",
            "time",
            "distance_history",
            "distance_gen",
            "distance_train",
            "d1_loss",
            "d1_real",
            "d1_fake",
            "d1_Wdistance",
            "d2_loss",
            "d2_real",
            "d2_fake",
            "d2_Wdistance",
            "g_loss",
            "g_d1_loss",
            "g_d2_loss",
            "spread_history",
            "spread_gen",
            "spread_train",
            "best_x",
        )
    )
    result_csv.close()


def LOGGING(save_csv, id, values):
    result_csv = open(save_csv + "/" + id + ".csv", "a+", newline="")
    result_csv_writer = csv.writer(result_csv)
    result_csv_writer.writerow(values)
    result_csv.close()


def genSavePath(id, config_wandb):
    return (
        "./data_save/"
        + id
        + "lambda"
        + str(config_wandb.lambda2)
        + "_optset"
        + str(config_wandb.opt_size)
        + "_pop"
        + str(config_wandb.pop_size)
        + "_gamma"
        + str(config_wandb.gamma)
    )


def OPTGAN_min(
    problem, config_init, config_wandb, problem_id, seed, landx=None, landy=None
):
    run_time_start = time.time()
    DIMENSION = config_wandb.dimension
    evaluations = 0
    lambda1 = config_init.lambda1
    lambda2 = config_init.lambda2
    pop_size = config_init.pop_size
    maxfun = int(config_wandb.MAXFes / pop_size + 1)
    init_size = config_init.opt_size
    para_dict = Dict(
        {
            "init_size": init_size,
            "upper": config_wandb.upper,
            "lower": config_wandb.lower,
            "ADD_SIZE": pop_size,
            "Gnodes_in": int(DIMENSION * 2),
            "DIMENSION": DIMENSION,
            "D_iter": config_init.D_iter,
            "epoch": config_init.epochs,
            "pretrain": config_init.pretrain,
            "LatentDim": config_init.latent_dim,
            "Lambda1": lambda1,
            "Lambda2": lambda2,
        }
    )
    bestf = np.zeros(maxfun)
    curr_bestf = np.zeros(maxfun)
    spread_train, spread_history, spread_gen = [list() for _ in range(3)]
    distance_train, distance_history, distance_gen = [[] for _ in range(3)]
    index = []

    optgan_module = OPTGANMD(
        config_init.batch_size,
        DIMENSION,
        para_dict.Gnodes_in,
        config_init.gen_lr,
        config_init.reg_lr,
        config_init.D_nodes,
        config_init.G_nodes,
    )
    # optgan_module.weights_init()

    save_csv = genSavePath(problem_id, config_wandb)
    os.makedirs(save_csv, exist_ok=True)
    figurepath = save_csv

    if ENABLE_LOGGING == True:
        InitLOGGING(save_csv, problem_id)

    data_history, fitness_history = init_history(problem, para_dict)
    evaluations += para_dict.init_size
    bestf_init = min(fitness_history)
    max_index = np.argmin(fitness_history)
    bestpoint_x = data_history[max_index, :]

    if ENABLE_LOGGING:
        LOGGING(
            save_csv,
            problem_id,
            [
                0,
                evaluations,
                bestf_init[0],
                bestf_init[0],
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                bestpoint_x,
            ],
        )

    train_data = data_history
    if En_SCALE:
        _, data_mean, data_std = Convert(data_history).zscore()
    else:
        data_mean = 0
        data_std = 1

    stop = False
    string = ""
    mean_bestf = 1e8

    if ENABLE_LOGGING:
        print(
            "epoch=%d, fes=%d,curr_bestf=%.3e,bestf=%.3e,bestpoint_x=%s\n"
            % (0, evaluations, bestf_init, bestf_init, bestpoint_x,)
        )

    train_data_torch = torch.Tensor(train_data).to(device)

    upper = (para_dict.upper - data_mean) / data_std
    lower = (para_dict.lower - data_mean) / data_std

    # pretrain
    for i in range(para_dict.pretrain):
        optgan_module.pretrain(
            para_dict.epoch, upper, lower,
        )

    fin_size = config_init.MAXFes
    for epoch in range(maxfun):
        ## Optimal Set Shrink
        top_size = int(
            np.ceil(
                init_size
                * np.exp(
                    config_init.gamma * np.log(1 / init_size) * (evaluations / fin_size)
                )
            )
        )
        setInd = np.argsort(fitness_history.reshape(-1))[:top_size]
        fitness_history = fitness_history[setInd]
        data_history = data_history[setInd, :]
        train_data = data_history

        if En_SCALE:
            _, data_add_mean, data_add_std = Convert(data_history).zscore()
            # _, data_add_mean, data_add_std = Convert(train_data).zscore()/
            data_mean = data_mean + config_wandb.ZSCORESTEP * (
                data_add_mean - data_mean
            )
            data_std = data_std + config_wandb.ZSCORESTEP * (data_add_std - data_std)

        train_score = (train_data - data_mean) / data_std

        start = time.time()
        train_data_torch = torch.Tensor(train_score).to(device)

        ## train G and D
        upper = (para_dict.upper - data_mean) / data_std
        lower = (para_dict.lower - data_mean) / data_std
        NN_dict = optgan_module.train(
            train_data_torch,
            para_dict.epoch,
            para_dict.D_iter,
            upper,
            lower,
            lambda1,
            lambda2,
        )
        end = time.time()
        if epoch == 0:
            print("train_time", end - start)

        ## generate new candidate and update Optimal Set
        start = time.time()
        noise = torch.rand(pop_size, para_dict.Gnodes_in, device=device) * 2 - 1
        z_fake = optgan_module.generator(noise)
        fake_data = Convert(z_fake.data).to_np()
        fake_data = fake_data * data_std + data_mean

        fake_fitness = fitnessf(problem, fake_data)
        evaluations += pop_size
        data_history, fitness_history, train_data, index_epoch = update_data_compare(
            fake_data,
            fake_fitness,
            data_history,
            fitness_history,
            train_data,
            para_dict.ADD_SIZE,
        )
        end = time.time()
        if epoch == 0:
            print("updata_time", end - start)

        start = time.time()
        bestf[epoch] = np.min(fitness_history)
        curr_bestf[epoch] = np.min(fake_fitness)

        distance_train.append(min_distance(train_data, para_dict.init_size))
        distance_history.append(min_distance(data_history, para_dict.init_size))
        distance_gen.append(min_distance(fake_data, pop_size))
        spread_gen.append(np.mean(np.std(fake_data)))
        spread_history.append(np.mean(np.std(data_history)))
        spread_train.append(np.mean(np.std(train_data)))
        index.append(index_epoch)

        max_index = np.argmin(fitness_history)
        bestpoint_x = data_history[max_index, :]
        if evaluations > DIMENSION * 3000:
            mean_bestf = np.abs(
                np.mean(bestf[int((evaluations - DIMENSION * 3000) / pop_size) : epoch])
            )

        figurepath, stop, string = print_flag(
            stop,
            figurepath,
            evaluations,
            bestf[epoch],
            distance_train[epoch],
            mean_bestf,
            DIMENSION,
            config_init,
        )

        if stop == True:
            if ENABLE_LOGGING:
                LOGGING(
                    save_csv,
                    problem_id,
                    [
                        epoch + 1,
                        evaluations,
                        curr_bestf[epoch],
                        bestf[epoch],
                        time.time() - run_time_start,
                        distance_history[epoch],
                        distance_gen[epoch],
                        distance_train[epoch],
                        NN_dict.D1_loss,
                        NN_dict.D1_real,
                        NN_dict.D1_fake,
                        NN_dict.D1_Wasserstein,
                        NN_dict.D2_loss,
                        NN_dict.D2_real,
                        NN_dict.D2_fake,
                        NN_dict.D2_Wasserstein,
                        NN_dict.G_loss,
                        NN_dict.G_D1_loss,
                        NN_dict.G_D2_loss,
                        spread_history[epoch],
                        spread_gen[epoch],
                        spread_train[epoch],
                        bestpoint_x,
                    ],
                )
            print(
                "epoch=%d, fes=%d,curr_bestf=%.3e,bestf=%.3e,distance_history=%.2e,distance_gen=%.2e,distance_train=%.2e,D1_loss=%.3e, D2_loss=%.3e,D1_Wasserstein=%.3e,D1_Wasserstein=%.3e,G_loss=%.3e,bestpoint_x=%s\n"
                % (
                    epoch + 1,
                    evaluations,
                    curr_bestf[epoch],
                    bestf[epoch],
                    distance_history[epoch],
                    distance_gen[epoch],
                    distance_train[epoch],
                    NN_dict.D1_loss,
                    NN_dict.D2_loss,
                    NN_dict.D1_Wasserstein,
                    NN_dict.D2_Wasserstein,
                    NN_dict.G_loss,
                    bestpoint_x,
                )
            )

            return [string, bestf[epoch], distance_history[epoch], evaluations]

        if ENABLE_LOGGING:
            if epoch % 10 == 0 or (bestf[epoch] != bestf[epoch - 1] and epoch > 0):
                LOGGING(
                    save_csv,
                    problem_id,
                    [
                        epoch + 1,
                        evaluations,
                        curr_bestf[epoch],
                        bestf[epoch],
                        time.time() - run_time_start,
                        distance_history[epoch],
                        distance_gen[epoch],
                        distance_train[epoch],
                        NN_dict.D1_loss,
                        NN_dict.D1_real,
                        NN_dict.D1_fake,
                        NN_dict.D1_Wasserstein,
                        NN_dict.D2_loss,
                        NN_dict.D2_real,
                        NN_dict.D2_fake,
                        NN_dict.D2_Wasserstein,
                        NN_dict.G_loss,
                        NN_dict.G_D1_loss,
                        NN_dict.G_D2_loss,
                        spread_history[epoch],
                        spread_gen[epoch],
                        spread_train[epoch],
                        bestpoint_x,
                    ],
                )

                print(
                    "epoch=%d, fes=%d,curr_bestf=%.3e,bestf=%.3e,distance_history=%.2e,distance_gen=%.2e,distance_train=%.2e,D1_loss=%.3e, D2_loss=%.3e,D1_Wasserstein=%.3e,D1_Wasserstein=%.3e,G_loss=%.3e,bestpoint_x=%s\n"
                    % (
                        epoch + 1,
                        evaluations,
                        curr_bestf[epoch],
                        bestf[epoch],
                        distance_history[epoch],
                        distance_gen[epoch],
                        distance_train[epoch],
                        NN_dict.D1_loss,
                        NN_dict.D2_loss,
                        NN_dict.D1_Wasserstein,
                        NN_dict.D2_Wasserstein,
                        NN_dict.G_loss,
                        bestpoint_x,
                    )
                )

        end = time.time()
        if epoch == 0:
            print("save_time", end - start)

