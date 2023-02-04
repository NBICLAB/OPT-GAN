import numpy as np
import warnings
import torch
from Variable import *
from OPTGAN_model import *
import time
import torch
import numpy as np

algo = args.my_alg


def main(RUN_TIMES):
    for ins in RUN_TIMES:
        ins = int(ins)
        setup_seed(ins)
        global fitness_cases
        fitness_cases = load_train_data(PROBLEM)
        best_expression = []

        gen = 0
        best_data = None
        best_gene = None
        best_x = None

        # Init the weight and bias of NEEP network
        W = np.zeros((HIDDEN, HIDDEN))
        B = np.random.uniform(-2, 2, (1, HIDDEN))
        for i in range(HIDDEN):
            for j in range(HIDDEN):
                if np.random.rand() < 0.5:
                    W[i][j] = np.random.uniform(-1, 1)
                else:
                    W[i][j] = 0

        evaluate_ind = lambda x: evaluate(W, B, x)
        print("algo:", algo)
        start_time = time.time()

        if algo == "OPT-GAN":
            optgan_module = OPTGANMD(
                config_init.batch_size,
                DIMENSION,
                para_dict.Gnodes_in,
                config_init.gen_lr,
                config_init.reg_lr,
                config_init.D_nodes,
                config_init.G_nodes,
            )

            ## Init Optimal Set
            data_history = np.random.rand(para_dict.init_size, para_dict.DIMENSION)
            data_history = (
                data_history * (para_dict.upper - para_dict.lower) + para_dict.lower
            )
            fitness_history, expression_history = evaluate_ind(data_history)

            max_index = np.argmin(fitness_history[0 : config_init.pop_size])
            best_x = data_history[max_index, :]
            best_expression = expression_history[max_index]
            best_fitness = fitness_history[max_index]
            curr_fitness = fitness_history[max_index]
            curr_best_x = data_history[max_index, :]
            fitness_cases = load_test_data(PROBLEM)
            best_gene = best_expression
            best_gene, best_data, best_x = test(
                best_expression,
                best_gene,
                best_x,
                curr_best_x,
                ins,
                algo,
                max_index + 1,
                start_time,
                True,
            )
            evaluations = config_init.pop_size
            for i in range(config_init.pop_size, para_dict.init_size):
                curr_fitness = fitness_history[i]
                curr_best_x = data_history[i, :]
                if best_fitness >= curr_fitness:
                    best_fitness = curr_fitness
                    best_expression = expression_history[i]
                    best_gene, best_data, best_x = test(
                        best_expression,
                        best_gene,
                        best_x,
                        curr_best_x,
                        ins,
                        algo,
                        evaluations,
                        start_time,
                        True,
                    )
                evaluations += 1
            fitness_cases = load_train_data(PROBLEM)
            ## finish Init Optimal Set

            print("Run: ", ins + 1)
            print(
                "Generation: %d \t Best Train Fitness: %e "
                % (evaluations, best_fitness)
            )
            print("Generation: %d \t Best Test Fitness: %e " % (evaluations, best_data))
            print("Best Expression: \n", best_gene)
            fin_size = config_init.MAXFes
            lambda2 = para_dict.Lambda2

            ## pretrain
            for i in range(para_dict.pretrain):
                optgan_module.pretrain(
                    para_dict.epoch, para_dict.upper, para_dict.lower,
                )

            while gen < maxfun:
                ## Optimal Set Shrink
                top_size = int(
                    np.ceil(
                        para_dict.init_size
                        * np.exp(
                            config_init.gamma
                            * np.log(1 / para_dict.init_size)
                            * (evaluations / fin_size)
                        )
                    )
                )
                setInd = np.argsort(fitness_history.reshape(-1))[:top_size]
                fitness_history = fitness_history[setInd]
                expression_history = expression_history[setInd, :]
                data_history = data_history[setInd, :]

                ## train G and D
                NN_dict = optgan_module.train(
                    torch.Tensor(data_history),
                    para_dict.epoch,
                    para_dict.D_iter,
                    para_dict.upper,
                    para_dict.lower,
                    config_init.lambda1,
                    config_init.lambda2,
                )

                ## generate new candidate and update
                noise = torch.rand(config_init.pop_size, para_dict.Gnodes_in) * 2 - 1
                z_fake = optgan_module.generator(noise)
                fake_data = Convert(z_fake.data).to_np()
                fake_fitness, fake_expression = evaluate_ind(fake_data)
                evaluations += config_init.pop_size

                data_history, fitness_history, expression_history = update_data_compare(
                    fake_data,
                    np.array(fake_fitness),
                    np.array(fake_expression),
                    data_history,
                    fitness_history,
                    expression_history,
                    para_dict.ADD_SIZE,
                )

                ## output infomation
                best_fitness = min(fitness_history)
                max_index = np.argmin(fitness_history)
                curr_best_x = data_history[max_index, :]
                best_expression = expression_history[max_index]
                fitness_cases = load_test_data(PROBLEM)
                best_gene, best_data, best_x = test(
                    best_expression,
                    best_gene,
                    best_x,
                    curr_best_x,
                    ins,
                    algo,
                    evaluations,
                    start_time,
                    True,
                )
                fitness_cases = load_train_data(PROBLEM)

                print("Run: ", ins + 1)
                print(
                    "Generation: %d \t Best Train Fitness: %e "
                    % (evaluations, best_fitness)
                )
                print(
                    "Generation: %d \t Best Test Fitness: %e "
                    % (evaluations, best_data)
                )
                print("Best Expression: \n", best_gene)
                # Gather all the fitnesses in one list and print the stats
                gen += 1
                if evaluations > config_init.MAXFes or time.time() - start_time > 36000:
                    break

            fitness_cases = load_test_data(PROBLEM)
            best_gene, best_data, best_x = test(
                best_expression,
                best_gene,
                best_x,
                curr_best_x,
                ins,
                algo,
                evaluations,
                start_time,
            )
        print("used time:", time.time() - start_time)


if __name__ == "__main__":
    ins = args.my_ins
    ins = ins.split(",")
    main(ins)
