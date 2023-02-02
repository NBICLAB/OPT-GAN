import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from Variable import *


class Discriminator(nn.Module):
    def __init__(self, in_nodes, hidden_nodes, out_nodes):
        super(Discriminator, self).__init__()
        modules = []
        in_dim = in_nodes
        for h_dim in hidden_nodes:
            modules.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.LeakyReLU()))
            in_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(in_dim, out_nodes)))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def gradient_penalty(self, x_real, x_fake, LAMBDA=10):
        alpha = torch.rand(x_real.size(), device=x_real.device)
        interpolates = alpha * x_real + ((1 - alpha) * x_fake)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self(interpolates)
        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty


class Generator(nn.Module):
    def __init__(self, in_nodes, hidden_nodes, out_nodes):
        super(Generator, self).__init__()
        modules = []
        in_dim = in_nodes
        for h_dim in hidden_nodes:
            modules.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.LeakyReLU()))
        in_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(in_dim, out_nodes)))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class OPTGANMD:
    def __init__(
        self, batch_size, x_dim, z_dim, gen_lr, reg_lr, D_nodes=[50], G_nodes=[50],
    ):  
        print(z_dim)
        self.generator = Generator(z_dim, G_nodes, x_dim)
        self.discriminator = Discriminator(  # Exploitation Discriminator
            x_dim, D_nodes, 1
        )
        self.generator_discriminator = Discriminator(  # Exploration Discriminator
            x_dim, D_nodes, 1
        )

        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=gen_lr)
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=reg_lr)
        self.gd_optim = torch.optim.Adam(
            self.generator_discriminator.parameters(), lr=reg_lr
        )

        self.batch_size = batch_size
        self.x_dim = x_dim
        self.z_dim = z_dim

    def train(self, data, epochs, d_iter, upper, lower, lambda1=0, lambda2=0):
        for epoch in range(epochs):
            for iters in range(d_iter):
                samInd = np.random.choice(
                    a=data.size()[0], size=self.batch_size, replace=True
                )
                real_data = data[samInd, :]
                z_data = torch.rand(self.batch_size, self.z_dim) * 2 - 1
                fake_data = self.generator(z_data).data

                ## train Exploration Discriminator
                self.gd_optim.zero_grad()
                gd_uniform_data = torch.rand_like(real_data) * (upper - lower) + lower
                gd_real_loss = self.generator_discriminator(gd_uniform_data)
                gd_fake_loss = self.generator_discriminator(fake_data)
                gd_gp_loss = self.generator_discriminator.gradient_penalty(
                    gd_uniform_data, fake_data
                )
                GD_LOSS = (
                    -torch.mean(gd_real_loss) + torch.mean(gd_fake_loss) + gd_gp_loss
                )
                GD_LOSS.backward()
                self.gd_optim.step()

                ## train Exploitation Discriminator
                d_real_loss = self.discriminator(real_data)
                d_fake_loss = self.discriminator(fake_data)
                d_gp_loss = self.discriminator.gradient_penalty(real_data, fake_data)
                D_LOSS = -torch.mean(d_real_loss) + torch.mean(d_fake_loss) + d_gp_loss

                self.d_optim.zero_grad()
                D_LOSS.backward()
                self.d_optim.step()

            ## train Generator
            self.g_optim.zero_grad()
            z_data = torch.rand(self.batch_size, self.z_dim) * 2 - 1
            fake_data = self.generator(z_data)
            g_d_loss = self.discriminator(fake_data)
            g_gd_loss = self.generator_discriminator(fake_data)
            G_LOSS = -torch.mean(g_d_loss) - lambda2 * torch.mean(g_gd_loss)
            G_LOSS.backward()
            self.g_optim.step()

        NN_dict = Dict(
            {
                "D1_loss": D_LOSS.item(),
                "D1_real": torch.mean(d_real_loss).item(),
                "D1_fake": torch.mean(d_fake_loss).item(),
                "D1_Wasserstein": (
                    torch.mean(d_real_loss) - torch.mean(d_fake_loss)
                ).item(),
                "D2_loss": GD_LOSS.item(),
                "D2_real": torch.mean(gd_real_loss).item(),
                "D2_fake": torch.mean(gd_fake_loss).item(),
                "D2_Wasserstein": (
                    torch.mean(gd_real_loss) - torch.mean(gd_fake_loss)
                ).item(),
                "G_loss": G_LOSS.item(),
                "G_D1_loss": torch.mean(g_d_loss).item(),
                "G_D2_loss": torch.mean(g_gd_loss).item(),
            }
        )
        return NN_dict

    def pretrain(self, epochs, upper, lower):
        for epoch in range(epochs):
            z_data = torch.rand(self.batch_size, self.z_dim) * 2 - 1
            fake_data = self.generator(z_data).data

            ## train Exploration Discriminator
            self.gd_optim.zero_grad()
            gd_uniform_data = (
                torch.rand((self.batch_size, self.x_dim)) * (upper - lower) + lower
            )
            gd_real_loss = self.generator_discriminator(gd_uniform_data)
            gd_fake_loss = self.generator_discriminator(fake_data)
            gd_gp_loss = self.generator_discriminator.gradient_penalty(
                gd_uniform_data, fake_data
            )
            GD_LOSS = -torch.mean(gd_real_loss) + torch.mean(gd_fake_loss) + gd_gp_loss
            GD_LOSS.backward()
            self.gd_optim.step()

            ## train Generator
            self.g_optim.zero_grad()
            z_data = torch.rand(self.batch_size, self.z_dim) * 2 - 1
            fake_data = self.generator(z_data)
            g_gd_loss = self.generator_discriminator(fake_data)
            G_LOSS = -torch.mean(g_gd_loss)
            G_LOSS.backward()
            self.g_optim.step()

    def weights_init(self):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.2)
                m.bias.data.zero_()

        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)
        self.generator_discriminator.apply(weights_init)
        [eld.apply(weights_init) for eld in self.encoder_latent_discriminators]


def update_data_compare(
    fake_data, fake_fitness, fake_expression, data, fitness, expression, ADD_SIZE
):
    data_samples = np.size(fake_data, 0)
    fake_fitness_clear, index_fake = np.unique(fake_fitness, return_index=True)
    fake_fitness_clear = np.expand_dims(fake_fitness_clear, axis=1)
    index = 0
    for i in range(min(ADD_SIZE, len(fake_fitness_clear))):
        if np.max(fitness) > fake_fitness_clear[i]:
            index += 1
            min_data_index = np.argwhere(fake_fitness == fake_fitness_clear[i])
            max_data_index = np.argwhere(fitness == np.max(fitness))
            data[max_data_index[0, 0], :] = fake_data[min_data_index[0, 0], :]
            expression[max_data_index[0, 0], :] = fake_expression[
                min_data_index[0, 0], :
            ]
            fitness[max_data_index[0, 0]] = fake_fitness[min_data_index[0, 0]]

    return data, fitness, expression


def update_data_compare_init(fake_data, fake_fitness, data, fitness, epoch):
    ADD_SIZE = fake_data.size()[0]
    data[(epoch * ADD_SIZE) : ((epoch + 1) * ADD_SIZE), :] = fake_data
    fitness[(epoch * ADD_SIZE) : ((epoch + 1) * ADD_SIZE), :] = fake_fitness
    return data, fitness


def print_flag(
    stop, figurepath, evaluations, bestf, distance, mean_bestf, DIMENSION, config
):
    string = ""
    if evaluations >= config.MAXFes:
        stop = True
        print("Equal MAXFes,fitness=%.2e" % (bestf))
        figurepath = figurepath + "/fail_fes" + str(bestf)
        string = "fail_fes"
    return figurepath, stop, string
