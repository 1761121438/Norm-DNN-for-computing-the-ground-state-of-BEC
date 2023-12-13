from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad
import math, torch, time, os
import torch.nn as nn
import numpy as np
import argparse
import pandas as pd
import random

random.seed(111)
np.random.seed(111)
torch.manual_seed(111)
torch.cuda.manual_seed(111)
torch.cuda.manual_seed_all(111)

parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--e', type=int, default=50000, help='Epochs')
parser.add_argument('--d', type=int, default=3, help='depth')
parser.add_argument('--n', type=int, default=70, help='width')
parser.add_argument('--beta', type=int, default=400, help='/beta')
parser.add_argument('--inter', type=int, default=12, help='the interval is [-inter, inter]')
parser.add_argument('--nx', type=int, default=128, help='Sampling')
parser.add_argument('--xi', type=float, default=1e-6, help='Threshold')

args = parser.parse_args()


def GetGradients(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True, allow_unused=True)[0]


def PHI_0(x):
    return np.exp(-1 * x ** 2 / 10) / np.pi ** (1 / 4)


def positive(x):
    return x - torch.min(x)


def V(x):
    return 0.5 * x ** 2


def errorFun(output, target, params):
    error = output - target
    error = math.sqrt(torch.mean(error * error))
    ref = math.sqrt(torch.mean(target * target))
    return error / (ref + params["minimal"])


def error0Fun(output, target, params):
    error = output - target
    error = torch.max(abs(error))
    ref = torch.max(abs(target))
    return error / (ref + params["minimal"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class Net(torch.nn.Module):
    def __init__(self, params, device):
        super(Net, self).__init__()
        self.params = params
        self.device = device
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        nn.init.xavier_normal_(self.linearIn.weight)
        nn.init.constant_(self.linearIn.bias, 0)

        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.m = nn.Linear(self.params["width"], self.params["width"])
            nn.init.xavier_normal_(self.m.weight)
            nn.init.constant_(self.m.bias, 0)
            self.linear.append(self.m)

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])
        nn.init.xavier_normal_(self.linearOut.weight)
        nn.init.constant_(self.linearOut.bias, 0)

    def forward(self, X):
        x = torch.tanh(self.linearIn(X))
        for layer in self.linear:
            x_temp = torch.tanh(layer(x))
            x = x_temp
        x = self.linearOut(x)
        x = positive(x)
        x = x / torch.sqrt(torch.sum(x * x) * 2 * self.params["interval"] / self.params["nx"])
        return x


def pre_train(model, device, params, optimizer, scheduler):
    start_time = time.time()
    pre_start_time = time.time()
    Loss = []

    for step in range(params["pre_trainstep"]):
        x = np.linspace(-1 * params["interval"], params["interval"], params["nx"] + 1)[1:-1][:, None]
        # x = 2 * params["interval"] * np.random_sampling.rand(params["nx"], 1) - params["interval"]
        u0 = PHI_0(x)

        X = torch.from_numpy(x).float().to(device)
        U0 = torch.from_numpy(u0).float().to(device)
        U0 = U0 / torch.sqrt(torch.sum(U0 * U0) * 2 * params["interval"] / params["nx"])

        U0_pred = model(X)

        model.zero_grad()

        loss = torch.mean(torch.square(U0_pred - U0))

        if step % params["pre_step"] == 0:
            elapsed = time.time() - start_time
            print('Epoch: %d, Time: %.2f, Loss: %.3e' %
                  (step, elapsed, loss))
            start_time = time.time()
            Loss.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        scheduler.step()
    pre_total_time = time.time() - pre_start_time

    return model, pre_total_time


def train(model, device, params, optimizer, scheduler, pre_time):
    # x_train_total = np.linspace(-1 * params["interval"], params["interval"], params["nt"] + 1)[1:-1][:, None]
    x_train = np.linspace(-1 * params["interval"], params["interval"], params["nx"] + 1)[1:-1][:, None]

    x_test = np.linspace(-1 * params["interval"], params["interval"], params["nt"] + 1)[1:-1][:, None]
    X_test = torch.from_numpy(x_test).float().to(device)

    start_time = time.time()
    total_start_time = start_time
    Loss = []
    Step = []
    Energy = []
    Test = []
    Time = []

    loss_epoch_list = []
    loss_std_list = []
    loss_mean_list = []

    loss_mean_last = 1

    for step in range(params["trainstep"]):
        X = torch.from_numpy(x_train).float().to(device)
        X = X.requires_grad_(True)
        U_pred = model(X)

        model.zero_grad()

        U_x = GetGradients(U_pred, X)[:, 0:1]
        Res = 0.5 * abs(U_x) ** 2 + V(X) * abs(U_pred) ** 2 + params["beta"] * abs(U_pred) ** 4 / 2
        loss_res = torch.sum((2 * params["interval"] / params["nx"]) * Res)
        loss = loss_res

        loss_stop = loss.cpu().detach().numpy()
        loss_epoch_list.append(loss_stop)

        if step % 100 == 0 and step != 0:
            loss_std = np.std(np.array(loss_epoch_list[(step - 100):step]))
            loss_std_list.append(loss_std)
            loss_mean = np.mean(np.array(loss_epoch_list[(step - 100):step]))
            loss_mean_list.append(loss_mean)

            loss_mean_dis = np.sqrt(np.mean(np.square(loss_mean - loss_mean_last))) / np.sqrt(
                np.mean(np.square(loss_mean_last)))
            loss_mean_last = loss_mean

            if abs(loss_mean_dis) < params["xi"] or step == params["trainstep"] - 100:
                total_time = time.time() - total_start_time + pre_time
                print('%% U no longer adapts, training stop')
                print('--------stop_step: %d' % step)
                print('--------final energy: %.3e' % loss_res)
                print("Training costs %s seconds." % (total_time))

                Step.append(step)
                Time.append(total_time)
                break

        params["nx"] = params["nt"]
        X_test = X_test.requires_grad_(True)
        U_pred_test = model(X_test)
        U_x_test = GetGradients(U_pred_test, X_test)[:, 0:1]
        Res_test = 0.5 * abs(U_x_test) ** 2 + V(X_test) * abs(U_pred_test) ** 2 + params["beta"] * abs(
            U_pred_test) ** 4 / 2
        Energy_test = torch.sum((2 * params["interval"] / params["nx"]) * Res_test)
        params["nx"] = args.nx

        if step % params["Writestep"] == 0:
            elapsed = time.time() - start_time

            phi_besp = np.array(pd.read_csv("Phi/Phi_1d_bt{}.csv".format(params["beta"]), header=None)).T
            Phi_besp = torch.from_numpy(phi_besp).float().to(device)

            test_error = errorFun(U_pred_test, Phi_besp, params)
            print('Epoch: %d, Time: %.2f, Loss: %.3e, Energy: %.3e, test: %.3e' %
                  (step, elapsed, loss, Energy_test, test_error))
            start_time = time.time()
            Energy.append(Energy_test.cpu().detach().numpy())
            Loss.append(loss.cpu().detach().numpy())
            Test.append(test_error)  # .cpu().detach().numpy()

        loss.backward()
        optimizer.step()
        scheduler.step()

    U_numerical = U_pred_test.cpu().detach().numpy()

    folder = './Mesh_1d_interval[{l}, {r}]_beta{b}_gauss_Depth{d}_Width{w}_nx{nx}'.format(l=-1 * params["interval"],
                                                                                          r=params["interval"],
                                                                                          b=params["beta"],
                                                                                          d=params["depth"] + 1,
                                                                                          w=params["width"],
                                                                                          nx=args.nx)

    os.mkdir(folder)
    np.savetxt(folder + "/loss.csv", Loss)
    np.savetxt(folder + "/Step.csv", Step)
    np.savetxt(folder + "/Test.csv", Test)
    np.savetxt(folder + "/Time.csv", Time)
    np.savetxt(folder + "/Energy.csv", Energy)
    np.savetxt(folder + "/Numerical.txt", U_numerical)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["d"] = 1  # 2D
    params["interval"] = args.inter
    params["nx"] = args.nx
    params["nt"] = 2048
    params["width"] = args.n  # Width of layers
    params["depth"] = args.d  # Hidden Layer: depth+1
    params["dd"] = 1  # Output
    params["lr"] = 0.001  # Learning rate
    params["beta"] = args.beta
    params["xi"] = args.xi
    params["trainstep"] = args.e
    params["pre_trainstep"] = 1000
    params["Writestep"] = 100
    params["pre_step"] = 100
    params["minimal"] = 10 ** (-14)
    params["step_size"] = 100  # lr decay
    params["gamma"] = 0.99  # lr decay rate
    startTime = time.time()

    model = Net(params, device).to(device)
    print("Generating network costs %s seconds." % (time.time() - startTime))
    print(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])

    startTime = time.time()
    model_pre, pre_time = pre_train(model, device, params, optimizer, scheduler)
    print("Pre-training costs %s seconds." % (time.time() - startTime))

    train(model_pre, device, params, optimizer, scheduler, pre_time)
    print("The number of parameters is %s," % count_parameters(model))


if __name__ == "__main__":
    main()
