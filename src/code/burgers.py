import math
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt  # 导入Matplotlib
import scipy.io
from torch.autograd import grad
import time
import os
import scipy
from scipy.interpolate import griddata
from pyDOE import lhs
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from pinn import *
from frecpinn import *
np.random.seed(1234)

layers = [2, 50, 50, 50, 50, 1]

device = "cuda" if torch.cuda.is_available() else "cpu"


def testgen():
    data = np.load("Burgers.npz")
    x, t, exact = data["x"], data["t"], data["usol"].T
    x1, t1 = np.meshgrid(x, t)
    x2 = np.vstack((np.ravel(x1), np.ravel(t1))).T
    y = exact.flatten()[:, None]
    return x2, y

x_exact, u_exact = testgen()
x_exact = torch.from_numpy(x_exact).float().to(device)



class Solver:
    def __init__(self, model, layers,  stepx=0.05, stept=0.05):

        self.model = model(layers).to(device)
        self.stepx = stepx
        self.stept = stept


        self.x_inside, self.x_boundary, self.u_boundary = self.create_grids()


        self.mse = nn.MSELoss()

        self.optim = torch.optim.Adam(self.model.parameters())

        self.loss_bcs, self.loss_res, self.loss_values = [], [], []

    def create_grids(self):

        x = torch.arange(-1, 1 + self.stepx, self.stepx)
        t = torch.arange(0, 1 + self.stept, self.stept)


        x_inside = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
        x_inside = x_inside.requires_grad_(True)

        bc_left = torch.stack(torch.meshgrid(x[0], t)).reshape(2, -1).T
        bc_right = torch.stack(torch.meshgrid(x[-1], t)).reshape(2, -1).T
        ic = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T
        x_boundary = torch.cat([bc_left, bc_right, ic])


        u_bcl = torch.zeros(len(bc_left))
        u_bcr = torch.zeros(len(bc_right))
        u_ic = -torch.sin(math.pi * ic[:, 0])
        u_boundary = torch.cat([u_bcl, u_bcr, u_ic]).unsqueeze(1)


        x_inside = x_inside.to(device)
        x_boundary = x_boundary.to(device)
        u_boundary = u_boundary.to(device)


        return x_inside, x_boundary, u_boundary

    def compute_gradients(self, u_inside):
        # 计算梯度
        du_dX = torch.autograd.grad(
            inputs=self.x_inside,
            outputs=u_inside,
            grad_outputs=torch.ones_like(u_inside),
            retain_graph=True,
            create_graph=True
        )[0]

        du_dx = du_dX[:, 0]
        du_dt = du_dX[:, 1]

        # 计算二阶导数
        du_dxx = torch.autograd.grad(
            inputs=self.x_inside,
            outputs=du_dX,
            grad_outputs=torch.ones_like(du_dX),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]

        return du_dx, du_dt, du_dxx

    def loss_func(self):

        self.optim.zero_grad()

        loss_bc = self.compute_boundary_loss()

        loss_res = self.compute_equation_loss()

        total_loss = loss_bc + loss_res
        total_loss.backward()


        self.loss_values.append(total_loss.item())
        self.loss_bcs.append(loss_bc.item())
        self.loss_res.append(loss_res.item())

        return total_loss

    def compute_boundary_loss(self):

        u_pred_bc = self.model(self.x_boundary)

        return self.mse(u_pred_bc, self.u_boundary)

    def compute_equation_loss(self):

        u_inside = self.model(self.x_inside)


        du_dx, du_dt, du_dxx = self.compute_gradients(u_inside)


        equation_loss = self.mse(du_dt + u_inside.squeeze() * du_dx, 0.01 / math.pi * du_dxx)

        return equation_loss

    def train(self,nIter):
        self.model.train()
        t0 = time.time()
        self.model.a =self.model.a.to(device)
        a_initial_value = self.model.a.item()
        growth_rate = 0.0005
        # log loss
        with open("loss_frpinnburg0.1(x0000).txt", "w") as log_file:
            log_file.write("Iteration, Loss, Time, bc, res, l2\n")
            for epoch in tqdm(range(nIter)):

                self.optim.step(self.loss_func)
                # set limitation
                if self.model.a.data < 0.10000:
                    new_a_value = a_initial_value * torch.exp(torch.tensor(growth_rate * epoch).to(device))
                    self.model.a.data = new_a_value



                if epoch % 100 == 0:
                    a_value = self.model.a.item()
                    elapsed_time = time.time() - t0
                    u_pred = self.model(x_exact)
                    u_pred = u_pred.cpu().detach().numpy()
                    rl2 = np.sqrt(np.sum((u_exact - u_pred) ** 2) / np.sum(u_exact ** 2))
                    log_file.write(f"{epoch}, {self.loss_values[epoch]}, {self.loss_bcs[epoch]}, {self.loss_res[epoch]}, {elapsed_time}, {rl2}\n")
                    tqdm.write(f"Iteration={epoch:04d}, loss={self.loss_values[epoch]:.6f}, l2={rl2:.6f}, a={a_value:.5f}")



# train
solver = Solver(frecPINN,layers)
solver.train(40000)

# torch.save(solver.model, 'burgers_fuck_p.pth')

