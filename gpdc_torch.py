import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plot_params = {'legend.fontsize': 18,
               'figure.figsize': (15, 5),
               'xtick.labelsize': '18',
               'ytick.labelsize': '18',
               'axes.titlesize': '24',
               'axes.labelsize': '22'}
plt.rcParams.update(plot_params)


class GPC(nn.Module):

    def __init__(self, kernel_id, l2=1., s2_n=1.,
                 s2=1., m=300, learn_inducing=False):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor(np.log(l2)))
        self.noise_scale = nn.Parameter(torch.tensor(np.log(s2_n)))
        self.amplitude_scale = nn.Parameter(torch.tensor(np.log(s2)))
        self.kernel_id = kernel_id
        self.m = m
        self.learn_inducing = learn_inducing

    def sample_x(self, times_x=torch.linspace(0, 1000, 100), params=[1, 1],
                 how_many=1):
        np.random.seed(1)
        with torch.no_grad():
            if self.kernel_id == 'RBF-RBF':
                cov_x = SE(times_x, times_x, params[0], params[1]) + 0*1e-5*np.eye(len(times_x))
        x = np.random.multivariate_normal(np.zeros((times_x.shape[0],)),
                                          cov_x, how_many)
        return x.T

    def load(self, times_y, y):
        self.times_y = times_y.clone().detach()
        self.y = torch.tensor(y).float()
        self.Ny = len(y)
        #half = (torch.max(self.times_y) - torch.min(self.times_y))/2
        self.u_loc = torch.linspace(torch.min(self.times_y),
                                    torch.max(self.times_y), self.m)
        # test ~ exact inference
        # self.u_loc = torch.tensor(self.times_y).float()
        # self.m = self.u_loc.shape[0]
        if self.learn_inducing:
            self.u_loc = nn.Parameter(self.u_loc)

    def set_filter_params(self, p):
        self.theta_h = p

    def elbo(self):

        if self.kernel_id == 'RBF-RBF':
            l2 = torch.exp(self.length_scale)
            s2 = torch.exp(self.amplitude_scale)
            s2_n = torch.exp(self.noise_scale)

            u_loc = self.u_loc  # torch.linspace(torch.min(self.times_y), torch.max(self.times_y), self.m)
            beta = 1/s2_n
            Y = self.y
            N = self.Ny
            mag_h = self.theta_h[0]
            l2_h = self.theta_h[1]

            s2_fx, l2_fx = RBF_convolution(s2, l2, mag_h, l2_h)
            s2_f, l2_f = RBF_convolution(s2_fx, l2_fx, mag_h, l2_h)

            Kmn = SE(u_loc, self.times_y, s2_fx, l2_fx)
            Kmm = SE(u_loc, u_loc, s2, l2)

            temp1 = Kmn.T@torch.linalg.solve(Kmm + torch.eye(self.m)*1e-8, Kmn)
            temp2 = temp1 + torch.eye(N)*s2_n

            trace_Knn = N*s2_f
            temp3 = beta/2*(trace_Knn - torch.trace(temp1))
            (sign, logdet) = torch.linalg.slogdet(temp2)

            ELBO = (- N/2*np.log(2*np.pi) - 1/2*logdet -
                    1/2*Y.T@torch.linalg.solve(temp2+torch.eye(N)*1e-8, Y) -
                    temp3)
            return -ELBO

    def forward(self, times_x):
        with torch.no_grad():
            # self.ind_loc = torch.linspace(torch.min(self.times_y), torch.max(self.times_y), self.m)
            u_loc = self.u_loc
            l2 = torch.exp(self.length_scale)
            s2_n = torch.exp(self.noise_scale)
            s2 = torch.exp(self.amplitude_scale)
            beta = 1/s2_n
            mag_h = self.theta_h[0]
            l2_h = self.theta_h[1]
            s2_fx, l2_fx = RBF_convolution(s2, l2, mag_h, l2_h)
            # u_loc = self.ind_loc
            m = len(u_loc)

            Kmn = SE(u_loc, self.times_y, s2_fx, l2_fx)
            Kmm = SE(u_loc, u_loc, s2, l2)
            temp1 = torch.linalg.solve(Kmm, Kmn)
            Lambda = torch.linalg.solve(Kmm, beta*Kmn@(temp1.T)
                                        + torch.eye(m))
            u_hat = beta * torch.linalg.solve(Kmm@Lambda, Kmn@self.y)

            Kxm = SE(times_x, u_loc, s2, l2)
            Kxx = SE(times_x, times_x, s2, l2)

            L = torch.linalg.solve(Kmm, Kxm.T)

            res =  L.T@u_hat, Kxx + (torch.linalg.solve(Lambda, L).T - Kxm)@L, u_hat
        return res

    def train_step(self, opt):
        opt.zero_grad()
        l = self.elbo()
        l.backward()
        opt.step()
        loss_dict = {'loss': l.item(),
                     'length': np.exp(self.length_scale.detach().cpu()),
                     'noise': np.exp(self.noise_scale.detach().cpu()),
                     'amplitude': np.exp(self.amplitude_scale.detach().cpu())}

        if self.learn_inducing:
            loss_dict['inducing points'] = self.u_loc.detach().cpu()
        return loss_dict


def outersum(a, b):
    return (torch.outer(a, torch.ones_like(b)) +
            torch.outer(torch.ones_like(a), b))

# TODO: compare new implementation with old one
#def SE(x, y, s2, l2):
#    return s2 * torch.exp(-outersum(x, -y)**2/(2*l2))


def SE(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
    X1: Array of m points (m x d).
    X2: Array of n points (n x d).

    Returns: (m x n) matrix.
    """
    sqdist = torch.sum(X1**2, 1).reshape(-1, 1) + torch.sum(X2**2, 1) - 2 * torch.matmul(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def RBF_convolution(s1, l1, s2, l2):
    # inputs are *square* variance and lengthscales
    s = s1*s2*torch.sqrt(2*np.pi*l1*l2/(l1+l2))
    l = l1 + l2
    return s, l



