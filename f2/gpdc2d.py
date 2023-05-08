import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fcn
from scipy.special import kv
from scipy.linalg.blas import sgemm
import numexpr as ne


plot_params = {'legend.fontsize': 18,
               'figure.figsize': (15, 5),
               'xtick.labelsize':'18',
               'ytick.labelsize':'18',
               'axes.titlesize':'24',
               'axes.labelsize':'22'}
plt.rcParams.update(plot_params)


class gpc:

    # Class Attribute none yet
    # Initializer / Instance Attributes
    def __init__(self, indices_x, indices_y, y,  weights=None, num_weights=None):

        self.Ny = len(indices_y)
        self.learn_weights = False
        self.weights = weights
        if weights is None:
            self.weights = np.ones(num_weights)/num_weights
            dim = int(np.sqrt(num_weights))
            self.weights = self.weights.reshape(dim, dim)
            print(self.weights.shape)
            self.learn_weights = True
        self.size_filter = len(self.weights)
        self.y = y
        self.x = None
        self.indices_x = indices_x
        self.indices_f = indices_x
        self.indices_y = indices_y
        self.post_mean_f = None
        self.post_cov_f = None
        self.post_mean_x = None
        self.post_cov_x = None
        self.img_size = int(np.max(indices_x) + 1) # assume square images
        self.train_track = []

        self.kern = rbf
        gamma = 0.02
        var = 1.0
        sigma_n = 0.1
        self.theta = [gamma, var, sigma_n]


    def neg_log_likelihood(self):
        Y = self.y
        Gram = self.kernel_sum(self.indices_y, self.indices_y, self.theta[:-1], self.weights) + 1e-8*np.eye(self.Ny)
        K = Gram + self.theta[-1]**2*np.eye(self.Ny)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Ny*np.log(2*np.pi))


    def nlogp(self, params, info):
        Y = self.y
        weights = self.weights
        if self.learn_weights:
            weights = params[len(self.theta):].reshape(self.size_filter, self.size_filter)
        params = np.exp(params[:len(self.theta)])
        Gram = self.kernel_sum(self.indices_y, self.indices_y, params[:-1], weights)
        K = Gram + params[-1]**2*np.eye(self.Ny) + 1e-5*np.eye(self.Ny)
        (sign, logdet) = np.linalg.slogdet(K)
        nlp = 0.5*(Y.T@np.linalg.solve(K,Y) + logdet + self.Ny*np.log(2*np.pi))

        # gaussian prior on weights N(O,sI) and sigma N(0,A)
        if self.learn_weights:
            s = 1; a = 1
            sigma = params[1]
            weights = weights.reshape(-1)
            prior_weights = 0.5 * (len(weights) * np.log(2*np.pi) + len(weights)*np.log(s) + np.dot(weights.T, weights)/s)
            prior_sigma = 0.5 * (np.log(2*np.pi) + np.log(a) + (sigma**2)/a)
            nlp = nlp + prior_weights + prior_sigma

        if info['Nfeval'] % 20 == 0:
            self.train_track.append(nlp)
        info['Nfeval'] += 1

        return nlp


    def train(self):
        hypers0 = np.log(self.theta)
        if self.learn_weights:
            hypers0 = np.append(hypers0, self.weights.reshape(-1))
        self.train_track = []
        res = minimize(self.nlogp, hypers0, args=({'Nfeval': 0}), method='Powell', options={'maxiter': 400, 'disp': True})
        self.theta = np.exp(res.x[:len(self.theta)])
        if self.learn_weights:
            self.weights = res.x[len(self.theta):].reshape(self.size_filter, self.size_filter)
        print('Hyperparameters are:')
        print(f'{self.theta}')


    def compute_moments(self):
        #posterior moments for f 
        cov_f = self.kernel_sum(self.indices_f, self.indices_f, self.theta[:-1], self.weights)
        cov_obs = self.kernel_sum(self.indices_y, self.indices_y, self.theta[:-1], self.weights)
        cov_obs = cov_obs + self.theta[-1]**2*np.eye(self.Ny) + 1e-5*np.eye(self.Ny)
        cov_star = self.kernel_sum(self.indices_f, self.indices_y, self.theta[:-1], self.weights)
        h = np.linalg.solve(cov_obs, self.y)
        self.post_mean_f = np.squeeze(cov_star@h)
        self.post_cov_f = cov_f - (cov_star@np.linalg.solve(cov_obs, cov_star.T))

        #posterior moments for x
        cov_x = self.kern(self.indices_x, self.indices_x, self.theta[:-1])
        cov_star = self.kernel_sum_xf(self.indices_x, self.indices_y, self.theta[:-1], self.weights)
        self.post_mean_x = np.squeeze(cov_star@h)
        self.post_cov_x = cov_x - (cov_star@np.linalg.solve(cov_obs, cov_star.T))
        #return cov_star


    def kernel_sum(self, indices_l, indices_r, params, weights):
        Wl = get_weight_mat(indices_l, self.img_size, weights)
        Wr = get_weight_mat(indices_r, self.img_size, weights)
        Kx = rbf(self.indices_x, self.indices_x, params)
        tmp = np.dot(Wl, Kx)
        Kobs = np.dot(tmp, Wr.transpose())
        return Kobs


    def kernel_sum_xf(self, indices_x, indices_f, params, weights):
        W = get_weight_mat(indices_f, self.img_size, weights)
        Kx = rbf(indices_x, indices_x, params)
        Kxf = np.dot(Kx, W.T)
        return Kxf


def get_weight_mat(indices_f, size, weights):
    rw, cw = weights.shape
    W = np.zeros((len(indices_f), size**2))
    rw2 = rw//2; cw2 = cw//2 # frame size
    for k, (i, j) in enumerate(indices_f):
        # index in augmented image
        ai = int(i + rw2); aj = int(j + cw2)
        img = np.zeros((size + rw-1, size + cw-1))
        img[(ai - rw2):(ai + rw2 + 1), (aj - cw2):(aj + cw2 + 1)] = weights
        mask = img[(rw2):-(rw2), (cw2):-(cw2)]
        W[k, :] = mask.reshape(-1)
    return W


"""
def get_weight_mat(x_size, size, weights):
    rw, cw = weights.shape
    weight_pad = np.zeros((rw, x_size))
    weight_pad[:, :cw] = weights
    weight_pad = weight_pad.reshape(-1,)
    offset = len(weight_pad)//2 + cw//2
    mask = np.zeros(x_size**2 + offset)
    mask[:len(weight_pad)] = weight_pad
    mask = mask.reshape(1,-1)
    W = mask.repeat(size**2, axis=0)
    for i in range(size**2):
        W[i,:] = np.roll(W[i,:], i)
    return W[:, offset:]
"""

def rbf(X,Y, params):
    gamma, var = params
    X_norm = -np.einsum('ij,ij->i',X,X)
    Y_norm = -np.einsum('ij,ij->i', Y,Y)
    return ne.evaluate('v * exp(g * (A + B + 2 * C))', {\
        'A' : X_norm[:,None],\
        'B' : Y_norm[None,:],\
        'C' : np.dot(X, Y.T),\
        'g' : gamma,\
        'v' : var\
        })


