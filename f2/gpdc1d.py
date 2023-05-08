import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fcn
from scipy.special import kv
#import scipy.signal as sp
#sns.set_style("whitegrid")
#import scipy.signal as sp
#import spectrum as spectrum

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
    def __init__(self, times_y, y, num_weights, weight_step, offset, kernel='rbf', weights=None):
        self.times_y = times_y
        self.offset = offset
        self.num_weights = num_weights
        if weights is None:
            self.weights = np.ones(num_weights) / num_weights
            self.learn_weights = True
        else:
            self.weights = weights
            self.learn_weights = False
        self.weight_step = weight_step
        self.Nw_half = int(np.floor(len(self.weights)/2)) # suppose that weights are odd
        self.y = y
        self.x = None
        self.times_f = np.linspace(np.min(times_y), np.max(times_y), 1000)
        self.times_x = np.linspace(np.min(times_y)-1, np.max(times_y)+1, 1500)
        self.f = None
        self.Ny = len(self.y)
        self.time_label = None
        self.signal_label = None
        self.post_mean_f = None
        self.post_cov_f = None
        self.post_mean_x = None
        self.post_cov_x = None
        self.train_track = []

        if kernel == 'rbf':
            self.kern = rbf
            gamma = 1/2/((np.max(self.times_y)-np.min(self.times_y))/self.Ny)**2
            sigma_n = np.std(self.y)/10
            self.theta = [gamma, sigma_n]

        elif kernel == 'spectral':
            self.kern = Spec_Mix
            gamma = 1/2/((np.max(self.times_y)-np.min(self.times_y))/self.Ny)**2
            nu = 0.01
            sigma_n = np.std(self.y)/10
            self.theta = [gamma, nu, sigma_n]
        elif kernel == 'matern':
            self.kern = matern
            nu = 1.0
            sigma_n = np.std(self.y)/10
            self.theta = [nu, sigma_n]
        else:
            print("Error: unknown kernel name {}, it has been replaced by rbf".format(kernel))
            self.kern = 'rbf'


    def neg_log_likelihood(self):
        Y = self.y
        Gram = self.kernel_sum(self.times_y, self.times_y, self.theta[:-1], self.weights) + 1e-8*np.eye(len(self.times_y))
        K = Gram + self.theta[-1]**2*np.eye(self.Ny)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Ny*np.log(2*np.pi))


    def nlogp(self, params, info):
        Y = self.y
        if self.learn_weights:
            weights = params[len(self.theta):]
        else:
            weights = self.weights
        #weight_step = np.exp(params[-1])
        params = np.exp(params[:len(self.theta)])
        Gram = self.kernel_sum(self.times_y, self.times_y, params[:-1], weights, self.weight_step)
        K = Gram + params[-1]**2*np.eye(self.Ny) + 1e-5*np.eye(self.Ny)
        (sign, logdet) = np.linalg.slogdet(K)
        f = 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Ny*np.log(2*np.pi))

        # gaussian prior on weights N(O,sI) and sigma N(0,A) weight_step N(0,B)
        s = 1; B=1;
        prior_weights = 0.5 * (len(weights) * np.log(2*np.pi) + len(weights)*np.log(s) + np.dot(weights.T, weights)/s)
        #prior_weight_step = 0.5 * (np.log(2*np.pi*B**2) + weight_step**2/(B**2))
        nlp = f
        if self.learn_weights:
            nlp += prior_weights# + prior_weight_step
        if info['Nfeval'] % 20 == 0:
            self.train_track.append(nlp)
        info['Nfeval'] += 1
        return nlp


    def dnlogp(self, hypers):
        Y = self.y
        gamma, sigma, sigma_n = np.exp(hypers)

        Gram = self.kernel_sum(self.times_y, self.times_y, [gamma, sigma])
        K = Gram + sigma_n**2*np.eye(self.Ny) + 1e-5*np.eye(self.Ny)
        h = np.linalg.solve(K,Y).T

        dKdsigma = 2*Gram/sigma
        dKdgamma = -Gram*(outersum(self.times_y,-self.times_y)**2)
        dKdsigma_n = 2*sigma_n*np.eye(self.Ny)

        H = (np.outer(h,h) - np.linalg.inv(K))
        dlogp_dsigma = sigma * 0.5*np.trace(H@dKdsigma)
        dlogp_dgamma = gamma * 0.5*np.trace(H@dKdgamma)
        dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)
        return np.array([-dlogp_dgamma, -dlogp_dsigma, -dlogp_dsigma_n])


    def train(self):
        hypers0 = np.log(self.theta)
        self.train_track = []
        #hypers0 = np.concatenate((hypers0, self.weights, [np.log(self.weight_step)]))
        if self.learn_weights:
            hypers0 = np.concatenate((hypers0, self.weights))
        res = minimize(self.nlogp, hypers0, args=({'Nfeval': 0}), method='Powell', options={'maxiter': 200, 'disp': True})
        self.theta = np.exp(res.x[:len(self.theta)])
        if self.learn_weights:
            self.weights = res.x[len(self.theta):]
        #self.weight_step = np.exp(res.x[-1])
        print('Hyperparameters are:')
        print(f'{self.theta}')


    def compute_moments(self):
        #posterior moments for f 
        cov_f = self.kernel_sum(self.times_f, self.times_f, self.theta[:-1], self.weights)
        cov_obs = self.kernel_sum(self.times_y, self.times_y, self.theta[:-1], self.weights) 
        cov_obs = cov_obs + self.theta[-1]**2*np.eye(self.Ny) + 1e-5*np.eye(self.Ny)

        cov_star = self.kernel_sum(self.times_f, self.times_y, self.theta[:-1], self.weights)
        h = np.linalg.solve(cov_obs, self.y)
        self.post_mean_f = np.squeeze(cov_star@h)
        self.post_cov_f = cov_f - (cov_star@np.linalg.solve(cov_obs,cov_star.T))


        #posterior moments for x
        cov_x = self.kern(self.times_x, self.times_x, self.theta[:-1])
        cov_star = self.kernel_sum_xf(self.times_x, self.times_y, self.theta[:-1])
        self.post_mean_x = np.squeeze(cov_star@h)
        self.post_cov_x = cov_x - (cov_star@np.linalg.solve(cov_obs, cov_star.T))
        #return cov_real, xcov_real, cov_space


    def plot_posterior_f(self):
        #posterior moments for time
        plt.figure(figsize=(18,6))
        plt.plot(self.times_y,self.y,'.r', label='observations')
        plt.plot(self.times_f,self.post_mean_f, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt(np.diag(self.post_cov_f))
        plt.fill_between(self.times_f, self.post_mean_f - error_bars, self.post_mean_f + error_bars, color='blue',alpha=0.1, label='95% error bars')
        plt.title('Observations and posterior convolved process')
        plt.xlabel(self.time_label)
        plt.ylabel(self.signal_label)
        plt.legend()
        plt.xlim([min(self.times_y),max(self.times_y)])
        plt.tight_layout()


    def plot_posterior_x(self, time_truth=None, ground_truth=None):
        #posterior moments for time
        plt.figure(figsize=(18,6))
        if ground_truth is not None:
            plt.plot(time_truth,ground_truth, color='k', label='ground truth')
        plt.plot(self.times_x,self.post_mean_x, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt(np.diag(self.post_cov_x))
        #print(np.diag(self.post_cov_x))
        plt.fill_between(self.times_x, self.post_mean_x - error_bars, self.post_mean_x + error_bars, color='blue',alpha=0.1, label='95% error bars')
        plt.title('Posterior latent (de-convolved) process')
        plt.xlabel(self.time_label)
        plt.ylabel(self.signal_label)
        plt.legend()
        plt.xlim([min(self.times_x),max(self.times_x)])
        plt.tight_layout()


    def set_labels(self, time_label, signal_label):
        self.time_label = time_label
        self.signal_label = signal_label


    def kernel_sum(self, x, y, params, weights, weight_step=None):
        if weight_step is None:
            weight_step = self.weight_step
        offset = self.offset
        N_x = len(x)
        N_y = len(y)
        # find high resolution position corresponding to x and y
        x = np.repeat(x, self.num_weights, axis=0).reshape(-1, self.num_weights)
        weight_shifts = offset + np.array([i*weight_step for i in range(self.num_weights)])
        x += weight_shifts
        x = x.reshape(-1)
        y = np.repeat(y, self.num_weights, axis=0).reshape(-1, self.num_weights)
        weight_shifts = offset + np.array([i*weight_step for i in range(self.num_weights)])
        y += weight_shifts
        y = y.reshape(-1)
        # compute the kernel
        Wx = np.kron(np.eye(N_x), weights)
        Wy = np.kron(np.eye(N_y), weights)
        Kx = self.kern(x, y, params)
        tmp = np.dot(Wx, Kx)
        Kobs = np.dot(tmp, Wy.transpose())
        return Kobs


    def kernel_sum_xf(self, x, f, params):
        N_x = len(x)
        # find all point of interest to compute x
        x = np.repeat(x, self.num_weights, axis=0).reshape(-1, self.num_weights)
        weight_shifts = self.offset + np.array([i*self.weight_step for i in range(self.num_weights)])
        x += weight_shifts
        x = x.reshape(-1)
        # compute the kernel
        Wx = np.kron(np.eye(N_x), self.weights)
        Kx = self.kern(x, f, params)
        Kobs = np.dot(Wx, Kx)
        return Kobs


def rbf(x,y, params):
    gamma = params[0]
    sigma = 1.0
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)


def Spec_Mix(x,y, params):
    gamma, nu = params
    sigma = 1.0
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.cos(2*np.pi*nu*outersum(x,-y))


def matern(x,y,params):
    nu = params[0]
    sigma = 1.0
    cste = 2**(1-nu)/gamma_fcn(nu)
    dist = np.abs(outersum(x, -y))
    dist = np.where(dist==0, 1e-8, dist)
    dist = np.sqrt(2*nu) * dist / sigma
    return cste * dist**nu * kv(nu, dist)


def Spec_Mix_sine(x,y, gamma, theta, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.sin(2*np.pi*theta*outersum(x,-y))


def Spec_Mix_spectral(x, y, alpha, gamma, theta, sigma=1):
    magnitude = np.pi * sigma**2 / (np.sqrt(alpha*(alpha + 2*gamma)))
    return magnitude * np.exp(-np.pi**2/(2*alpha)*outersum(x,-y)**2 - 2*np.pi*2/(alpha + 2*gamma)*(outersum(x,y)/2-theta)**2)


def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)


