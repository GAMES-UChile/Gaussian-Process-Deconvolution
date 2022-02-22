import numpy as np
import matplotlib.pyplot as plt
from gpdc_torch import *
import statsmodels.api as sm
from scipy.io import wavfile
import statsmodels as sm
import scipy.signal as ss
import torch
from misc import *

# experiment parameters
N_data = 1000
num_steps = 50 
learning_rate = 0.1
length_scale = 15
amplitude_scale = 1
sigma_noise = 0.1
filter_amplitude = 0.15
filter_length_scale = 20
times_h = np.arange(-30, 30, 1)
h = filter_amplitude*np.exp(-(1/(2*filter_length_scale))*(times_h)**2)

# generate data
gpc = GPC('RBF-RBF')
times_x = torch.arange(0, N_data, 1).detach()
x = gpc.sample_x(times_x, params = [amplitude_scale, length_scale], how_many=1)
f = np.convolve(x.flatten(),h,mode='same')
y = f + sigma_noise*np.random.randn(len(times_x))
idx = np.random.choice(len(times_x), 1000)
times_y = times_x[idx]
y = y[idx]
del gpc

# train model
gpc = GPC('RBF-RBF', learn_inducing=True)
gpc.load(times_y, y)
gpc.set_filter_params([filter_amplitude, filter_length_scale])
opt = torch.optim.Adam(gpc.parameters(),
                       lr=learning_rate, weight_decay=0.8)
for i in range(num_steps):
    res = gpc.train_step(opt)
    print_loss(res, i)

# visualize setting and results
fig, ax = plt.subplots(2, 1, figsize=(10,8))
ax[0].plot(times_x, x, label = 'x');
ax[0].plot(times_x, f, label = 'f');
ax[0].plot(times_y, y, '.r', label = 'y');
ax[0].plot(times_h, 10*h, lw=3, label = 'h');
ax[0].legend()
# plot deconvolution with sparse representation and learnt parameters
x_hat, V_hat, u_hat = gpc.forward(times_x)
ax[1].plot(times_x, x_hat, label = 'deconv')
error_bars = np.sqrt(np.diag(V_hat))
ax[1].fill_between(times_x, x_hat-2*error_bars,
                   x_hat+2*error_bars,
                   label='error bars', alpha=0.5)
ax[1].plot(times_x, x, label = 'true')
ax[1].plot(times_y, y, '.r', label='observations')
u_loc = gpc.u_loc.detach().numpy()
ax[1].scatter(u_loc, -2*np.ones_like(u_loc), marker="^", color='r', label='inducing points')
ax[1].legend()
ax[1].set_xlim([0, 500])
plt.show()

