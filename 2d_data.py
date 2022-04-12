import numpy as np
import matplotlib.pyplot as plt
from gpdc_2d import *
import statsmodels.api as sm
from scipy.io import wavfile
import statsmodels as sm
import scipy.signal as ss
import torch
from misc import *

#############################################
# Parameters of the experiment
#############################################

dim = 32

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


#############################################
# Helper function for image manipulation
#############################################

def get_img_indices(img):
    row, col = img.shape
    indices = np.zeros((row*col,2))
    idx = 0
    for i in range(row):
        for j in range(col):
            indices[idx, 0] = i
            indices[idx, 1] = j
            idx += 1
    return indices
    
    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_image(image_name, dim):
    # load an image
    if image_name == 'grid':
        img = mpimg.imread('/home/ard/data/images/dtd/images/grid/grid_0011.jpg')
        x = img[:dim, :dim, :]

    elif image_name == 'swiss_plane':
        data = unpickle('/home/ard/data/images/cifar-10-batches-py/data_batch_1')[b'data']
        x = np.transpose(data[1400].reshape(3,32,32), (1, 2, 0))
        dim = 32

    elif image_name == 'frog':
        data = unpickle('/home/ard/data/images/cifar-10-batches-py/data_batch_1')[b'data']
        x = np.transpose(data[2100].reshape(3,32,32), (1, 2, 0))
        dim = 32

    elif image_name == 'plane':
        data = unpickle('/home/ard/data/images/cifar-10-batches-py/data_batch_1')[b'data']
        x = np.transpose(data[700].reshape(3,32,32), (1, 2, 0))
        dim = 32 # original image size

    elif image_name == 'rider':
        data = unpickle('/home/ard/data/images/cifar-10-batches-py/data_batch_1')[b'data']
        x = np.transpose(data[1200].reshape(3,32,32), (1, 2, 0))
        dim = 32

    elif image_name == 'bird':
        data = unpickle('/home/ard/data/images/cifar-10-batches-py/data_batch_1')[b'data']
        x = np.transpose(data[800].reshape(3,32,32), (1, 2, 0))
        dim = 32

    elif image_name == 'chequered':
        #img = mpimg.imread('/home/ard/data/images/dtd/images/chequered/chequered_0066.jpg')
        img = mpimg.imread('/home/ard/data/images/dtd/images/chequered/chequered_0106.jpg')
        x = img[:dim, :dim, :]

    elif image_name == 'woven':
        img = mpimg.imread('/home/ard/data/images/dtd/images/woven/woven_0056.jpg')
        x = img[:dim, :dim, :]

    elif image_name == 'horse':
        data = unpickle('/home/ard/data/images/cifar-10-batches-py/data_batch_1')[b'data']
        x = np.transpose(data[7].reshape(3,32,32), (1, 2, 0))
        dim = 32 # original image size

    elif image_name == 'emu':
        data = unpickle('/home/ard/data/images/cifar-10-batches-py/data_batch_1')[b'data']
        x = np.transpose(data[2224].reshape(3,32,32), (1, 2, 0))
        dim = 32 # original image size

    return x, dim
    
    
def get_filter(dim_w, mode='diag', seed=123):
    if mode == 'diag':
        h = np.ones((dim_w, dim_w)) * 1e-1 + np.eye(dim_w)
        h /= np.sum(h)
    
    elif mode == 'gauss':
        shape = (dim_w, dim_w)
        sigma =1.
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        h /= sumh
    
    elif mode == 'flat':
        h = np.ones((dim_w, dim_w))
        h /= np.sum(h)
    
    elif mode == 'random':
        np.random.seed(seed)
        h = np.random.rand(dim_w**2).reshape(dim_w, dim_w)
        h /= np.sum(h)
        
    elif mode == 'centered':
        h = np.ones((dim_w, dim_w))*1e-3
        h[dim_w//2, dim_w//2] = 1
        h /= np.sum(h)
    
    return h
    


#############################################
# Main experiment script
#############################################

data = unpickle('/home/ard/data/images/cifar-10-batches-py/data_batch_1')[b'data']
x = np.transpose(data[2100].reshape(3,32,32), (1, 2, 0))
x = x.reshape(-1, 3)
# light grey scale : https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm
x = np.array([0.3, 0.59, 0.11])@x.T
x -= np.mean(x)
x /= np.std(x)
x = x.reshape(dim, dim)
indices = get_img_indices(x)	


fig, ax = plt.subplots(1,3)

ax[0].imshow(x)
z = get_filter(5, 'gauss')
ax[1].imshow(z, cmap='gray')
y = ss.convolve2d(x, z, mode='valid')
ax[2].imshow(y)
plt.show()

print(x.shape, indices.shape)

indices = torch.tensor(indices)
x = torch.tensor(x.flatten())

# train model
gpc = GPC('RBF-RBF')
gpc.load(indices, x)
gpc.set_filter_params([filter_amplitude, filter_length_scale])
opt = torch.optim.Adam(gpc.parameters(),
                       lr=learning_rate, weight_decay=0.8)
for i in range(num_steps):
    res = gpc.train_step(opt)
    print_loss(res, i)

x_hat, V_hat, u_hat = gpc.forward(indices)
plt.imshow(x_hat.reshape(32,32))
plt.show()
exit()
