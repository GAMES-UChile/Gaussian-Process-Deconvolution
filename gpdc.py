import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
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
    def __init__(self,kernel_id, params=None):
        #self.times_y = None
        #self.y = None
        #self.times_x = None
        #self.x = None
        #self.times_f = None
        #self.f = None
        #self.Ny = None
        self.params = params
        self.kernel_id = kernel_id
        #self.sigma = None
        #self.gamma = None
        #self.theta = None
        #self.sigma_x = None
        #self.gamma_x = None
        #self.theta_x = None
        #self.gamma_h = None
        #self.sigma_n = None
        #self.time_label = None
        #self.signal_label = None
        #self.post_mean_f = None
        #self.post_cov_f = None
        #self.post_mean_x = None
        #self.post_cov_x = None

    def set_observations(self, times_y,y): 
        self.times_y = times_y
        self.y = y
        self.times_x = np.linspace(np.min(times_y), np.max(times_y),25)
        self.x = None
        self.times_f = np.linspace(np.min(times_y), np.max(times_y),100)
        self.f = None
        self.Ny = len(self.y)
        self.sigma = np.std(self.y)
        self.gamma = 1/2/((np.max(self.times_y)-np.min(self.times_y))/self.Ny)**2
        self.theta = 0.0
        self.sigma_x = np.std(self.y)
        self.gamma_x = 1/2/((np.max(self.times_y)-np.min(self.times_y))/self.Ny)**2
        self.theta_x = 0.0
        self.gamma_h = None
        self.sigma_n = np.std(self.y)/10

    def set_observations_no_hype(self, times_y,y):
        self.times_y = times_y
        self.y = y
        self.Ny =len(y)

    def load(self, times_y,y):
        self.times_y = times_y
        self.y = y
        self.Ny =len(y)

    def set_filter_params(self, p):
        # set the filter h parameters for the not-blind case
        # For RBF: p0 * exp(-t**2/(2*p1)), i.e., variance and sq lengthscale
        self.theta_h = p

    def neg_log_likelihood(self):
        Y = self.y
        Gram = Spec_Mix(self.times_y,self.times_y,self.gamma,self.theta,self.sigma) + 1e-8*np.eye(self.Ny)
        K = Gram + self.sigma_n**2*np.eye(self.Ny)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Ny*np.log(2*np.pi))


    def nlogp(self, hypers):
        sigma = np.exp(hypers[0])
        gamma = np.exp(hypers[1])
        theta = np.exp(hypers[2])*0
        sigma_n = np.exp(hypers[3])

        Y = self.y
        Gram = Spec_Mix(self.times_y,self.times_y,gamma,theta,sigma)
        K = Gram + sigma_n**2*np.eye(self.Ny) + 1e-5*np.eye(self.Ny)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Ny*np.log(2*np.pi))

    def dnlogp(self, hypers):
        sigma = np.exp(hypers[0])
        gamma = np.exp(hypers[1])
        theta = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Y = self.y
        Gram = Spec_Mix(self.times_y,self.times_y,gamma,theta,sigma)
        K = Gram + sigma_n**2*np.eye(self.Ny) + 1e-5*np.eye(self.Ny)
        h = np.linalg.solve(K,Y).T

        dKdsigma = 2*Gram/sigma
        dKdgamma = -Gram*(outersum(self.times_y,-self.times_y)**2)
        dKdtheta = -2*np.pi*Spec_Mix_sine(self.times_y,self.times_y, gamma, theta, sigma)*outersum(self.times_y,-self.times_y)
        dKdsigma_n = 2*sigma_n*np.eye(self.Ny)

        H = (np.outer(h,h) - np.linalg.inv(K))
        dlogp_dsigma = sigma * 0.5*np.trace(H@dKdsigma)
        dlogp_dgamma = gamma * 0.5*np.trace(H@dKdgamma)
        dlogp_dtheta = theta * 0.5*np.trace(H@dKdtheta)
        dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)
        return np.array([-dlogp_dsigma, -dlogp_dgamma, -dlogp_dtheta*0, -dlogp_dsigma_n])

    def train(self):
        hypers0 = np.array([np.log(self.sigma), np.log(self.gamma), (self.theta), np.log(self.sigma_n)])
        res = minimize(self.nlogp, hypers0, args=(), method='L-BFGS-B', jac = self.dnlogp, options={'maxiter': 500, 'disp': False})
        self.sigma = np.exp(res.x[0])
        self.gamma = np.exp(res.x[1])
        self.theta = np.exp(res.x[2])
        self.sigma_n = np.exp(res.x[3])
        self.u
        print('Hyperparameters are:')
        print(f'sigma ={self.sigma}')
        print(f'gamma ={self.gamma}')
        print(f'theta ={self.theta}')
        print(f'sigma_n ={self.sigma_n}')



    def train_sparse(self, m=10):
        self.m = m
        if self.kernel_id == 'RBF-RBF':
            print(f'training GPDC using {self.kernel_id} VGP with {m} inducing points')
            #initialisation
            hypers0 = np.zeros(3)
            # std is empirical std divided by filter magnitude
            hypers0[0] = np.var(self.y)/self.theta_h[0] 
            # lenghtscale is 20% of the data range
            hypers0[1] = ((np.max(self.times_y)-np.min(self.times_y))/5)**2
            # noise var is 10% of data variance
            hypers0[2] = 0.01*np.var(self.y)

            # init cond and optimisation
            hypers0 = np.log(hypers0)
            # loc0 = np.linspace(np.min(self.times_y),np.max(self.times_y), m)
            print(f'initial hypers: {hypers0}')
            # hypers0 = np.concatenate((hypers0, loc0), axis=0) 
            res = minimize(self.VSGP_loss, hypers0, args=(), method='L-BFGS-B', options={'maxiter': 500, 'disp': False})
            self.params = np.exp(res.x[0:3])
            self.ind_loc = res.x[3:]

            print('Hyperparameters are:')
            print(f'sigma2 ={self.params[0]}')
            print(f'lenghtscale2 ={self.params[1]}')
            print(f'sigma2_n ={self.params[2]}')
            # print(f'induc locations ={self.ind_loc}')

    def VSGP_loss(self, hypers):

        if self.kernel_id == 'RBF-RBF':
            # print(f'loss function for {self.kernel_id}')
            # use more eloquent notation:
            # kernel & variational params
            s2 = np.exp(hypers[0])    # marginal variance
            l2 = np.exp(hypers[1])    # square lengthscale
            s2_n = np.exp(hypers[2])  # noise variance
            # u_loc = hypers[3:] 
            # m = len(u_loc)
            u_loc = np.linspace(np.min(self.times_y), np.max(self.times_y), self.m)
            beta = 1/s2_n  # noise precision
            # data
            Y = self.y
            N = self.Ny
            # filter params
            mag_h = self.theta_h[0]  # filter's magnitude or power
            l2_h = self.theta_h[1]   # filter's square lengthscale

            # building cost fn
            s2_fx, l2_fx = RBF_convolution(s2, l2, mag_h, l2_h)
            s2_f, l2_f = RBF_convolution(s2_fx, l2_fx, mag_h, l2_h)

            Kmn = SE(u_loc, self.times_y, s2_fx, l2_fx)
            Kmm = SE(u_loc, u_loc, s2, l2) 

            temp1 = Kmn.T@np.linalg.solve(Kmm + np.eye(self.m)*1e-8,Kmn)
            temp2 = temp1 + np.eye(N)*s2_n

            trace_Knn = N*s2_f
            temp3 = beta/2*(trace_Knn - np.trace(temp1))
            (sign, logdet) = np.linalg.slogdet(temp2)

            ELBO = -N/2*np.log(2*np.pi) - 1/2*logdet - 1/2*Y.T@np.linalg.solve(temp2+ np.eye(N)*1e-8   , Y) - temp3
            # print(beta)
            return -ELBO

    def predict_VSGP(self, times_x = None):

        self.ind_loc = np.linspace(np.min(self.times_y),
                                   np.max(self.times_y), self.m)
        s2, l2, s2_n = self.params
        beta = 1/s2_n
        mag_h = self.theta_h[0]  # filter's magnitude or power
        l2_h = self.theta_h[1]   # filter's square lengthscale
        s2_fx, l2_fx = RBF_convolution(s2, l2, mag_h, l2_h)
        u_loc = self.ind_loc
        m = len(u_loc)

        Kmn = SE(u_loc, self.times_y, s2_fx, l2_fx)
        Kmm = SE(u_loc, u_loc, s2, l2) 
        temp1 = np.linalg.solve(Kmm, Kmn)
        Lambda = np.linalg.solve(Kmm, beta*Kmn@(temp1.T) + np.eye(m))
        u_hat = beta * np.linalg.solve(Kmm@Lambda, Kmn@self.y)

        Kxm = SE(times_x, u_loc, s2, l2)
        Kxx = SE(times_x, times_x, s2, l2)

        # return  Kxm@np.linalg.solve(Kmm, u_hat), Kxx -Kxm@np.linalg.solve(Kmm, Kxm.T), u_hat
        L = np.linalg.solve(Kmm, Kxm.T)
        return L.T@u_hat, Kxx + (np.linalg.solve(Lambda, L).T - Kxm)@L, u_hat

    def predict(self, times_x = None):
        np.random.seed(1)

        s2, l2, s2_n = self.params
        mag_h = self.theta_h[0]  # filter's magnitude or power
        l2_h = self.theta_h[1]   # filter's square lengthscale
        s2_fx, l2_fx = RBF_convolution(s2, l2, mag_h, l2_h)
        s2_f, l2_f = RBF_convolution(s2_fx, l2_fx, mag_h, l2_h)

        Kyy = SE(self.times_y, self.times_y, s2_f, l2_f) + np.eye(self.Ny)*s2_n
        Kxy = SE(times_x, self.times_y, s2_fx, l2_fx)
        Kxx = SE(times_x, times_x, s2, l2)
        return  Kxy@np.linalg.solve(Kyy, self.y), Kxx - Kxy@np.linalg.solve(Kyy, Kxy.T)

    def sample_x(self, times_x=np.linspace(0, 1000, 100), params=[1, 1], how_many=1):
        np.random.seed(1)
        if self.kernel_id == 'RBF-RBF':
            cov_x = SE(times_x, times_x, params[0], params[1]) + 0*1e-5*np.eye(len(times_x))
        x = np.random.multivariate_normal(np.zeros_like(times_x), cov_x, how_many)
        return x.T

    def sample_from_prior(self, times_x, times_f, params):
        np.random.seed(1)

        if self.kernel_id == 'RBF-RBF':
            gamma_x, theta_x, sigma_x, gamma_h, theta_h = params[:5]
            sigma_h = (gamma_h/np.pi)**(0.25)
            gamma_xh = 1/(1/gamma_x + 1/gamma_h)
            sigma_xh = sigma_x*sigma_h*(np.pi/(gamma_x+gamma_h))**(0.25)
            sigma_y = sigma_x*sigma_h*sigma_h*(np.pi/np.sqrt(2*gamma_x*gamma_h + gamma_h**2))**(0.5)
            theta_y = theta_x
            gamma_y = 1/(1/gamma_x + 2/gamma_h)
            cov_x = Spec_Mix(times_x, times_x, gamma_x, theta_x,
                             sigma_x) + 1e-5*np.eye(len(times_x))
            cov_fx = Spec_Mix(times_f, times_x, gamma_xh, 
                              theta_x, sigma_xh)
            cov_f = Spec_Mix(times_f, times_f, gamma_y, theta_y,
                             sigma_y) + 1e-5*np.eye(len(times_f))
            paramy = np.array([gamma_y, theta_y, sigma_y, ])

        elif self.kernel_id == 'Sinc-Sinc':
            xi_x, delta_x, sigma_x, xi_h, delta_h = params[:5]
            sigma_h = 1
            borders = np.sort(np.array([xi_x-delta_x/2,xi_x+delta_x/2,xi_h-delta_h/2,xi_h+delta_h/2]))
            xi_xh = (borders[1] + borders[2])/2 
            if borders[3] - borders[0] - delta_h - delta_x > 0:
                delta_xh = 0
                sigma_xh = 0
                sigma_xhh = 0
            else:
                delta_xh = borders[2] - borders[1]
                sigma_xh = np.sqrt(sigma_x**2/(2*delta_x)*sigma_h**2/(2*delta_h))*2*delta_xh
                sigma_xhh = np.sqrt(sigma_xh**2/(2*delta_xh)*sigma_h**2/(2*delta_h))*2*delta_xh
            paramy = np.array([xi_xh,delta_xh,sigma_xhh])
            cov_x = Sinc(times_x,times_x,xi_x,delta_x,sigma_x) + 1e-5*np.eye(len(times_x))
            cov_fx = Sinc(times_f,times_x,xi_xh,delta_xh,sigma_xh)
            cov_f = Sinc(times_f,times_f,xi_xh,delta_xh,sigma_xhh) + 1e-5*np.eye(len(times_f))
        
        x = np.random.multivariate_normal(np.zeros_like(times_x),cov_x)
        mf = cov_fx@np.linalg.solve(cov_x,x)
        vf = cov_f - cov_fx@np.linalg.solve(cov_x,cov_fx.T)
        f = np.random.multivariate_normal(mf,vf)
        return x, f, mf, vf, paramy
    



    def compute_moments(self,times_x=None,times_y=None):
        self.times_x = times_x
        self.times_y = times_y
        #posterior moments for f
        if self.kernel_id == 'RBF-RBF':

            #cov_obs = Spec_Mix(self.times_y,self.times_y,self.gamma,self.theta,self.sigma) + 1e-5*np.eye(self.Ny) + self.sigma_n**2*np.eye(self.Ny)
            #cov_f = Spec_Mix(self.times_f,self.times_f, self.gamma, self.theta, self.sigma)
            #cov_star = Spec_Mix(self.times_f,self.times_y, self.gamma, self.theta, self.sigma)
            #h = np.linalg.solve(cov_obs,self.y)
            #self.post_mean_f = np.squeeze(cov_star@h)
            #self.post_cov_f = cov_f - (cov_star@np.linalg.solve(cov_obs,cov_star.T))
            
            #posterior moments for x
            #l_h = np.sqrt(0.05)
            #gamma_h = 1/(2*l_h**2)
            gamma_x, theta_x, sigma_x, gamma_h, theta_h, sigma_h,  sigma_n = self.params[:7]

            gamma_xh = 1/(1/gamma_x + 1/gamma_h)
            sigma_xh = sigma_x*sigma_h*(np.pi/(gamma_x+gamma_h))**(0.25)
            #sigma_y = sigma_x*sigma_h*sigma_h*(np.pi/np.sqrt(2*gamma_x*gamma_h + gamma_h**2))**(0.5)
            sigma_y = sigma_x*sigma_h*sigma_h*(np.pi/np.sqrt((gamma_x+gamma_h)*(gamma_xh+gamma_h)))**(0.5)
            theta_y = theta_x
            gamma_y = 1/(1/gamma_x + 2/gamma_h)

            #l_h = 1/np.sqrt(2*gamma_h)
            #gamma_para = self.gamma*gamma_h/(gamma_h - self.gamma)
            #gamma_x = gamma_para*gamma_h/(gamma_h-gamma_para)
            #print(f'gamma_x es: {gamma_x}')
            #cov_x = Spec_Mix(self.times_x,self.times_x, gamma_x, self.theta, self.sigma) / ( ((2*np.pi*l_h**2)**(-0.5)*0.1)**2 * np.pi / np.sqrt((gamma_h+gamma_x)*(gamma_para+gamma_h))) 
            #cov_star = Spec_Mix(self.times_x,self.times_y, gamma_para, self.theta, self.sigma) * np.sqrt(np.pi/(gamma_h+gamma_x))/ ( ((2*np.pi*l_h**2)**(-0.5)*0.1) * np.pi / np.sqrt((gamma_h+gamma_x)*(gamma_para+gamma_h))) 
            cov_x = Spec_Mix(self.times_x,self.times_x, gamma_x, theta_x, sigma_x) 
            cov_star = Spec_Mix(self.times_x,self.times_y, gamma_xh, theta_x, sigma_xh)
            cov_obs = Spec_Mix(self.times_y,self.times_y,gamma_y,theta_x,sigma_y) + sigma_n**2*np.eye(self.Ny)
            vec = np.linalg.solve(cov_obs,self.y)
            self.post_mean_x = np.squeeze(cov_star@vec)
            self.post_cov_x = cov_x - (cov_star@np.linalg.solve(cov_obs,cov_star.T))
            #return cov_real, xcov_real, cov_space
        elif self.kernel_id == 'Sinc-Sinc':
            #cov_obs = Spec_Mix(self.times_y,self.times_y,self.gamma,self.theta,self.sigma) + 1e-5*np.eye(self.Ny) + self.sigma_n**2*np.eye(self.Ny)
            #cov_f = Spec_Mix(self.times_f,self.times_f, self.gamma, self.theta, self.sigma)
            #cov_star = Spec_Mix(self.times_f,self.times_y, self.gamma, self.theta, self.sigma)
            #h = np.linalg.solve(cov_obs,self.y)
            #self.post_mean_f = np.squeeze(cov_star@h)
            #self.post_cov_f = cov_f - (cov_star@np.linalg.solve(cov_obs,cov_star.T))
            
            #posterior moments for x
            #l_h = np.sqrt(0.05)
            #gamma_h = 1/(2*l_h**2)

            xi_x, delta_x, sigma_x, xi_h, delta_h, sigma_h, sigma_n = self.params[:7]
            borders = np.sort(np.array([xi_x-delta_x/2,xi_x+delta_x/2,xi_h-delta_h/2,xi_h+delta_h/2]))
            xi_xh = (borders[1] + borders[2])/2 
            if borders[3] - borders[0] - delta_h - delta_x > 0:
                delta_xh = 0
                sigma_xh = 0
                sigma_xhh = 0
            else:
                delta_xh = borders[2] - borders[1]
                sigma_xh = np.sqrt(sigma_x**2/(2*delta_x)*sigma_h**2/(2*delta_h))*2*delta_xh
                sigma_xhh = np.sqrt(sigma_xh**2/(2*delta_xh)*sigma_h**2/(2*delta_h))*2*delta_xh
            paramy = np.array([xi_xh,delta_xh,sigma_xhh])
            cov_x = Sinc(times_x,times_x,xi_x,delta_x,sigma_x) 
            cov_star = Sinc(times_x, self.times_y,xi_xh,delta_xh,sigma_xh)
            cov_obs = Sinc(self.times_y,self.times_y,xi_xh,delta_xh,sigma_xhh) + sigma_n**2*np.eye(len(times_y))
            vec = np.linalg.solve(cov_obs,self.y)
            
            self.post_mean_x = np.squeeze(cov_star@vec)
            self.post_cov_x = cov_x - (cov_star@np.linalg.solve(cov_obs,cov_star.T))
            #return cov_real, xcov_real, cov_space

    def plot_posterior_f(self):
        #posterior moments for time
        np.random.seed(1)
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

    def plot_posterior_x(self, time_truth=None, ground_truth=None, format=False):
        #posterior moments for time
        plt.figure(figsize=(16,4))
        np.random.seed(1)
        if ground_truth is not None:
            plt.plot(time_truth,ground_truth, color='cornflowerblue', label='True speech')
        plt.plot(self.times_x,self.post_mean_x, color='green', label='GPDC mean')
        error_bars = 2 * np.sqrt(np.diag(self.post_cov_x))
        plt.fill_between(self.times_x, self.post_mean_x - error_bars, self.post_mean_x + error_bars, color='green',alpha=0.2, label='95% error bars')

        
        if format:
            plt.title('GPDC deconvolution and true speech signal (time domain)')
            #plt.title('Posterior latent (de-convolved) process')
            plt.xlabel(self.time_label)
            plt.ylabel(self.signal_label)
            plt.legend(loc = 'upper left', ncol = 3)
        plt.xlim([min(self.times_x),max(self.times_x)])
        plt.tight_layout()

    def plot_freq_posterior(self):
        #posterior moments for frequency
        plt.figure(figsize=(18,6))
        plt.plot(self.w,self.post_mean_r, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_r)))
        plt.fill_between(self.w, self.post_mean_r - error_bars, self.post_mean_r + error_bars, color='blue',alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (real part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()


        plt.figure(figsize=(18,6))
        plt.plot(self.w,self.post_mean_i, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_i)))
        plt.fill_between(self.w, self.post_mean_i - error_bars, self.post_mean_i + error_bars, color='blue',alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (imaginary part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()

    def plot_power_spectral_density(self, how_many):
        #posterior moments for frequency
        plt.figure(figsize=(18,6))
        freqs = len(self.w)
        samples = np.zeros((freqs,how_many))
        for i in range(how_many):               
            sample_r = np.random.multivariate_normal(self.post_mean_r,(self.post_cov_r+self.post_cov_r.T)/2 + 1e-5*np.eye(freqs))
            sample_i = np.random.multivariate_normal(self.post_mean_i,(self.post_cov_i+self.post_cov_i.T)/2 + 1e-5*np.eye(freqs))
            samples[:,i] = sample_r**2 + sample_i**2
        plt.plot(self.w,samples, color='red', alpha=0.35)
        plt.plot(self.w,samples[:,0], color='red', alpha=0.35, label='posterior samples')
        posterior_mean_psd = self.post_mean_r**2 + self.post_mean_i**2 + np.diag(self.post_cov_r + self.post_cov_r)
        plt.plot(self.w,posterior_mean_psd, color='black', label = '(analytical) posterior mean')
        plt.title('Sample posterior power spectral density')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()

    def set_labels(self, time_label, signal_label):
        self.time_label = time_label
        self.signal_label = signal_label


def outersum(a, b):
    return np.outer(a, np.ones_like(b))+np.outer(np.ones_like(a), b)


def Spec_Mix(x, y, gamma, theta, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x, -y)**2)*np.cos(2*np.pi*theta*outersum(x, -y))


def SE(x,y, s2, l2):
    return s2 * np.exp(-outersum(x, -y)**2/(2*l2))


def Spec_Mix_sine(x,y, gamma, theta, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.sin(2*np.pi*theta*outersum(x,-y))

def Spec_Mix_spectral(x, y, alpha, gamma, theta, sigma=1):
    magnitude = np.pi * sigma**2 / (np.sqrt(alpha*(alpha + 2*gamma)))
    return magnitude * np.exp(-np.pi**2/(2*alpha)*outersum(x,-y)**2 - 2*np.pi*2/(alpha + 2*gamma)*(outersum(x,y)/2-theta)**2)

def Sinc(x,y, xi, delta, sigma=1):
    return sigma**2 * np.cos(2*np.pi*xi*outersum(x,-y))*np.sinc(delta*outersum(x,-y))

def RBF_convolution(s1,l1, s2, l2):
    # inputs are *square* variance and lengthscales
    s = s1*s2*np.sqrt(2*np.pi*l1*l2/(l1+l2))
    l = l1 + l2
    return s, l
