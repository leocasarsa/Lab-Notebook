# TODO:
#   - generalize train_gpmcc to accept any model
#   - adapt functions for when num_states > 1

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import pickle
import sys
# CHANGE THIS: gpmcc directory address
sys.path.append('/home/casarsa/Git/gpmcc/')

from matplotlib import cm, colors
from scipy.stats import norm as normal
from numpy.linalg import norm
from gpmcc.engine import Engine
from sigma_cmi import gaussian_cmi, automatic_Sigma

## Data Generation ##
def simulate_X(N, rng, loc=0, scale=1):
    return normal.rvs(loc=loc, scale=scale , size=[N,1], random_state=rng)

def simulate_Y_gX(N, gX, rng, loc=0, scale=1):
    assert gX.shape[0]==N
    return gX + normal.rvs(loc=loc, scale=scale , size=[N,1], random_state=rng)

def simulate_Z_gY(N, gY, rng, loc=0, scale=1):
    assert gY.shape[0]==N
    return gY + normal.rvs(loc=loc, scale=scale , size=[N,1], random_state=rng)

def simulate_chain(N, rng, scaleX=1, scaleY=1, scaleZ=1):
    X = simulate_X(N, rng, scale=scaleX)
    Y = simulate_Y_gX(N, X, rng, scale=scaleY)
    Z = simulate_Z_gY(N, Y, rng, scale=scaleZ)
    return np.hstack((X,Y,Z))

# Parameters for automatic_Sigma
def dag_and_noise(scaleX=1, scaleY=1, scaleZ=1):
  dag = np.array([[0,0,0],
                  [1,0,0],
                  [0,1,0]])
  noise = np.array([scaleX, scaleY, scaleZ])
  return dag, noise


## Training GPMCC ##
def train_gpmcc_chain(num_points=100, num_states=1, num_transitions = 100, save_out=True):
    data = simulate_chain(num_points, rng = np.random.RandomState(0))
    start_time = time.time()
    engine = Engine(data, ['normal','normal','normal'], num_states=num_states, initialize=1)
    engine.transition(N=num_transitions, do_progress=True, do_plot=False)
    train_time = time.time() - start_time
    engine.num_points = num_points
    engine.num_transitions = num_transitions
    engine.data = data
    if save_out:
        file_engine = file('resources/eng_chain_pnts%d_stats%d_trans%d.pkl' % (num_points, num_states, num_transitions),'wb')
        file_data_time =file('resources/data_time_chain_pnts%d_stats%d_trans%d.pkl' % (num_points, num_states, num_transitions),'wb') 
        engine.to_pickle(file_engine)
        pickle.dump([data, train_time],file_data_time)
    return engine, data, train_time


def load_gpmcc_chain(num_points=100, num_states=1, num_transitions = 100):
    file_engine = file('resources/eng_chain_pnts%d_stats%d_trans%d.pkl' % (num_points, num_states, num_transitions),'rb')
    file_data_time =file('resources/data_time_chain_pnts%d_stats%d_trans%d.pkl' % (num_points, num_states, num_transitions),'rb') 
    engine = Engine.from_pickle(file_engine)
    data, train_time = pickle.load(file_data_time)
    return engine, data, train_time


def init_gpmcc_chain(learn_params, save_out=True):
    '''
    Train engine if there is not already a saved trained engine.
    '''
    try:
        engine, data, train_time = load_gpmcc_chain(**learn_params)
        engine.num_points = learn_params['num_points']
        engine.num_transitions = learn_params['num_transitions']
        engine.data = data
        print "\nEngine successfully loaded\n"
    except IOError as err:
        print err, "\nFile not found. Training engine.\n"
        engine, data, train_time = train_gpmcc_chain(save_out=save_out, **learn_params)
    finally:
        return engine, data, train_time


def compare_covariance_matrix (engine, data):
    '''
    Compute Frobenius distance between empirical covariance matrices from data and gpmcc.
    '''
    num_samples = data.shape[0]
    Sigma_emp = np.cov(data.T)
    cov_dist = []
    for i_state in range(engine.num_states):
        samples_cc = engine.simulate(-1,[0,1,2],N = num_samples)[i_state,:,:]
        Sigma_cc = np.cov(samples_cc.T)
        cov_dist.append(norm(Sigma_cc - Sigma_emp))
    return cov_dist


## Computing mi and cmi
def compute_mi_cc(engine, num_samples = 1000, save_out=True):
    print("\nComputing Mutual Information with CrossCat\n")
    start_time = time.time()
    mi_XZ_cc = engine.mutual_information(0,2,N = num_samples)
    time_compute_mi = time.time() - start_time
    if save_out:
        file_mi = 'resources/mi_XZ_cc_pnts%d_stats%d_trans%d_samps%d' %(engine.num_points, engine.num_states, engine.num_transitions, num_samples)
        np.save(file_mi, [mi_XZ_cc, time_compute_mi])
    return mi_XZ_cc, time_compute_mi


def compute_cmi_cc(engine, num_samples = 1000, num_condition = 100, save_out=True):
    print("\nComputing Conditional Mutual Information with CrossCat\n")
    start_time = time.time()
    num_cond_per_state = int(np.ceil(num_condition * 1. / engine.num_states))
    Y_samples_cc = np.ndarray.flatten(engine.simulate(-1,[1],N = num_cond_per_state))
    mi_XZ_cond_Y = [engine.mutual_information(0,2,[(1, y_val)], N = num_samples) for y_val in Y_samples_cc] 
    cmi_XZ_gY_cc = np.mean(mi_XZ_cond_Y, axis=0)
    time_compute_cmi = time.time() - start_time
    if save_out:
        file_cmi = 'resources/cmi_XZ_cc_pnts%d_stats%d_trans%d_samps%d_conds%d' %(engine.num_points, engine.num_states, engine.num_transitions, num_samples, num_condition)
        np.save(file_cmi, [cmi_XZ_gY_cc, time_compute_cmi])    
    return cmi_XZ_gY_cc, time_compute_cmi


def compare_mutual_information(data,mi_XZ_cc, cmi_XZ_gY_cc):
    Sigma_emp = np.cov(data.T)
    mi_XZ_emp = gaussian_cmi(Sigma_emp, 0, 2, [])
    cmi_XZ_gY_emp = gaussian_cmi(Sigma_emp, 0, 2, [1])
    delta_mi = abs(mi_XZ_emp - mi_XZ_cc)
    delta_cmi = abs(cmi_XZ_gY_emp - cmi_XZ_gY_cc)
    return delta_mi, delta_cmi


##  Plot utilities ##
def scatter_data(engine, data, delta_cov, delta_mi, delta_cmi):
    '''
    Plot scatter plots of data (column 0 x column 2, color: column 1).
    Plots four scatter plots corresponding to:
    1. Original Data
    2. GPMCC-trained data for state with closest MI + CMI / 2
    3. GPMCC-trained data for state with closest Covariance Frobenius distance
    4. GPMCC-trained data for state with farthest Covariance Frobenius distance
    '''
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.set_size_inches(10,10)
    orig_data = data
    sim_data = engine.simulate(-1,[0,1,2],N=orig_data.shape[0])
    delta_mean = (delta_mi + delta_cmi)/2
    k_ix = [np.argmin(delta_mean), np.argmin(delta_cov), np.argmax(delta_cov)]                          
    def sub_scatter(data, ax, title):
        ax.scatter(data[:,0], data[:,2], c=data[:,1], s=20, marker='o', cmap=cm.jet)
        ax.set_ylabel('Z', fontsize=17)
        ax.set_xlabel('X', fontsize=17)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=17)  
    sub_scatter(orig_data, ax[0,0], 'Original Data (Color: Y)')
    sub_scatter(sim_data[k_ix[0], :, :], ax[0,1], 'Closest MI')
    sub_scatter(sim_data[k_ix[1], :, :], ax[1,0], 'Closest Covariance')
    sub_scatter(sim_data[k_ix[2], :, :], ax[1,1], 'Farthest Covariance')


def raster_cmi_state(mi_cc, mi, cmi_cc, cmi, delta_cov=None):
    '''
    Plot raster plot of MI(X;Z) and CMI(X;Z|Y) color-coded for
    all states. Size of vertical lines is largest (closest to 10) 
    for states whose Covariance Frobenius is smallest.
    '''
    fig, ax = plt.subplots(2, 1, sharey=True)
    fig.set_size_inches(10,7)
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=len(mi_cc))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    def raster(mi_cc, mi, a, ylabel, delta_cov=delta_cov):
        if delta_cov==None:
            delta_cov = np.zeros(len(mi_cc))
        else:
            assert len(delta_cov)==len(mi_cc)
        for i in range(len(mi_cc)):
            ax[a].vlines(mi_cc[i], 0, 10 - delta_cov[i], color=scalarMap.to_rgba(i), lw=3,  cmap = cm.jet)
            ax[a].vlines(mi, 10, 20, color='k', lw=3)
            ax[a].set_ylim([0,20])
            ax[a].margins(0.1,0.1)
            ax[a].set_ylabel(ylabel, fontsize=17)
    raster(mi_cc, mi, 0, ylabel='MI(X;Z)', delta_cov=delta_cov)
    raster(cmi_cc, cmi, 1, ylabel='CMI(X;Z|Y)', delta_cov=delta_cov)
    ax[1].set_xlabel('Mutual Information (Estimates)', fontsize=17)
    black = matplotlib.patches.Patch(color='k', label='Analytical')
    ax[0].legend(handles=[black], framealpha=0)    
                               

def dependence_heatmap(engine, data, delta_cov, delta_mi, delta_cmi):
    '''
    Plots heatmap for probability of dependence corresponding to:
    1. Original Data
    2. GPMCC-trained data for state with closest MI + CMI / 2
    3. GPMCC-trained data for state with closest Covariance Frobenius distance
    4. GPMCC-trained data for state with farthest Covariance Frobenius distance
    '''
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(7,7)
    sns.set(context="paper", font="monospace")
    orig_dep = (np.cov(data.T)>1e-3).astype(float)
    delta_mean = (delta_mi + delta_cmi)/2
    k_ix = [np.argmin(delta_mean), np.argmin(delta_cov), np.argmax(delta_cov)]
    def sub_heatmap(sigma, ax, title):
        sns.heatmap(sigma, square=True, ax=ax, vmin=0, vmax=1, cbar=False, \
                    linewidths=.5, xticklabels=['X','Y','Z'], yticklabels=['X','Y','Z'], cmap=cm.Greens)
        ax.set_title(title, fontsize=17)
    sub_heatmap(orig_dep, ax[0,0], 'Original Data')
    sub_heatmap(engine.dependence_probability_pairwise(states=[k_ix[0]]), ax[0,1], 'Closest MI')
    sub_heatmap(engine.dependence_probability_pairwise(states=[k_ix[1]]), ax[1,0], 'Closest Covariance')
    sub_heatmap(engine.dependence_probability_pairwise(states=[k_ix[2]]), ax[1,1], 'Farthest Covariance')
    fig.tight_layout()


def scatter_cmi_state(delta_cov, mi_cc, cmi_cc, mi_emp=None, cmi_emp=None):
    '''
    Plot scatter plot of MI(X;Z) and CMI(X;Z|Y) versus Covariance Frobenius distance 
    color-coded for all states.
    '''
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(10,7)
    color_code = np.array(range(len(delta_cov)))
    my_cmap = cm.jet
    ax[0].scatter(delta_cov, mi_cc, s=50, c=color_code, marker='o', cmap=my_cmap)
    if mi_emp != None:
        ax[0].axhline(mi_emp, c = 'k', ls='--', label='Analytical')    
    ax[0].set_ylabel('$MI(X;Z)$', fontsize=18)
    ax[0].legend(framealpha=0)
    ax[0].margins(0.1,0.1)
    ax[1].scatter(delta_cov, cmi_cc, s=50, c=color_code, marker='o', cmap=my_cmap)
    if cmi_emp != None:
        ax[1].axhline(cmi_emp, c = 'k', ls='--', label='Analytical')
    ax[1].set_xlabel('$\Delta_{cov} = ||\Sigma_{cc} - \Sigma_{emp}||_{F}$', fontweight='bold', fontsize=18)
    ax[1].set_ylabel('$CMI(X;Z|Y)$', fontsize=18)
    ax[1].legend(framealpha=0)
    ax[1].margins(0.1, 0.1)
