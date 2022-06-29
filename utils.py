from random import randint
import numpy as np
from scipy import signal
from scipy import io
from scipy.fft import fft, fftfreq
import visualizer as vsl
from pathlib import Path
import pickle


def attach_voltmeter(nest, pop_list, sampling_resolution=1.0, target_neurons='all'):
    """ Function to attach a voltmeter to all populations in list
        Returns a list of vms coherent with passed population
        Default resolution is 1 ms                                  """
    vm_list = []
    for id, pop in enumerate(pop_list):
        vm = nest.Create('voltmeter', params={'interval': sampling_resolution, 'record_from': ["V_m"],
                                              'to_file': False})
        if target_neurons == 'all':
            nest.Connect(vm, pop)
        elif target_neurons == 'random':
            idx = randint(0, max(pop) - min(pop))
            print(f'max {max(pop)}, min {min(pop)}, selected {idx}')
            nest.Connect(vm, [pop[idx]])
        elif target_neurons == 'one-by-one':
            idx = id
            nest.Connect(vm, [pop[idx]])
        else:
            idx = target_neurons
            nest.Connect(vm, [pop[idx]])
        vm_list = vm_list + [vm]
    return vm_list


def attach_spikedetector(nest, pop_list, pop_list_to_ode=None, sd_list_to_ode=None):
    """ Function to attach a spike_detector to all populations in list
        Returns a list of vms coherent with passed population    """
    sd_list = []

    if pop_list_to_ode is not None:     # if there is no spike detector already set, load it
        pop_list_to_ode_pointer = 0
        for pop in pop_list:
            if pop_list_to_ode[pop_list_to_ode_pointer] == pop:
                sd_list = sd_list + [sd_list_to_ode[pop_list_to_ode_pointer]]
                if pop_list_to_ode_pointer < len(pop_list_to_ode):
                    pop_list_to_ode_pointer += 1
            else:
                sd = nest.Create('spike_detector', params={'to_file': False})
                nest.Connect(pop, sd)
                sd_list = sd_list + [sd]

    else:       # if there is no spike detector already set
        for pop in pop_list:
            sd = nest.Create('spike_detector', params={'to_file': False})
            nest.Connect(pop, sd)
            sd_list = sd_list + [sd]
    return sd_list


def get_voltage_values(nest, vm_list, pop_names):
    """ Function to select mean voltage and time from voltmeter events
        Returns a list of dictionaries with potentials and times  """
    dic_list = []
    for vm, name in zip(vm_list, pop_names):
        potentials = nest.GetStatus(vm, "events")[0]["V_m"]
        times = nest.GetStatus(vm, "events")[0]["times"]
        dic = {'times': times, 'potentials': potentials, 'compartment_name': name}
        dic_list = dic_list + [dic]
    return dic_list


def get_spike_values(nest, sd_list, pop_names):
    """ Function to select spike idxs and times from spike_det events
        Returns a list of dictionaries with spikes and times  """
    dic_list = []
    for sd, name in zip(sd_list, pop_names):
        spikes = nest.GetStatus(sd, "events")[0]["senders"]
        times = nest.GetStatus(sd, "events")[0]["times"]
        dic = {'times': times, 'neurons_idx': spikes, 'compartment_name': name}
        dic_list = dic_list + [dic]
    return dic_list

def get_weights_values(nest, weights_recorder):
    """ Function to select mean voltage and time from voltmeter events
        Returns a list of dictionaries with potentials and times  """

    dic_list = []

    weights = nest.GetStatus(weights_recorder, "events")[0]["weights"]
    times = nest.GetStatus(weights_recorder, "events")[0]["times"]
    senders = nest.GetStatus(weights_recorder, "events")[0]["senders"]
    targets = nest.GetStatus(weights_recorder, "events")[0]["targets"]

    # for s_i, t_i in zip([67838, 22216, 80039], [95457, 95457, 95525]):
    for s_i, t_i in zip([7714, 19132], [95514, 95473]):
        idx = [s == s_i and t == t_i for s, t in zip(senders, targets)]
        dic = {'times': times[idx], 'weights': weights[idx], 'sender_receiver': f's = {s_i}, t = {t_i}'}
        dic_list = dic_list + [dic]

    return dic_list


def create_model_dictionary(N_neurons, pop_names, pop_ids, sim_time, sample_time=None, settling_time=None, trials=None, b_c_params=None):
    """ Function to create a dictionary containing model parameters  """
    dic = {}
    dic['N_neurons'] = N_neurons
    dic['pop_names_list'] = pop_names
    dic['pop_ids'] = pop_ids
    dic['simulation_time'] = sim_time
    dic['sample_time'] = sample_time
    dic['settling_time'] = settling_time
    dic['trials'] = trials
    dic['b_c_params'] = b_c_params
    return dic


def calculate_fr_stats(raster_list, pop_dim_ids, t_start=0., t_end=None):
    """ Function to evaluate the firing rate and the
    coefficient of variation of the inter spike interval"""
    fr_list, CV_list, name_list = calculate_fr(raster_list, pop_dim_ids, t_start, t_end, return_CV_name=True)

    pop_list_dim = get_pop_dim_from_ids(name_list, pop_dim_ids)
    # expand with average values for GPe and MSN if there are
    if 'MSND1' in name_list and 'GPeTA' in name_list:
        average_MSN = lambda list: round(
            (list[1] * pop_list_dim[1] + list[2] * pop_list_dim[2]) / (pop_list_dim[1] + pop_list_dim[2]), 2)
        average_GPe = lambda list: round(
            (list[3] * pop_list_dim[3] + list[4] * pop_list_dim[4]) / (pop_list_dim[3] + pop_list_dim[4]), 2)
        fr_list = fr_list[0:3] + [average_MSN(fr_list)] + fr_list[3:5] + [average_GPe(fr_list)] + fr_list[5:7]
        CV_list = CV_list[0:3] + [average_MSN(CV_list)] + CV_list[3:5] + [average_GPe(CV_list)] + CV_list[5:7]
        name_list = name_list[0:3] + ['MSN'] + name_list[3:5] + ['GPe'] + name_list[5:7]

    # elif 'GPeTA' in name_list:
    #     average_GPe = lambda list: round(
    #         (list[3] * pop_list_dim[3] + list[4] * pop_list_dim[4]) / (pop_list_dim[3] + pop_list_dim[4]), 2)
    #     fr_list = fr_list[0:3] + fr_list[3:5] + [average_GPe(fr_list)] + fr_list[5:7]
    #     CV_list = CV_list[0:3] + CV_list[3:5] + [average_GPe(CV_list)] + CV_list[5:7]
    #     name_list = name_list[0:3] + name_list[3:5] + ['GPe'] + name_list[5:7]

    ret = {'fr': fr_list, 'CV': CV_list, 'name': name_list}
    return ret


def calculate_fr(raster_list, pop_dim_ids, t_start=0., t_end=None, return_CV_name=False):
    """ Function to evaluate the firing rate and the
    coefficient of variation of the inter spike interval"""
    fr_list = []
    if return_CV_name:
        CV_list = []
        name_list = []
    min_idx = 0  # useful to process neurons indexes

    if t_end == None:
        t_end = np.inf

    for raster in raster_list:
        pop_name = raster['compartment_name']
        pop_dim = pop_dim_ids[pop_name][1] - pop_dim_ids[pop_name][0] + 1
        t_prev = -np.ones(pop_dim)  # to save the last spike time for idx-th neuron
        ISI_list = [[] for _ in range(pop_dim)]  # list of list, will contain the ISI for each neuron
        for tt, idx in zip(raster['times'], raster['neurons_idx'] - pop_dim_ids[pop_name][0] - 1):
            if tt > t_start:  # consider just element after t_start
                if tt < t_end:
                    if t_prev[idx] == -1:  # first spike of the neuron
                        t_prev[idx] = tt
                    else:
                        ISI = (tt - t_prev[idx])  # inter spike interval
                        if ISI != 0:
                            ISI_list[idx] = ISI_list[idx] + [ISI]
                            t_prev[idx] = tt  # update the last spike time

        # we calculate the average ISI for each neuron, comprehends also neurons with fr = 0
        inv_mean_ISI = np.array([1000. / (sum(elem) / len(elem)) if len(elem) != 0 else 0. for elem in ISI_list])
        fr = inv_mean_ISI.mean()
        fr_list = fr_list + [round(fr, 2)]
        if return_CV_name:
            CV_el = np.array([np.array(sublist).std() / np.array(sublist).mean() if len(sublist) != 0 else 0. for sublist in ISI_list])
            CV_list = CV_list + [round(CV_el.mean(), 2)]
            # ISI_array = np.array([item for sublist in ISI_list for item in sublist])  # flat the ISI array
            # CV_list = CV_list + [round(ISI_array.std() / ISI_array.mean(), 2) if len(ISI_array) != 0 else 0.]   # calculate CV between all of the ISI of that population
            name_list = name_list + [raster['compartment_name']]

    if return_CV_name:  # return also ISI array as flatten np.array
        return fr_list, CV_list, name_list
    else:
        return fr_list

def add_spikes_to_potential(rasters, index, ax, pot_min=-50, pot_max=0):
    min_idx = min(rasters[index]['neurons_idx'])
    condition = lambda x: x == min_idx
    mask = [i for i, idx in enumerate(rasters[index]['neurons_idx']) if condition(idx)]
    for el in rasters[index]['times'][mask]:
        ax[index].plot([el, el], [pot_max, pot_min], c='tab:blue')


def get_pop_dim(pop_list):
    ''' Given a population list, return a list of pop dimentions '''
    dim_list = []

    for el in pop_list:
        dim = len(el)
        dim_list = dim_list + [dim]

    return dim_list

def get_pop_dim_from_ids(pop_list, pop_ids):
    ''' Given a dictionary of pop indexes, return a list of pop_dimention '''
    dim_list = []

    for pop in pop_list:
        dim = pop_ids[pop][1] - pop_ids[pop][0] + 1
        dim_list = dim_list + [dim]

    return dim_list

def gaussian(x, mu, sig):
    g = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)
    return g/g.sum()

def rectangular(x, center, widht):
    y = np.zeros(x.shape[0])
    for idx, el in enumerate(x):
        if el < center - widht/2:
            y[idx] = 0
        elif el > center + widht/2:
            y[idx] = 0
        else:
            y[idx] = 1
        pass
    return y/y.sum()


def calculate_fourier_idx(xf, range):
    lower = range[0]
    upper = range[1]

    xf_lower = np.abs(xf - lower)
    # min_idx = xf_lower.where(min(np.abs(xf_lower)))
    min_idx = np.where(xf_lower <= min(xf_lower+0.01))
    xf_upper = np.abs(xf - upper)
    # max_idx = xf_upper.where(max(np.abs(xf_upper)))
    max_idx = np.where(xf_upper <= min(xf_upper))
    return [min_idx[0][0], max_idx[0][0]+1]


def calculate_hyperbolic_interpol(yf, xf, rng):
    idxs = calculate_fourier_idx(xf, rng)
    yf_restr = yf[idxs[0]:idxs[1], :]
    min_val = yf_restr.min(axis=0)
    min_freq_idx = [idxs[0] + np.where(yf_restr[:, i] == min_val[i]) for i in range(len(min_val))]
    min_freq = np.concatenate([xf[i][0] for i in min_freq_idx])
    return min_val/(1./min_freq)


def fitness_function(fr, fr_target, mass_fr, T_sample, filter_range, filter_sd, t_start=0., fr_weights=None):
    '''
    Evaluate fitness function
    :param fr: average firing rate calculated from simulation raster plots
    :param fr_target: the desired value of the average firing rate
    :param mass_fr: the mass model firing rate over time, will be analyzed in frequencies
    :param T_sample: sampling time of the mass models, necessary for Fourier transform
    :param mean: of the gaussian filter applied to extract Fourier fitness
    :param sd: of the gaussian filter applied to extract Fourier fitness
    :param t_start: if you want to neglect the first simulation instants
    :param fr_weights: weights of the fr distances
    :return: return teh fitness
    '''

    # # evaluate fourier transform
    # T = T_sample / 10.
    # mass_fr = mass_fr[int(t_start / T):]
    # N = mass_fr.shape[0]
    #
    # yf = fft(mass_fr)
    # yf = 2.0 / N * np.abs(yf[0:N // 2])
    # xf = fftfreq(N, d=T/1000)[:N // 2]     # take just pos freq
    # fourier_idx = calculate_fourier_idx(xf, [mean-sd*3, mean+sd*3])
    # print(f'Considering frequencies in the range {xf[fourier_idx[0]], xf[fourier_idx[1] - 1]}')
    # fourier = yf[fourier_idx[0]:fourier_idx[1]]
    #
    # dim = fourier_idx[1] - fourier_idx[0]
    # freq_p = np.linspace(-10, 10, dim, endpoint=True)
    # kernel_f = gaussian(freq_p, 0., sd)
    # # vsl.simple_plot(freq_p+40, kernel_f)
    #
    # # freq_p = np.linspace(30, 50, dim, endpoint=True)
    # # kernel_f = rectangular(freq_p, 42.5, 5)
    # conv = np.dot(fourier, kernel_f)

    # evaluate wavelet transform
    T = T_sample / 10.
    y = mass_fr[int(t_start / T):, :]  # calculate tf after the t_start
    T = T_sample  # resample to 1 ms
    y = y[::10]  # select one time sample every 10

    fs = 1000. / T  # Hz
    w = 15  # []
    freq = np.linspace(17, fs / 2, 2 * int(fs / 2 - 17 + 1))  # frequency range
    widths = w * fs / (2 * freq * np.pi)  # reduce time widths for higher frequencies

    wt = np.zeros((len(freq), y.shape[1]))
    for idx in range(y.shape[1]):
        cwtm = signal.cwt(y[:, idx], signal.morlet2, widths, w=w)
        wt[:, idx] = np.abs(cwtm).sum(axis=1)  # /(np.abs(cwtm)).sum()

    wavelet_idx = calculate_fourier_idx(freq, filter_range)
    print(f'In fitness: considering frequencies in the range {[freq[wavelet_idx[0]], freq[wavelet_idx[1] - 1]]}')

    dim = wavelet_idx[1] - wavelet_idx[0]
    freq_p = np.linspace(-10, 10, dim, endpoint=True)
    kernel_f = gaussian(freq_p, 0., filter_sd)
    conv = (np.dot(wt[wavelet_idx[0]:wavelet_idx[1], :].T, kernel_f)).sum()

    # evaluate fr accuracy
    dist = np.array(fr) - fr_target
    if fr_weights is not None:
        dist = dist * fr_weights
    dist = np.power(dist, 2)
    fitness_Cereb = np.sum(dist)
    fitness_fourier = conv / 2
    # fitness = 1.0 / (fitness_Cereb + fitness_fourier)
    fitness = - fitness_Cereb + fitness_fourier

    print(f'fitness_firing_rate = {"%.2f" % - fitness_Cereb}, fitness_fourier = {"%.2f" % fitness_fourier}, fitness = {"%.2f" % fitness}')

    return fitness


def minimumJerk(x_init, x_des, final_time):
    T_max = final_time

    a = 6 * (x_des - x_init) / np.power(T_max, 5)
    b = -15 * (x_des - x_init) / np.power(T_max, 4)
    c = 10 * (x_des - x_init) / np.power(T_max, 3)
    g = x_init

    pp = lambda t: a * np.power(t, 5) + b * np.power(t, 4) + c * np.power(t, 3) + g
    pp_dt = lambda t: (5*a * np.power(t, 4) + 4*b * np.power(t, 3) + 3*c * np.power(t, 2)) * 1000

    g0 = 9.81
    l = 1.  # m
    m = 1.  # kg
    I = (m * l ** 2) / 3
    pp_dtdt = lambda t: (20*a * np.power(t, 3) + 12*b * np.power(t, 2) + 6*c * np.power(t, 1)) * 1000 * 1000
    jerk = lambda t: (60*a * np.power(t, 2) + 24*b * np.power(t, 1) + 6*c) * 1000 * 1000 * 1000

    tau_forw = lambda t: pp_dtdt(t) * I + g0 * m * l / 2 * np.sin(pp(t))

    return [pp], [pp_dt], [pp_dtdt], [jerk], tau_forw

def circular_traj_joints(sim_time):
    circular_trajectory = io.loadmat('misc/desired_trajectory.mat')
    j0_traj = lambda tt: circular_trajectory['j0'][0][int(tt / sim_time * 5000. % 5000)]
    j1_traj = lambda tt: circular_trajectory['j1'][0][int(tt / sim_time * 5000. % 5000)]
    jd0_traj = lambda tt: circular_trajectory['jd0'][0][int(tt / sim_time * 5000. % 5000)]
    jd1_traj = lambda tt: circular_trajectory['jd1'][0][int(tt / sim_time * 5000. % 5000)]
    des_traj = [j0_traj, j1_traj]
    des_traj_vel = [jd0_traj, jd1_traj]
    return des_traj, des_traj_vel

def get_cortex_activity(dopa_depl, sim_time, sim_period):
    if str(Path.home()) == '/home/marco':
        if dopa_depl == 0:
            mass_frs_path = str(Path.home()) + '/BGs-Cereb-nest/shared_results/complete_5500ms_sol17/mass_frs'
        else:
            mass_frs_path = str(
                Path.home()) + f'/BGs-Cereb-nest/shared_results/complete_3000ms_sol17_dopadepl_{int(-dopa_depl * 10)}/mass_frs'
    elif str(Path.home()) == '/home/gambosi':
        if dopa_depl == 0:
            mass_frs_path = str(Path.home()) + '/BGs-Cereb-nest/shared_results/complete_5500ms_sol17/mass_frs'
        else:
            mass_frs_path = str(
                Path.home()) + f'/BGs-Cereb-nest/shared_results/complete_3000ms_sol17_dopadepl_{int(-dopa_depl * 10)}/mass_frs'
    else:
        if dopa_depl == 0:
            mass_frs_path = str(Path.home()) + '/Desktop/BGs-Cereb-nest/shared_results/complete_5500ms_sol17/mass_frs'
        else:
            mass_frs_path = str(
                Path.home()) + f'/Desktop/BGs-Cereb-nest/shared_results/complete_3000ms_sol17_dopadepl_{int(-dopa_depl * 10)}/mass_frs'
    with open(mass_frs_path, 'rb') as pickle_file:
        mass_frs = pickle.load(pickle_file)

    mass_frs = mass_frs[::10]
    ctx_frs = mass_frs[:, 0]
    extended_ctx_frs = np.tile(ctx_frs[1000:], 3)

    return extended_ctx_frs
