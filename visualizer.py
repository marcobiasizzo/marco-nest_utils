import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal
from marco_nest_utils import utils
from fooof import FOOOF
import pickle
from math import ceil

''' Plot membrane potential '''


def plot_potential(times, potentials, compartment_name, ax_plt=None, time_sorting=True, t_start=0.):
    if ax_plt is None:
        fig, ax = plt.subplots()  # create a new figure
    else:
        ax = ax_plt
    if time_sorting:  # sort along time before plotting, useful if running multiple cores
        p = times.argsort()
        if not (times == times[p]).all():  # if sorting is needed
            print('Data is sorted according to time before plotting!')
            potentials = potentials[p]
            times = times[p]
    ax.plot(times, potentials, label=f'{compartment_name} potential', color='tab:blue')
    ax.axvline(x=t_start, color='tab:red')
    # ax.scatter(spike_times, [-70.] * len(spike_times), label='Input spikes', color='tab:orange')
    ax.set_xlabel('Time [ms]')  # Add an x-label to the axes.
    ax.set_ylabel(f'{compartment_name} mem. pot. [mV]')  # Add a y-label to the axes.
    # ax.set_title(f"{compartment_name} potential")  # Add a title to the axes.
    # ax.legend()  # Add a legend.
    if ax_plt is None:
        res = fig, ax
    else:
        res = ax
    return res


''' Plot multiple membrane potentials '''


def plot_potential_multiple(potentials_list, time_sorting=True, clms=2, t_start=0.):
    keywords = {'x_data': 'times', 'y_data': 'potentials', 'name': 'compartment_name'}
    return multiple_plots(potentials_list, plot_potential, keywords, clms=clms, time_sorting=time_sorting,
                          t_start=t_start)


''' Plot a raster plot '''


def raster_plot(times, neurons_idx, compartment_name, ax_plt=None, start_stop_times=None, t_start=0., n_joints=None, ctx_in=None):
    if ax_plt is None:
        fig, ax = plt.subplots()  # create a new figure
    else:
        ax = ax_plt
    if start_stop_times is not None:
        scale_xy_axes(ax, xlim=start_stop_times)  # x range
    if len(neurons_idx) > 0:  # is there are spikes
        if compartment_name == 'dcn': compartment_name = 'dcnp'
        elif compartment_name == 'dcnp': compartment_name = 'dcni'

        if compartment_name == 'IO':
            if n_joints is not None:
                if n_joints == 1:
                    neg_idxs = [n in range(6373, 6409) for n in neurons_idx]
                    pos_idxs = [n in range(6409, 6445) for n in neurons_idx]
                elif n_joints == 2:
                    neg_idxs = [n in list(range(6373, 6391)) + list(range(6409, 6427)) for n in neurons_idx]
                    pos_idxs = [n in list(range(6391, 6409)) + list(range(6427, 6445)) for n in neurons_idx]
                ax.scatter(times[neg_idxs], neurons_idx[neg_idxs], c='tab:orange', s=4)
                ax.scatter(times[pos_idxs], neurons_idx[pos_idxs], c='tab:blue', s=4)
        elif compartment_name == 'io':
            neg_idxs = [n in range(96756, 96762) for n in neurons_idx]
            ax.scatter(times[neg_idxs], neurons_idx[neg_idxs], c='tab:orange', s=4)
            pos_idxs = [n in range(96762, 96768) for n in neurons_idx]
            ax.scatter(times[pos_idxs], neurons_idx[pos_idxs], c='tab:blue', s=4)


        elif compartment_name == 'DCN':
            if n_joints is not None:
                if n_joints == 1:
                    neg_idxs = [n in range(6445, 6463) for n in neurons_idx]
                    pos_idxs = [n in range(6463, 6481) for n in neurons_idx]
                elif n_joints == 2:
                    neg_idxs = [n in list(range(6445, 6454)) + list(range(6463, 6472)) for n in neurons_idx]
                    pos_idxs = [n in list(range(6454, 6463)) + list(range(6472, 6481)) for n in neurons_idx]
                ax.scatter(times[neg_idxs], neurons_idx[neg_idxs], c='tab:orange', s=4)
                ax.scatter(times[pos_idxs], neurons_idx[pos_idxs], c='tab:blue', s=4)
        # print(times)
        elif compartment_name == 'dcn':
            neg_idxs = [n in [96735, 96736, 96737, 96741, 96742, 96743] for n in neurons_idx]
            ax.scatter(times[neg_idxs], neurons_idx[neg_idxs], c='tab:orange', s=4)
            pos_idxs = [n in [96732, 96733, 96734, 96738, 96739, 96740] for n in neurons_idx]
            ax.scatter(times[pos_idxs], neurons_idx[pos_idxs], c='tab:blue', s=4)
        # print(times)
        elif compartment_name == 'MF' and ctx_in is not None:
            ax.scatter(times, neurons_idx, c='tab:blue', s=4)
            ax.plot(np.linspace(0, len(ctx_in)*10, len(ctx_in)), min(neurons_idx) + ctx_in / max(ctx_in) * (max(neurons_idx) - min(neurons_idx)), c='tab:orange', linewidth=2., alpha=0.8)
        else:
            ax.scatter(times, neurons_idx, c='tab:blue', s=4)
            if compartment_name == 'purkinje':
                mylist = sorted(neurons_idx)
                mylist2 = list(dict.fromkeys(mylist))
                occ = []
                for key in mylist2:
                    occr = mylist.count(key)
                    occ += [occr]
                # print(compartment_name, len(mylist), mylist)
                print(occ)
        max_idx = max(neurons_idx)
        min_idx = min(neurons_idx)
        if max_idx - min_idx >= 4:
            scale_xy_axes(ax, ylim=[min_idx, max_idx])  # y range
            number_of_ticks = 5  # with min and max too
            yticks = [min_idx + int((max_idx - min_idx) * i / (number_of_ticks - 1)) for i in range(number_of_ticks)]
        else:
            yticks = list(dict.fromkeys(neurons_idx))  # select elements without repets
        ax.set_yticks(yticks)
    else:
        scale_xy_axes(ax, ylim=[0, 1])
        ax.set_yticks([0, 1])
    ax.axvline(x=t_start, color='tab:red')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(f'{compartment_name} neuron idx')
    # ax.set_title(f"{compartment_name} raster plot")
    if ax_plt is None:
        res = fig, ax
    else:
        res = ax
    return res


''' Plot a multiple rasterplot in 2 columns'''


def raster_plots_multiple(rasters_list, start_stop_times=None, clms=2, t_start=0., n_joints=None,  ctx_in=None):
    keywords = {'x_data': 'times', 'y_data': 'neurons_idx', 'name': 'compartment_name'}
    fig, axs = multiple_plots(rasters_list, raster_plot, keywords, clms=clms, start_stop_times=start_stop_times,
                              t_start=t_start, n_joints=n_joints,  ctx_in=ctx_in)
    return fig, axs


''' UTILS '''

''' Scale x and y axes '''


def scale_xy_axes(ax, xlim=None, ylim=None):
    if xlim is not None:
        if xlim != [None, None]:
            range_norm = xlim[1] - xlim[0]
            border = range_norm * 5 / 100  # leave a 5% of blank space at borders
            ax.set_xlim(xlim[0] - border, xlim[1] + border)
    if ylim is not None:
        if ylim != [None, None]:
            range_norm = ylim[1] - ylim[0]
            border = range_norm * 5 / 100  # leave a 5% of blank space at borders
            ax.set_ylim(ylim[0] - border, ylim[1] + border)


''' Multiple plots framework '''


def multiple_plots(plot_list, plot_function, keywords, clms=2, ext_axs=None, **kwargs):
    n_plots = len(plot_list)
    rows = int(np.ceil(n_plots / clms))
    fig_width = 8.0
    plot_height = 3.0
    if ext_axs is None:
        fig, axs = plt.subplots(rows, clms, figsize=(fig_width * clms, plot_height * rows))
    else:
        axs = ext_axs  # if provided, use external axes
    if n_plots % clms != 0:  # some empty plots at the and
        for idx in range(clms - 1):
            if ext_axs is None:
                fig.delaxes(axs[rows - 1, clms - 1 - idx])  # delete last plots
    for i, plot in enumerate(plot_list):
        if clms == 1:
            index = i
        else:
            index = int(i / clms), i % clms  # if not 1 column, manage 2 indexes
        ax = plot_function(plot[keywords['x_data']], plot[keywords['y_data']], plot[keywords['name']],
                           ax_plt=axs[index], **kwargs)
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(plot_height * 5)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(plot_height * 4)
        if (n_plots - 1 - i) >= clms:  # not the last #clms plots
            ax.set_xlabel(None)
        # if int(i % clms) != 0:          # not the first column
        #     ax.set_ylabel(None)
    if ext_axs is None:
        fig.tight_layout()
        return fig, axs
    else:
        ax.set_xlabel(None)
        return axs


def plot_my_histogram(ax, width, value_list, name_list, target_list, y_label):
    x = list(range(len(value_list)))
    t = []
    tval = []

    if target_list is not None:
        for idx, val in enumerate(target_list):
            if val != 0:
                t = t + [x[idx] + width]  # place it the right
                tval = tval + [np.round(val, 2)]

    ax.set_xticks(np.array(x))
    ax.set_xticklabels(name_list)

    width = width * 0.9
    bars1 = ax.bar(x, value_list, width, label='Simulation')
    ax.bar_label(bars1, rotation=60)

    if target_list is not None:
        bars2 = ax.bar(t, tval, width, label='Reference')
        ax.bar_label(bars2, rotation=60)

    if target_list is not None:
        ax.set_ylim(0, max(np.max(value_list), np.max(target_list)) * 1.4)
    else:
        ax.set_ylim(0, np.max(value_list) * 1.4)

    ax.legend(loc='upper left')   # , bbox_to_anchor=(0.14, 1.0))

    ax.set_ylabel(y_label)

    ax.tick_params(bottom=False)

def plot_instant_fr(times, instant_fr, compartment_name, ax_plt=None, t_start=0.):
    if ax_plt is None:
        fig, ax = plt.subplots()  # create a new figure
    else:
        ax = ax_plt

    pop_instant_fr = instant_fr.sum(axis=0) / 1000.     # every [ms]

    ax.plot(times, pop_instant_fr, label=f'{compartment_name} potential', color='tab:blue')
    # ax.axvline(x=t_start, color='tab:red')
    ax.set_xlabel('Time [ms]')  # Add an x-label to the axes.
    ax.set_ylabel(f'{compartment_name} a.f.r. [sp./ms]')  # Add a y-label to the axes.
    # ax.set_title(f"{compartment_name} potential")  # Add a title to the axes.
    # ax.legend()  # Add a legend.
    if ax_plt is None:
        res = fig, ax
    else:
        res = ax
    return res


def plot_instant_fr_multiple(instant_fr_list, clms=2, t_start=0.):
    keywords = {'x_data': 'times', 'y_data': 'instant_fr', 'name': 'name'}
    return multiple_plots(instant_fr_list, plot_instant_fr, keywords, clms=clms,
                          t_start=t_start)


def plot_fr_learning1(average_fr_per_trial, experiment, pop_name, labels=None):
    fig_width = 7.0
    plot_height = 5.0

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))
    number_shift = len(average_fr_per_trial) - 1
    time_shift = np.linspace(-number_shift, number_shift, number_shift+1) * 0.05

    for av_fr_per_trial, t_shift in zip(average_fr_per_trial, time_shift):
        averages = np.array(av_fr_per_trial[pop_name])
        times = np.linspace(1 + t_shift, len(averages) + t_shift, len(averages))

        av = np.mean(averages, axis=1)
        sd = np.std(averages, axis=1)
        tm = np.array(times)

        # ax.plot(tm, av, marker='o')
        plt.errorbar(tm, av, sd, marker='o')

    # scale_xy_axes(ax, ylim=[43, 70])

    ax.set_xlabel('Trial')
    ax.set_ylabel('Average firing rate [sp/s]')

    ax.set_title(f'Average firing rate of {pop_name} over trials, with {experiment}')
    ax.legend(labels)
    return fig, ax

def plot_fr_learning2(instant_fr_io, t_start, t_end, t_pre, trials, pop, labels=None):
    fig_width = 7.0
    plot_height = 5.0

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))
    number_shift = len(instant_fr_io) - 1
    time_shift = np.linspace(-number_shift, number_shift, number_shift+1) * 25

    for io_fr, t_shift in zip(instant_fr_io, time_shift):
        averages = []
        sds = []
        times = []

        for k in range(trials):
            t0 = t_pre + t_end * k          # trial init
            tf = t0 + t_start       # before IO spikes

            sel_times1 = io_fr['times'] < tf
            sel_times2 = io_fr['times'] >= t0
            sel_times = np.logical_and(sel_times1, sel_times2)
            sel_fr = io_fr['instant_fr'].mean(axis=0)[sel_times]

            average = np.mean(sel_fr)
            sd = np.std(sel_fr)

            averages += [average]
            sds += [sd]
            times += [(t0+tf)*0.5 + t_shift]

        av = np.array(averages)
        sd = np.array(sds)
        tm = np.array(times)

        # ax.scatter(tm, av)
        plt.errorbar(tm, av, sd, marker='o')

    ax.set_title(f'Average firing rate of {pop} over trials')
    ax.legend(labels)
    return fig, ax


def firing_rate_histogram(fr_list, name_list, CV_list=None, target_fr=None, target_CV=None):
    ''' Bar plot of the avarage firing rates of the population rasters '''

    fig_width = 6.0
    width = 0.4  # columns width
    plot_height = 2.5

    if CV_list == None:
        rows = 1  # one for fr
    else:
        rows = 2  # one for fr and onw fo CV
    fig, axes = plt.subplots(rows, 1, figsize=(fig_width, plot_height * rows))
    if rows == 1: axes = [axes]

    # plot afr
    plot_my_histogram(axes[0], width, fr_list, name_list, target_fr, 'Firing rate [Hz]')

    if CV_list != None:
        # plot CV
        plot_my_histogram(axes[1], width, CV_list, name_list, target_CV, 'CV []')

    fig.suptitle(f'Average firing rate')  # in mode: {mode}')

    fig.tight_layout()

    return fig, axes


def firing_rate_histogram_old(fr_list, CV_list, name_list, dopa_depl, mode, target_fr=None):
    ''' Bar plot of the avarage firing rates of the population rasters '''

    fig_width = 8.0
    width = 0.3         # columns width
    plot_height = 2.5
    rows = 1            # one for fr and onw fo CV
    fig, axes = plt.subplots(rows, 1, figsize=(fig_width, plot_height * rows))
    if rows == 1: axes = [axes]

    if dopa_depl:
        state = 'lesion'
    else:
        state = 'control'

    # reference fr
    lind_fr = {'control': {'slow': [0, 0, 0, 0, 0, 0, 26.0, 11.0, 0],
                           'active': [0, 0, 0, 0, 0, 0, 30.5, 12.0, 0],
                           'cereb': [0] * 5,
                           'complete': [0] * 3 + [0, 0, 30.5, 12.0, 0],
                           'invitro': [0, 0, 0, 0, 8.0, 18.0, 0, 10.0, 15.0]},
               'lesion': {'slow': [0, 0, 0, 0, 9, 21.5, 0, 20.5, 0],
                          'active': [0, 0, 0, 0, 22, 15, 0, 30.0, 0],
                          'cereb': [0] * 5,
                           'complete': [0] * 3 + [22, 15, 0, 30.0, 0],
                          'invitro': [0, 0, 0, 0, 8.0, 18.0, 0, 10.0, 15.0]}}
    ref_fr = {'control': {'slow': [0, 0, 0, 0, 0, 0, 25.9, 13.0, 0],
                          'active': [0, 0, 0, 0, 0, 0, 33.7, 15.0, 0],
                          'cereb': [0] * 5,
                           'complete': [0] * 3 + [0, 0, 33.7, 15.0, 0],
                          'invitro': [0, 0, 0, 0, 8.0, 18.0, 0, 10.0, 15.0]},
              'lesion': {'slow': [0, 0, 0, 0, 10.5, 23, 0, 20.5, 0],
                         'active': [0, 0, 0, 0, 19, 14, 0, 32.0, 0],
                         'cereb': [0] * 5,
                           'complete': [0] * 3 + [19, 14, 0, 32.0, 0],
                         'invitro': [0, 0, 0, 0, 8.0, 18.0, 0, 10.0, 15.0]}}
    bounds_fr = {'control': {'slow': [[10, 0, 0, 0.01, 0, 0, 0, 0, 20], [20, 0, 0, 2.0, 0, 0, 0, 0, 35]],
                             'active': [[0] * 9, [0] * 9],
                             'cereb': [[0] * 5, [0] * 5],
                           'complete': [0] * 3 + [19, 14, 0, 32.0, 0],
                             'invitro': [[0] * 5, [0] * 5]},
                 'lesion': {'slow': [[10, 0, 0, 0.01, 0, 0, 0, 0, 20], [20, 0, 0, 2.0, 0, 0, 0, 0, 35]],
                             'active': [[0] * 9, [0] * 9],
                             'cereb': [[0] * 5, [0] * 5],
                             'complete': [[0] * 8, [0] * 8],
                             'invitro': [[0] * 5, [0] * 5]}}

    ax = axes[0]
    x = list(range(len(fr_list)))

    ax.set_facecolor('#f0f7f4')
    fig.patch.set_facecolor('#f0f7f4')

    plt.rcParams.update({'text.color': "#2d2a32",
                         'axes.labelcolor': "#2d2a32",
                         'axes.edgecolor': "#2d2a32",
                         'axes.labelcolor': "#2d2a32",
                         'legend.facecolor': '#f0f7f4',
                         'xtick.color': "#2d2a32",
                         'xtick.labelcolor': "#2d2a32",
                         'legend.labelcolor': "#2d2a32"})

    y1 = y2 = z = t = []  # will place in y1, y2 Lindhal and exp refs. In z boundaries
    y1val = y2val = zvalu = zvall = tval = []  # save elements before plotting
    # for idx, val in enumerate(lind_fr[state][mode]):
    #     if val != 0:
    #         y1 = y1 + [x[idx]]              # place Lindhal in the centre
    #         y2 = y2 + [x[idx] + width]      # place Mallet on the right
    #         x[idx] = x[idx] - width         # move it to the left
    #         y1val = y1val + [lind_fr[state][mode][idx]]
    #         y2val = y2val + [ref_fr[state][mode][idx]]
    # for idx, val in enumerate(bounds_fr[state][mode][1]):
    #     if val != 0:
    #         z = z + [x[idx] + width / 2]  # place Lindhal on the left
    #         zvalu = zvalu + [bounds_fr[state][mode][1][idx]]
    #         zvall = zvall + [bounds_fr[state][mode][0][idx]]
    if target_fr is not None:
        for idx, val in enumerate(target_fr):
            if val != 0:
                t = t + [x[idx] + width]  # place it the right
                tval = tval + [np.round(val, 2)]

    ax.set_xticks(np.array(x) + width/2)
    ax.set_xticklabels(name_list)
    ax.set_ylim(0, max(fr_list + y1val + y2val + zvalu + tval) * 1.4)

    width = width * 0.9
    if dopa_depl:
        bars1 = ax.bar(x, fr_list, width, label='Simulation') # , color='tab:red')
    else:
        bars1_dummy = ax.bar(x[0], 0., width, color='tab:grey', label='Simulation')
        bars11 = ax.bar(x[:4], fr_list[3:], width, color='#70abaf') # , label='Simulation (cereb)')
        bars12 = ax.bar(x[4:], fr_list[:3], width, color='#99e1d9') # , label='Simulation (BGs)')
    ax.bar_label(bars11, rotation=60)
    ax.bar_label(bars12, rotation=60)
    if mode == 'slow' or mode == 'active' or mode == 'invitro':
        bars2 = ax.bar(y1, y1val, width, label='Lindh. ref.', color='tab:green')
        bars3 = ax.bar(y2, y2val, width, label='Exper. ref.', color='tab:purple')
        ax.bar_label(bars2, rotation=60)
        ax.bar_label(bars3, rotation=60)
    if mode == 'slow':
        bars4 = ax.bar(z, zvalu, width, label='Higher bound', color='tab:orange')
        bars5 = ax.bar(z, zvall, width, label='Lower bound', color='tab:red')
        ax.bar(x, fr_list, width, color='tab:red')
        # ax.bar_label(bars4)
        # ax.bar_label(bars5)
    if target_fr is not None:
        bars6_dummy = ax.bar(t[0], 0., width, color='tab:grey', hatch='///', alpha=0.8, label='Reference models')
        bars61 = ax.bar(t[:4], tval[3:], width, color='#70abaf', hatch='///', alpha=0.8) # , label='Healthy target values')
        bars62 = ax.bar(t[4:], tval[:3], width, color='#99e1d9', hatch='///') # , label='Healthy target values')
        ax.bar_label(bars61, rotation=60)
        ax.bar_label(bars62, rotation=60)

    if target_fr is not None:
        ax.legend(loc='upper left')
    else:
        ax.legend()     # loc='upper left', bbox_to_anchor=(0.14, 1.0))
    ax.set_ylabel('Firing rate [Hz]')
    ax.set_title(f'Average firing rate') #  in mode: {mode}')

    ax.tick_params(bottom=False)

    # # CV
    # lind_CV = {'control': {'slow': [0, 0, 0, 0, 0, 0, 0.35, 0.45, 0],
    #                        'active': [0, 0, 0, 0, 0, 0, 0.4, 0.4, 0],
    #                        'cereb': [0] * 5,
    #                        'complete': [0] * 3 + [0, 0, 0.4, 0.4, 0],
    #                        'invitro': [0] * 8},
    #            'lesion': {'slow': [0, 0, 0, 0, 0.85, 0.75, 0, 0.45, 0],
    #                       'active': [0, 0, 0, 0, 0.1, 0.8, 0, 0.3, 0],
    #                       'cereb': [0] * 5,
    #                        'complete': [0] * 3 + [0.1, 0.8, 0, 0.3, 0],
    #                       'invitro': [0] * 8}}
    # ref_CV = {'control': {'slow': [0., 0., 0., 0., 0., 0., 0.49, 1.75, 0.],
    #                       'active': [0., 0., 0., 0., 0., 0., 0.43, 0.85, 0.],
    #                       'cereb': [0] * 5,
    #                        'complete': [0.] * 3 + [0., 0., 0.43, 0.85, 0.],
    #                       'invitro': [0] * 8},
    #           'lesion': {'slow': [0., 0., 0., 0., 1.7, 1.4, 0, 2.0, 0.],
    #                      'active': [0., 0., 0., 0., 0.6, 0.75, 0, 0.6, 0.],
    #                      'cereb': [0] * 5,
    #                      'complete': [0.] * 3 + [0.6, 0.75, 0, 0.6, 0.],
    #                      'invitro': [0] * 8}}
    #
    # ax = axes[1]
    # x = list(range(len(fr_list)))
    # ax.set_xticks(x)
    # y1 = y2 = []  # will place in y1, y2 Lindhal and exp refs. In z boundaries
    # y1val = y2val = []  # save elements before plotting
    # for idx, val in enumerate(lind_CV[state][mode]):
    #     if val != 0:
    #         y1 = y1 + [x[idx]]  # place Lindhal in the centre
    #         y2 = y2 + [x[idx] + width]  # place Mallet on the right
    #         x[idx] = x[idx] - width  # move it to the left
    #         y1val = y1val + [lind_CV[state][mode][idx]]
    #         y2val = y2val + [ref_CV[state][mode][idx]]
    #
    # ax.set_xticklabels(name_list)
    # ax.set_ylim(0, max(CV_list + y1val + y2val) * 1.4)
    #
    # bars1 = ax.bar(x, CV_list, width, label='Simulation')
    # bars2 = ax.bar(y1, y1val, width, label='Lindh. ref.', color='tab:green')
    # bars3 = ax.bar(y2, y2val, width, label='Exper. ref.', color='tab:purple')
    # ax.bar_label(bars1, rotation=60)
    # ax.bar_label(bars2, rotation=60)
    # ax.bar_label(bars3, rotation=60)
    #
    # ax.set_ylabel('CV [ ]')
    # ax.set_title(f'Coefficient of Variation (between all ISI) in mode: {mode}')

    fig.tight_layout()

    return fig, axes


# def plot_mass_frs(fr_array, start_stop_times, legend_labels, u_array=None, xlim=None, ylim=None, ext_ax=None, title=None):
#     """ Plot the equivalent firing rate of the mass models """
#     fig_width = 8.0
#     plot_height = 4.0
#     if ext_ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))
#     else:
#         ax = ext_ax
#
#     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
#     ax.set_prop_cycle(color=colors)
#
#     t_array = np.linspace(start_stop_times[0], start_stop_times[1], fr_array.shape[0])
#     ax.grid(linestyle='-.')
#     ax.plot(t_array, fr_array)
#     if u_array is not None:
#         ax.plot(t_array, u_array)
#     scale_xy_axes(ax, xlim, ylim)
#     ax.set_xlabel('Time [ms]')
#     ax.set_ylabel(f'Activity [sp/(neu*s)]')
#
#     if title is not None:
#         ax.set_title(f'{title}')
#
#     # print(f'Final mass f.r. value = {fr_array[-1, :]}')
#     if ext_ax is None:
#         ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
#         fig.tight_layout()
#         return fig, ax
#     else:
#         ax.set_title(f'Average activity with dopa. depl. = -0.4')
#         # ax.legend(legend_labels, fontsize=16)
#         return ax


def plot_fourier_transform(fr_array, T_sample, legend_labels, mean=None, sd=None, t_start=0.):
    """ Plot the discrete fourier transform of the mass models """
    fig_width = 9.
    plot_height = 3.5
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, plot_height))

    ax = axes[0]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    T = T_sample / 10.
    y = fr_array[int(t_start / T):, :]  # calculate tf after the t_start

    T = T_sample  # resample to 1 ms
    y = y[::10]  # select one time sample every 10
    N = y.shape[0]

    yf = fft(y, axis=0)
    xf = fftfreq(N, d=T/1000)[:N // 2]     # take just pos freq
    y_plot = 2.0 / N * np.abs(yf[0:N // 2])

    freq_p = np.linspace(-10, 10, 21, endpoint=True)
    kernel = utils.gaussian(freq_p, 0., 2.)
    for i in range(yf.shape[1]):
        y_plot[:, i] = np.convolve(y_plot[:, i], kernel, 'same')

    fourier_idx = utils.calculate_fourier_idx(xf, [1, 60])
    # print(f'In plot: considering frequencies in the range {[xf[fourier_idx[0]], xf[fourier_idx[1]-1]]}')

    fm_list = []
    ax.set_prop_cycle(color=colors)
    peak_width_limits = [[2, 8], [2, 8], [2, 12]]
    for idx in range(len(y_plot[0, :])):
        fm = FOOOF(peak_width_limits=peak_width_limits[idx])  # aperiodic_mode='knee')
        fm.fit(xf[fourier_idx[0]:fourier_idx[1]], y_plot[:, idx][fourier_idx[0]:fourier_idx[1]])
        # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(y_val[:, idx]), label=legend_labels[idx], alpha=0.5)
        # diff = np.log10(y_plot[:, idx][fourier_idx[0]:fourier_idx[1]]) - np.log10(
        #     1 / xf[fourier_idx[0]:fourier_idx[1]] ** fm.aperiodic_params_[1]) - \
        #        fm.aperiodic_params_[0]
        diff = fm.fooofed_spectrum_ - np.log10(
            1 / xf[fourier_idx[0]:fourier_idx[1]] ** fm.aperiodic_params_[1]) - \
               fm.aperiodic_params_[0]
        ax.plot(xf[fourier_idx[0]:fourier_idx[1]], diff, label=legend_labels[idx])
        fm_list += [fm]
        # fm.report(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx], [1, 60])

    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_plot[0, :])):
    #     ax.plot(xf[0], np.log10(y_plot[:, idx][0]), alpha=1, label=legend_labels[idx])
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_plot[0, :])):
    #     ax.plot(xf[fourier_idx[0]:fourier_idx[1]], np.log10(y_plot[:, idx][fourier_idx[0]:fourier_idx[1]]), alpha=.5) # , label=legend_labels[idx])
    #
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_plot[0, :])):
    #     ax.plot(xf[fourier_idx[0]:fourier_idx[1]], fm_list[idx].fooofed_spectrum_, '-.') # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.plot(xf[fourier_idx[0]:fourier_idx[1]], np.log10(1/xf[fourier_idx[0]:fourier_idx[1]]**fm_list[0].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':', c='tab:grey', label='aperiodic') # , label=f'{legend_labels[idx]} aperiodic')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_plot[0, :])):
    #     ax.plot(xf[fourier_idx[0]:fourier_idx[1]], np.log10(1/xf[fourier_idx[0]:fourier_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':') # , label=f'{legend_labels[idx]} aperiodic')

    if mean is not None:
        ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')

    ax.grid(linestyle='-.')
    scale_xy_axes(ax, ylim=[0., 0.58])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(f'Log Power [Activity^2]')

    ax.set_title('Fourier transform')
    ax.legend(legend_labels, fontsize=14) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    ax = axes[1]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    T = T_sample / 10.
    y = fr_array[int(t_start / T):, :]  # calculate tf after the t_start

    T = T_sample  # resample to 1 ms
    y = y[::10]  # select one time sample every 10

    fs = 1000. / T  # [Hz], sampling time
    w = 15.  # [adim], "omega0", in the definition of Morlet wavelet: pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
    freq = np.linspace(1, fs / 2, 2 * int(fs / 2 - 1) + 1)  # frequency range, until half fs
    widths = w * fs / (
                2 * freq * np.pi)  # [adim] reduce time widths for higher frequencies. Widhts / sample_freq = time

    y_plot = np.zeros((len(freq), y.shape[1]))
    for idx in range(y.shape[1]):
        cwtm = signal.cwt(y[:, idx], signal.morlet2, widths, w=w)
        y_plot[:, idx] = np.abs(cwtm).sum(axis=1)

    wavelet_idx = utils.calculate_fourier_idx(freq, [1, 60])
    # print(f'In plot: considering frequencies in the range {[freq[wavelet_idx[0]], freq[wavelet_idx[1]-1]]}')

    y_val = y_plot[wavelet_idx[0]:wavelet_idx[1], :]

    fm_list = []
    ax.set_prop_cycle(color=colors)
    peak_width_limits = [[2, 8], [2, 8], [3, 20]]

    if mean is not None:
        ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')

    diff = np.zeros(y_val.shape)
    for idx in range(len(y_val[0, :])):
        fm = FOOOF(peak_width_limits=peak_width_limits[idx])  # aperiodic_mode='knee')
        fm.fit(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx])
        # diff[:, idx] = np.log10(y_val[:, idx]) - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        diff[:, idx] = fm.fooofed_spectrum_ - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], diff[:, idx], label=legend_labels[idx])
        fm_list += [fm]
        # fm.report(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx], [1, 60])

    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(y_val[:, idx]), alpha=.5, label=legend_labels[idx])
    #
    # # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_, '-.', c='tab:grey',
    # #         label='fooof interp')  # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_,
    #             '-.')  # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]],
    #         np.log10(1 / freq[wavelet_idx[0]:wavelet_idx[1]] ** fm_list[idx].aperiodic_params_[1]) +
    #         fm_list[idx].aperiodic_params_[0], ':', c='tab:grey',
    #         label='aperiodic')  # , label=f'{legend_labels[idx]} aperiodic')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]],
    #             np.log10(1 / freq[wavelet_idx[0]:wavelet_idx[1]] ** fm_list[idx].aperiodic_params_[1]) +
    #             fm_list[idx].aperiodic_params_[0], ':')  # , label=f'{legend_labels[idx]} aperiodic')

    ax.grid(linestyle='-.')
    scale_xy_axes(ax, ylim=[0., 0.58])
    ax.set_xlabel('Frequency [Hz]')
    # ax.set_ylabel(f'Log Power [Activity^2]')

    # ax.legend(fontsize=16)  # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    ax.set_title('Wavelet transform')
    ax.legend(fontsize=14) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    fig.tight_layout()




    return fig, ax


def plot_wavelet_transform(mass_models_sol, T_sample, legend_labels, mean=None, sd=None, t_start=0., y_range=None, dopa_depl=None):
    """ Plot the discrete fourier transform of the mass models """
    fig_width = 6.
    plot_height = 4.
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    if not isinstance(mass_models_sol, list):
        mass_models_sol = [mass_models_sol]

    T = mass_models_sol[0]["mass_frs_times"][1] - mass_models_sol[0]["mass_frs_times"][0]
    sel_mass_times1 = mass_models_sol[0]["mass_frs_times"] > t_start
    sel_mass_times2 = mass_models_sol[0]["mass_frs_times"] % T_sample == 0
    y_list = [mms["mass_frs"][np.logical_and(sel_mass_times1, sel_mass_times2), :] for mms in mass_models_sol]  # calculate tf after the t_start

    # T = T_sample    # resample to 1 ms
    # y = y[::10]     # select one time sample every 10

    fs = 1000./T_sample    # [Hz], sampling time
    w = 15.         # [adim], "omega0", in the definition of Morlet wavelet: pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
    freq = np.linspace(1, fs / 2, 2*int(fs/2-1)+1)      # frequency range, until half fs
    widths = w * fs / (2 * freq * np.pi)    # [adim] reduce time widths for higher frequencies. Widhts / sample_freq = time

    y_plot_list = [np.zeros((len(freq), y.shape[1])) for y in y_list]
    for y, y_plot in zip(y_list, y_plot_list):
        for idx in range(y.shape[1]):
            cwtm = signal.cwt(y[:, idx], signal.morlet2, widths, w=w)
            y_plot[:, idx] = np.abs(cwtm).sum(axis=1)

    wavelet_idx = utils.calculate_fourier_idx(freq, [1, 60])
    # print(f'In plot: considering frequencies in the range {[freq[wavelet_idx[0]], freq[wavelet_idx[1]-1]]}')

    y_val_list = [y_plot[wavelet_idx[0]:wavelet_idx[1], :] for y_plot in y_plot_list]

    fm_list = []
    ax.set_prop_cycle(color=colors)
    peak_width_limits = [[2, 8], [2, 8], [3, 20]]

    if mean is not None:
        ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')

    diff_list = [np.zeros(y_val.shape) for y_val in y_val_list]
    fm_list_list = []
    for y_val, diff in zip(y_val_list, diff_list):
        fm_list = []
        for idx in range(len(y_val[0, :])):
            fm = FOOOF(peak_width_limits=peak_width_limits[idx])  # aperiodic_mode='knee')
            fm.fit(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx])
            fm_list += [fm]
            # diff[:, idx] = np.log10(y_val[:, idx]) - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
            diff[:, idx] = fm.fooofed_spectrum_ - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        fm_list_list += [fm_list]

    diff_array = np.array(diff_list)
    diff_mean = diff_array.mean(axis=0)
    diff_std = diff_array.std(axis=0)
    for idx in range(diff_mean.shape[1]):
        ax.fill_between(freq[wavelet_idx[0]:wavelet_idx[1]], diff_mean[:, idx] - diff_std[:, idx], diff_mean[:, idx] + diff_std[:, idx], alpha=0.5)
        ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], diff_mean[:, idx], label=legend_labels[idx])

    # fm.report(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx], [1, 60])

    # if mean is not None:
    #     ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val_list[0][0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(np.array(y_val_list).mean(axis=0)[:, idx]), alpha=.5, label=legend_labels[idx])
    #
    #     fooofed = [fm_list_list[k][idx].fooofed_spectrum_ for k in range(len(mass_models_sol))]
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.array(fooofed).mean(axis=0), '-.', c='tab:grey', label='fooof interp') # , label=f'{legend_labels[idx]} fooof')

    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_, '-.') # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':', c='tab:grey', label='aperiodic') # , label=f'{legend_labels[idx]} aperiodic')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':') # , label=f'{legend_labels[idx]} aperiodic')

    ax.grid(linestyle='-.')
    scale_xy_axes(ax, ylim=[-0.0, 1.1])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(f'Log Power [Activity^2]')

    ax.legend(fontsize=12) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    ax.set_title(f'Wavelet transform with dopa. depl. = {dopa_depl}')

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    fig.tight_layout()


    return fig, ax, diff_mean


def plot_wavelet_transform_and_mass(mass_models_sol, T_sample, legend_labels, mean=None, sd=None, t_start=0., t_end=0., y_range=None):
    """ Plot the discrete fourier transform of the mass models """
    fig_width = 10.
    plot_height = 3.
    # fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, plot_height))

    ax = axes[0]
    plot_mass_frs(mass_models_sol, legend_labels, ext_ax=ax)

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    ax = axes[1]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    T = T_sample / 10.
    sel_mass_times = mass_models_sol["mass_frs_times"] > t_start
    y = mass_models_sol["mass_frs"][sel_mass_times, :]  # calculate tf after the t_start

    T = T_sample    # resample to 1 ms
    y = y[::10]     # select one time sample every 10

    fs = 1000./T    # [Hz], sampling time
    w = 15.         # [adim], "omega0", in the definition of Morlet wavelet: pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
    freq = np.linspace(1, fs / 2, 2*int(fs/2-1)+1)      # frequency range, until half fs
    widths = w * fs / (2 * freq * np.pi)    # [adim] reduce time widths for higher frequencies. Widhts / sample_freq = time

    y_plot = np.zeros((len(freq), y.shape[1]))
    for idx in range(y.shape[1]):
        cwtm = signal.cwt(y[:, idx], signal.morlet2, widths, w=w)
        y_plot[:, idx] = np.abs(cwtm).sum(axis=1)

    wavelet_idx = utils.calculate_fourier_idx(freq, [1, 60])
    # print(f'In plot: considering frequencies in the range {[freq[wavelet_idx[0]], freq[wavelet_idx[1]-1]]}')

    y_val = y_plot[wavelet_idx[0]:wavelet_idx[1], :]

    fm_list = []
    ax.set_prop_cycle(color=colors)
    peak_width_limits = [[2, 8], [2, 8], [3, 20]]

    if mean is not None:
        ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')

    diff = np.zeros(y_val.shape)
    for idx in range(len(y_val[0, :])):
        fm = FOOOF(peak_width_limits=peak_width_limits[idx])  # aperiodic_mode='knee')
        fm.fit(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx])
        # diff[:, idx] = np.log10(y_val[:, idx]) - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        diff[:, idx] = fm.fooofed_spectrum_ - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], diff[:, idx], label=legend_labels[idx])
        fm_list += [fm]
        # fm.report(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx], [1, 60])

    # if mean is not None:
    #     ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(y_val[:, idx]), alpha=.5, label=legend_labels[idx])
    #
    # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_, '-.', c='tab:grey', label='fooof interp') # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_, '-.') # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':', c='tab:grey', label='aperiodic') # , label=f'{legend_labels[idx]} aperiodic')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':') # , label=f'{legend_labels[idx]} aperiodic')

    ax.grid(linestyle='-.')
    # scale_xy_axes(ax, ylim=[-0.2, 0.7])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(f'Log Power [Activity^2]')

    ax.legend(fontsize=12) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    ax.set_title('b) Wavelet transform with dopa. depl. = -0.4')

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    fig.tight_layout()


    return fig, ax, diff


def combine_axes_in_figure(rasters_list, mass_models_sol, legend_labels=None, clms=2, t_start=0.,
                           ylim=None):
    """ Plot rasters and the equivalent firing rate of the mass models """
    n_plots = len(rasters_list)
    rows = int(np.ceil(n_plots / clms)) + 1  # n° rasters + 1 fr plot
    fig_width = 8.0
    plot_height = 2.8
    fig, axs = plt.subplots(rows, clms, figsize=(fig_width * clms, plot_height * rows))

    keywords = {'x_data': 'times', 'y_data': 'neurons_idx', 'name': 'compartment_name'}
    multiple_plots(rasters_list, raster_plot, keywords, clms=clms, ext_axs=axs[:-1],
                   t_start=t_start)

    plot_mass_frs(mass_models_sol, legend_labels, xlim=None, ylim=ylim, ext_ax=axs[-1])
    for item in [axs[-1].title, axs[-1].xaxis.label, axs[-1].yaxis.label]:
        item.set_fontsize(3. * 5)

    fig.tight_layout()

    # rows = sum([axes.shape[0] if isinstance(axes, np.ndarray) else 1 for axes in axes_list])
    # clms = ceil(sum([axes.size if isinstance(axes, np.ndarray) else 1 for axes in axes_list]) / rows)
    # fig, new_axes = plt.subplots(rows, clms, figsize=(fig_width * clms, plot_height * rows))
    #
    # counter = 0
    # for axes in axes_list:
    #     if isinstance(axes, np.ndarray):
    #         act_rows = axes.shape[0]
    #     else:
    #         act_rows = 1
    #     if clms == 1:
    #         new_axes[:counter + act_rows] = axes
    #         axes[0].plot()
    #         print(axes[0].lines[0])
    #         print(new_axes[0].lines[0])
    #     else:
    #         new_axes[:counter + act_rows, :clms] = pickle.loads(pickle.dumps(axes))
    #     counter = counter + act_rows

    # new_axes.plot()
    return fig, axs


def plot_mass_frs(mass_models_sol, legend_labels, u_array=False, xlim=None, ylim=None, ext_ax=None, title=None):
    """ Plot the equivalent firing rate of the mass models """
    fig_width = 8.0
    plot_height = 4.0
    if ext_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))
    else:
        ax = ext_ax

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    # fr_array = fr_array[15000:]
    ax.grid(linestyle='-.')
    ax.plot(mass_models_sol['mass_frs_times'], mass_models_sol['mass_frs'])
    if u_array:
        ax.plot(mass_models_sol['mass_frs_times'], mass_models_sol['in_frs'])
    scale_xy_axes(ax, xlim, ylim)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(f'Activity [sp/(neu*s)]')

    if title is not None:
        ax.set_title(f'{title}')

    # print(f'Final mass f.r. value = {fr_array[-1, :]}')
    if ext_ax is None:
        ax.legend(legend_labels, fontsize=16) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
        fig.tight_layout()
        return fig, ax
    else:
        ax.set_title(f'a) Average activity with dopa. depl. = -0.4')
        # ax.legend(legend_labels, fontsize=16)
        return ax


def plot_weights(weights, sim_time, settling_time, trials):
    fig, ax = plt.subplots()

    print(weights)

    for w, c in zip(weights, ['tab:blue', 'tab:orange']):
        ax.step(w['times'], w['weights'], label=w['sender_receiver'], c=c)

    for i in range(trials):
        t_low = i * (sim_time + settling_time)
        t_high = i * (sim_time + settling_time) + settling_time
        ax.axvspan(t_low, t_high, alpha=0.2, color='tab:red')

    ax.set_title('Weights temporal evolution')
    ax.legend()

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Weight')

    ax.grid()
    return fig, ax


def robot_plot(y, sim_time, settling_time, sim_period, trials, y_des=None, title=None, legend=None, ax_labels=None, x_range=None):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    if x_range is not None:
        fig, ax = plt.subplots(figsize=(3.5, 5.5))
    # t_grid = np.linspace(0, (sim_time+settling_time)*trials, (int((sim_time+settling_time)/sim_period) + 1)*trials)
    t_grid = np.linspace(0, (sim_time+settling_time)*trials, (int((sim_time+settling_time)/sim_period))*trials, endpoint=False)

    if title == 'Joint position':
        ax.plot(t_grid, y[:, 0], c='tab:green')
        ax.plot(t_grid, np.tile(y_des[0, :], trials).T, c='tab:grey', linewidth=0.5)
        ax.plot(t_grid, y[:, 0] - np.tile(y_des[0, :], trials).T, c='tab:red')

        if y.shape[1] > 1:
            ax.plot(t_grid, y[:, 1], c='tab:green', linestyle='--')
            ax.plot(t_grid, np.tile(y_des[1, :], trials).T, c='tab:grey', linestyle='--', linewidth=0.5)
            ax.plot(t_grid, np.tile(y_des[1, :], trials).T, c='tab:red', linestyle='--', linewidth=0.5)

        # t_grid_onetrial = np.linspace(0, (sim_time+settling_time), int((sim_time+settling_time)/sim_period) + 1)
        # t_grid = t_grid_onetrial
        #
        # y_new = np.zeros(len(t_grid_onetrial))
        # y_new[-int(sim_time/sim_period)-1:] = y_des[:int(sim_time/sim_period)+1]
        # # y_new = np.concatenate((np.zeros((int(settling_time/sim_period, size))), y[:len(t_grid)-int(settling_time/sim_period)]), axis = 0)
        # for i in range(1, trials):
        #     t_grid = np.concatenate((t_grid, t_grid_onetrial+t_grid[-1]), axis=0)
        #     y_new = np.concatenate((y_new, np.zeros(len(t_grid_onetrial))))
        #     y_new[-int(sim_time / sim_period)-1:] = y_des[(int(sim_time / sim_period)+1)*i:(int(sim_time / sim_period)+1) * (i+1)]

        # scale_xy_axes(ax, ylim=[-10., 10.])

    elif title == 'Resulting torque':
        conv_kern = np.ones(int(100/sim_period))
        conv_kern = conv_kern/conv_kern.sum()
        delta_tau = y[:, 1::2] - y[:, ::2]
        # tau_res = np.convolve(delta_tau, conv_kern, mode='same')
        ax.plot(t_grid, delta_tau[:, 0], color='tab:purple')
        if delta_tau.shape[1] > 1:
            ax.plot(t_grid, delta_tau[:, 1], color='tab:purple', linestyle='--')
        # ax.plot(t_grid, y, alpha=0.2)
        # ax.fill_between(t_grid, tau_res, color='purple', alpha=0.4)

    elif title == 'Cortex average activity':
        t_grid = np.linspace(0, (sim_time + settling_time) * trials,
                             (int((sim_time + settling_time) / sim_period) * 10) * trials + 1)
        ax.plot(t_grid, y, color='tab:purple')

    else:
        # if title == 'DCN torque':
            # scale_xy_axes(ax, ylim=[0, 5])

        conv_kern = np.ones(int(100/sim_period))
        conv_kern = conv_kern/conv_kern.sum()
        ax.plot(t_grid, np.convolve(y[:, 0], conv_kern, mode='same'), color='tab:orange')
        ax.plot(t_grid, np.convolve(y[:, 1], conv_kern, mode='same'), color='tab:blue')
        if y.shape[1] > 2:
            ax.plot(t_grid, np.convolve(y[:, 2], conv_kern, mode='same'), color='tab:orange', linestyle='--')
            ax.plot(t_grid, np.convolve(y[:, 3], conv_kern, mode='same'), color='tab:blue', linestyle='--')
        # ax.plot(t_grid, y, alpha=0.2)

    for i in range(trials):
        t_low = i * (sim_time + settling_time)
        t_high = i * (sim_time + settling_time) + settling_time
        ax.axvspan(t_low, t_high, alpha=0.2, color='tab:red')

    # scale_xy_axes(ax, ylim=[-6., 6.])

    if title is not None:
        ax.set_title(title)
    if legend is not None:
        ax.legend(legend)
    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
    ax.grid()

    if x_range is not None:
        scale_xy_axes(ax, xlim=x_range, ylim=[0, 2.8])
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(12)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)

    else:
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(12)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)
        # fig.tight_layout()

    return fig, ax


def rms_plot(y, y_des, sim_time, settling_time, sim_period, trials, title=None, legend=None, ax_labels=None, x_range=None):
    fig, ax = plt.subplots()

    rms_list = []
    t_grid = np.linspace(settling_time + 0.5*sim_time, (settling_time+sim_time)*trials - 0.5*sim_time, trials, endpoint=True)

    y_des = np.tile(y_des, trials).T

    for i in range(0, trials):
        y_t = y[(int((sim_time) / sim_period)) * i:(int((sim_time) / sim_period)) * (i+1)]
        y_d = y_des[(int(sim_time / sim_period)) * i:(int(sim_time / sim_period)) * (i + 1)]
        rms = np.sqrt(np.sum((y_t-y_d) ** 2))
        rms_list = rms_list + [rms]
        # simple_plot(range(len(y_t)), np.array([y_t, y_d]).T)

    ax.plot(t_grid, rms_list, c='tab:red', linestyle='-.')
    ax.scatter(t_grid, rms_list, c='tab:red')

    if title is not None:
        ax.set_title(title)
    if legend is not None:
        ax.legend(legend)
    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
    ax.grid()

    scale_xy_axes(ax, ylim=[0, max(rms_list)])

    rms_array = np.array(rms_list)
    print(f'mean = {rms_array.mean()}')

    kernel = np.linspace(6, trials, trials - 6 + 1)
    kernel = (kernel - 4.) / 3. / len(kernel)

    if len(rms_list) >= 6:
        rms_error = sum(rms_list[5:] * kernel)
        last_5_rms = np.array(rms_list[-5:])
        rms_std = last_5_rms.std()

    else:
        rms_error = 0.
        rms_std = 0.

    return fig, ax, rms_error, rms_std


def combined_robot_plot(y1, y2, y3, sim_time, settling_time, sim_period, trials, y_des=None, title=None, legend=None, ax_labels=None, x_range=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    ax = axes[0]
    t_grid = np.linspace(0, (sim_time + settling_time) * trials,
                         (int((sim_time + settling_time) / sim_period) + 1) * trials)
    ax.plot(t_grid, y1, c='tab:green')

    t_grid_onetrial = np.linspace(0, (sim_time + settling_time), int((sim_time + settling_time) / sim_period) + 1)
    t_grid = t_grid_onetrial

    y_new = np.zeros(len(t_grid_onetrial))
    y_new[-int(sim_time / sim_period) - 1:] = y_des[:int(sim_time / sim_period) + 1]
    # y_new = np.concatenate((np.zeros((int(settling_time/sim_period, size))), y[:len(t_grid)-int(settling_time/sim_period)]), axis = 0)
    for i in range(1, trials):
        t_grid = np.concatenate((t_grid, t_grid_onetrial + t_grid[-1]), axis=0)
        y_new = np.concatenate((y_new, np.zeros(len(t_grid_onetrial))))
        y_new[-int(sim_time / sim_period) - 1:] = y_des[(int(sim_time / sim_period) + 1) * i:(int(sim_time / sim_period) + 1) * (i + 1)]

        ax.plot(t_grid, y_new, c='tab:green', linestyle='-.', linewidth=0.5)
    ax.set_title('Joint position')
    ax.set_ylabel('Angle [rad]')
    ax.grid()

    for i in range(trials):
        t_low = i * (sim_time + settling_time)
        t_high = i * (sim_time + settling_time) + settling_time
        ax.axvspan(t_low, t_high, alpha=0.2, color='tab:red')

    if x_range is not None:
        scale_xy_axes(ax, xlim=x_range)
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(12)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)

    ax = axes[1]
    conv_kern = np.ones(int(100 / sim_period))
    conv_kern = conv_kern / conv_kern.sum()
    ax.plot(t_grid, np.convolve(y2[:, 0], conv_kern, mode='same'), color='tab:blue')
    ax.plot(t_grid, np.convolve(y2[:, 1], conv_kern, mode='same'), color='tab:orange')
    ax.plot(t_grid, y2, alpha=0.2)

    ax.set_title('IO average frequency')
    ax.set_ylabel('f.r. [sp/s]')
    ax.grid()

    for i in range(trials):
        t_low = i * (sim_time + settling_time)
        t_high = i * (sim_time + settling_time) + settling_time
        ax.axvspan(t_low, t_high, alpha=0.2, color='tab:red')

    if x_range is not None:
        scale_xy_axes(ax, xlim=x_range)
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(12)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)

    ax = axes[2]
    t_grid = np.linspace(0, (sim_time + settling_time) * trials,
                         (int((sim_time + settling_time) / sim_period) * 10) * trials + 1)
    ax.plot(t_grid, y3, color='tab:purple')

    for i in range(trials):
        t_low = i * (sim_time + settling_time)
        t_high = i * (sim_time + settling_time) + settling_time
        ax.axvspan(t_low, t_high, alpha=0.2, color='tab:red')

    ax.set_title('CTX activity')
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('Average acitvity [sp/(s*neuron)]')
    ax.grid()

    if x_range is not None:
        scale_xy_axes(ax, xlim=x_range)
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(12)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)

    fig.tight_layout()

    return fig, ax


def plot_pos_vel(tt, pos, vel, title=None, legend=None, ax_labels=None):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.plot(tt, pos, c='tab:green')
    ax.plot(tt, vel, c='tab:blue')

    if title is not None:
        ax.set_title(title)
    # if legend is not None:
        # ax.legend(legend[0])
        # ax2.legend(legend[1])
    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0])
        ax2.set_ylabel('Position [rad]', color='tab:green')
        ax.set_ylabel('Velocity [rad/s]', color='tab:blue')

    ax.grid()
    fig.show()

def simple_plot(x, y, title=None, legend=None, ax_labels=None):
    fig, ax = plt.subplots()
    if title == 'Joint position':
        ax.plot(x, y[:, 0], c='tab:green')
        ax.plot(x, y[:, 1], c='tab:green', linestyle='-.', linewidth=0.5)
    elif title == 'Position and velocity':
        ax.plot(x, y[:, 0], c='tab:red')
        ax.plot(x, y[:, 1], c='tab:green')
        ax.plot(x, y[:, 2], c='tab:red', linestyle='-.', linewidth=0.5)
        ax.plot(x, y[:, 3], c='tab:green', linestyle='-.', linewidth=0.5)
    else:
        ax.plot(x, y)
    if title is not None:
        ax.set_title(title)
    if legend is not None:
        ax.legend(legend)
    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
    ax.grid()
    fig.show()



