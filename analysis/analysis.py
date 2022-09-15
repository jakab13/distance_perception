import json
import numpy
from mne.stats import spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency, make_1020_channel_selections
from scipy.stats import ttest_ind
import numpy as np
import pathlib
import mne
import os
import matplotlib.pyplot as plt
import slab
from matplotlib.gridspec import GridSpec
from analysis.joint_decorrelation import compute_transformation
from analysis.utils import apply_transform
# %matplotlib qt


plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])


def ignore_conds(d, *keys):
    return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))


def ci_calc(d):
    return mne.stats.bootstrap_confidence_interval(d, ci=0.5, stat_fun="median")


if __name__ == "__main__":
    experiment = "pinknoise"  # either "pinknoise" or "vocal_effort"
    # get pilot folder directory.
    DIR = pathlib.Path(os.getcwd())
    fig_path = pathlib.Path(os.getcwd()) / "analysis" / "figures"
    sub_DIR = DIR / "analysis" / "data" / f"{experiment}"  # pilot_laughter or pilot_noise
    with open(DIR / "analysis" / "preproc_config.json") as file:
        cfg = json.load(file)
    # get subject ids
    # IDs of participants with 500ms stimulus lengths
    ids_500ms = ["cl3vh1", "e0q0kf", "zx4v7q", "l6e5vs", "atz2ps", "t895pw", "22ocyt", "c93bxb", "8adzch", "4q868v",
                 "03d3rc", "375j87", "gnnl7z", "gml110", "2ojf8e", "oucxs1", "4jyisj", "kptdo4", "oht0ey"]
    ids_300ms = ["g7rpgd", "u88rdo", "topo6t", "hufhxx", "e3km75", "1r3qdv", "k9p0u7", "5epwx8", "ihc3jn", "8jhsvx",
                 "omfdt1", "cs2i9p"]
    ids_vocalist_2 = ["cl3vh1", "zx4v7q", "atz2ps", "22ocyt", "8adzch", "03d3rc", "gnnl7z", "2ojf8e",
                      "kptdo4", "g7rpgd", "topo6t", "e3km75", "k9p0u7", "ihc3jn", "omfdt1"]
    ids_vocalist_11 = ["e0q0kf", "l6e5vs", "t895pw", "c93bxb", "4q868v", "375j87", "gml110", "oucxs1", "4jyisj",
                       "oht0ey", "u88rdo", "hufhxx", "1r3qdv", "5epwx8", "8jhsvx", "cs2i9p"]
    ids_500ms_vocalist_2 = list(set(ids_500ms) & set(ids_vocalist_2))
    ids_300ms_vocalist_2 = list(set(ids_300ms) & set(ids_vocalist_2))
    ids_500ms_vocalist_11 = list(set(ids_500ms) & set(ids_vocalist_11))
    ids_300ms_vocalist_11 = list(set(ids_300ms) & set(ids_vocalist_11))
    id_score_7 = ["e3km75"] # behaviour average_score = 7.0
    id_score_9 = ["kptdo4"]  # behaviour average_score = 8.75
    id_score_16 = ["zx4v7q"] # behaviour average_score = 16.0

    ids = list(name for name in os.listdir(sub_DIR)
               if os.path.isdir(os.path.join(sub_DIR, name)))
    # ids = ids_300ms
    # make dictionaries with empty event keys.
    # first copy config file to prevent changes.
    evokeds, evokeds_avrgd, evokeds_data = cfg["epochs"][f"event_id_{experiment}"].copy(
    ), cfg["epochs"][f"event_id_{experiment}"].copy(), cfg["epochs"][f"event_id_{experiment}"].copy()
    # epochs = cfg["epochs"][f"event_id_{experiment}"].copy()
    for key in cfg["epochs"][f"event_id_{experiment}"]:
        evokeds[key], evokeds_avrgd[key], evokeds_data[key] = list(), list(), list()
        # epochs[key] = list()
        # get evokeds for every condition and subject.

    evoked_jds = [[], [], [], [], []]
    evoked_jds_avrgd = [[], [], [], [], []]
    # peak_times = {}
    # peak_amplitudes = {}
    # gfps = {}
    # all_epochs = {}
    for id in ids:
        evokeds_folder = sub_DIR / id / "evokeds"
        epochs_folder = sub_DIR / id / "epochs"
        # evoked = mne.read_evokeds(
        #     evokeds_folder / pathlib.Path(id + '-ave.fif'))

        epochs = mne.read_epochs(epochs_folder / pathlib.Path(id + '-epo.fif'))
        epochs.apply_baseline(baseline=(-0.2, 0))
        epochs.shift_time(-0.1, relative=True)

        to_jd1, from_jd1, to_jd2, from_jd2 = \
            compute_transformation(epochs, 1, 5, 1)
        Y = apply_transform(epochs.get_data(), to_jd1)
        idx1 = np.where(epochs.events[:, 2] == 1)[0]
        idx2 = np.where(epochs.events[:, 2] == 2)[0]
        idx3 = np.where(epochs.events[:, 2] == 3)[0]
        idx4 = np.where(epochs.events[:, 2] == 4)[0]
        idx5 = np.where(epochs.events[:, 2] == 5)[0]
        evoked_jd = [Y[idx1, 0, :].mean(axis=0),
                     Y[idx2, 0, :].mean(axis=0),
                     Y[idx3, 0, :].mean(axis=0),
                     Y[idx4, 0, :].mean(axis=0),
                     Y[idx5, 0, :].mean(axis=0)]
        evoked = [epochs[condition].average()
                   for condition in cfg["epochs"][f"event_id_{experiment}"].keys()]
        for condition in evoked:
            if condition.comment in evokeds:
                evokeds[condition.comment].append(condition.crop(-0.1, 0.5))
                if len(evokeds[condition.comment]) == len(ids):
                    evokeds_avrgd[condition.comment] = mne.grand_average(
                        evokeds[condition.comment])
                else:
                    continue

        for condition in [0, 1, 2, 3, 4]:
            evoked_jds[condition].append(evoked_jd[condition])
            if len(evoked_jds[condition]) == len(ids):
                evoked_jds_avrgd[condition] = evoked_jds[condition].mean(axis=0)
        # peak_times[id] = {}
        # peak_amplitudes[id] = {}
        # for condition in evoked:
        #     print(condition)
        #     evoked_picked = condition.copy()
        #     evoked_picked.pick_channels(["FCz"])
        #     peak_data = evoked_picked.get_peak(return_amplitude=True)
        #     peak_times[id][condition.comment] = peak_data[1]
        #     peak_amplitudes[id][condition.comment] = peak_data[2]

    FCz_peak_times_VE = [0.110, 0.226, 0.341]
    FCz_peak_times_PN = [0.102, 0.198, 0.355]
    gfp_peak_times = [0.214, 0.318]

    # show some topographic and line plots to get an impression of evoked responses.
    # plot control condition to have a look at simple auditory evoked signals without reverb info.
    # channels = ["FCz"]  # select electrode of interest.
    # evokeds_avrgd["deviant"].plot_joint()
    # evokeds_avrgd["distance/20"].plot_topomap(times=np.linspace(
    #     0.05, 0.6, 10), title="Topograhical responses control")
    # mne.viz.plot_evoked_topo(evokeds_avrgd["distance/20"])
    # Central electrodes show auditory evoked activity and differences between conditions.
    # Select Cz and look at conditions around 350 ms after stimuluis onset.
    mne.viz.plot_compare_evokeds(ignore_conds(
        evokeds_avrgd, "deviant", "button_press"),
        # cmap=plt.get_cmap('viridis'),
        # colors=[0, 0.2, 0.4, 0.6, 0.8],
        colors=["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"],
        title="Compare Evokeds",
        vlines=FCz_peak_times_PN,
        ylim=dict(eeg=[0, 2.7]),
        legend="upper right"
    )
    # evokeds_avrgd["vocal_effort/1"].plot_image()
    # combined_evokeds_avrgd = mne.combine_evoked(
    #     [evokeds_avrgd["vocal_effort/1"], evokeds_avrgd["vocal_effort/2"],
    #      evokeds_avrgd["vocal_effort/3"], evokeds_avrgd["vocal_effort/4"],
    #      evokeds_avrgd["vocal_effort/5"]], weights=[0.2, 0.2, 0.2, 0.2, 0.2])
    combined_evokeds_avrgd = mne.combine_evoked(
        [evokeds_avrgd[f"{experiment}/1"], evokeds_avrgd[f"{experiment}/2"],
         evokeds_avrgd[f"{experiment}/3"], evokeds_avrgd[f"{experiment}/4"],
         evokeds_avrgd[f"{experiment}/5"]], weights=[0.2, 0.2, 0.2, 0.2, 0.2])
    # combined_evokeds_avrgd.plot_topomap(times=numpy.linspace(0.15, 0.5, 20),
    #                                     title="Combined Evoked (long) (150-500ms)",
    #                                     vmin=-5, vmax=5)
    # time_ranges_of_interest = [0.108, 0.226, 0.341]


    topomap = combined_evokeds_avrgd.plot_topomap(times=FCz_peak_times_PN)
    joint_plot = combined_evokeds_avrgd.plot_joint(times=FCz_peak_times_PN)
    combined_evokeds_avrgd.plot(gfp="only", spatial_colors=True)
    # combined_evokeds_avrgd.plot(gfp=True, spatial_colors=True)

    mne.viz.plot_topomap(combined_evokeds_avrgd, time=FCz_peak_times_PN)
    plt.savefig(fig_path / "Topomaps.png")

    ######################################################################################################

    # Compare 500ms and 300ms stimuli

    ######################################################################################################

    evokeds_300, evokeds_avrgd_300, evokeds_data_300 = cfg["epochs"][f"event_id_{experiment}"].copy(
    ), cfg["epochs"][f"event_id_{experiment}"].copy(), cfg["epochs"][f"event_id_{experiment}"].copy()
    # epochs = cfg["epochs"][f"event_id_{experiment}"].copy()
    for key in cfg["epochs"][f"event_id_{experiment}"]:
        evokeds_300[key], evokeds_avrgd_300[key], evokeds_data_300[key] = list(), list(), list()

    evokeds_500, evokeds_avrgd_500, evokeds_data_500 = cfg["epochs"][f"event_id_{experiment}"].copy(
    ), cfg["epochs"][f"event_id_{experiment}"].copy(), cfg["epochs"][f"event_id_{experiment}"].copy()
    # epochs = cfg["epochs"][f"event_id_{experiment}"].copy()
    for key in cfg["epochs"][f"event_id_{experiment}"]:
        evokeds_500[key], evokeds_avrgd_500[key], evokeds_data_500[key] = list(), list(), list()

    for id in ids_300ms:
        evokeds_folder = sub_DIR / id / "evokeds"
        epochs_folder = sub_DIR / id / "epochs"
        evoked = mne.read_evokeds(
            evokeds_folder / pathlib.Path(id + '-ave.fif'))
        # epochs = mne.read_epochs(epochs_folder / pathlib.Path(id + '-epo.fif'))
        # all_epochs[id] = epochs
        for condition in evoked:
            if condition.comment in evokeds_300:
                evokeds_300[condition.comment].append(condition.crop(-0.3, 0.7))
                if len(evokeds_300[condition.comment]) == len(ids):
                    evokeds_avrgd_300[condition.comment] = mne.grand_average(
                        evokeds_300[condition.comment])
                else:
                    continue
    for id in ids_500ms:
        evokeds_folder = sub_DIR / id / "evokeds"
        epochs_folder = sub_DIR / id / "epochs"
        evoked = mne.read_evokeds(
            evokeds_folder / pathlib.Path(id + '-ave.fif'))
        # epochs = mne.read_epochs(epochs_folder / pathlib.Path(id + '-epo.fif'))
        # all_epochs[id] = epochs
        for condition in evoked:
            if condition.comment in evokeds_500:
                evokeds_500[condition.comment].append(condition.crop(-0.3, 0.7))
                if len(evokeds_500[condition.comment]) == len(ids):
                    evokeds_avrgd_500[condition.comment] = mne.grand_average(
                        evokeds_500[condition.comment])
                else:
                    continue

    evokeds_diff = {}
    evokeds_diff["300"] = [evoked_obj for condition in evokeds_300 if condition not in ["button_press", "deviant"] for
                           evoked_obj in evokeds_300[condition]]
    evokeds_diff["500"] = [evoked_obj for condition in evokeds_500 if condition not in ["button_press", "deviant"] for
                           evoked_obj in evokeds_500[condition]]
    evokeds_diff_avrgd = mne.combine_evoked(
        [mne.grand_average(evokeds_diff['500']), mne.grand_average(evokeds_diff['300'])],
        weights=[1, -1]
        )
    contrast = "500ms-300ms"
    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    # ax2 = plt.subplot2grid((2, 1), (1, 0))
    # ax3 = plt.subplot2grid((3, 1), (2, 0))
    mne.viz.plot_compare_evokeds(evokeds_diff, title="ERP of 500ms and 300ms stimuli", axes=ax1)
    ax1.set_title("ERP (500ms - 300ms)")
    # mne.viz.plot_compare_evokeds(evokeds_diff, picks="FCz", title="ERP of 500ms and 300ms stimuli at FCz", axes=ax2)
    # ax2.set_title("ERP (500ms - 300ms) at FCz")
    mne.viz.plot_compare_evokeds(evokeds_diff_avrgd, axes=ax1)
    # ax2.set_title("Difference")
    ax1.set_ylim([-3.5, 3.5])
    # ax2.set_ylim([-3.5, 3.5])
    # ax3.set_ylim([-3.5, 3.5])

    folder_path = DIR / 'experiment' / 'samples' / 'VEs' / 'vocalist-all-500ms'
    file_names = sorted(folder_path.glob('*.wav'))
    file_names = sorted(folder_path.glob('*.wav'))
    results = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    single_envs = dict()
    for file_name in file_names:
        distance = file_name.name[file_name.name.find('dist-') + len('dist-'):file_name.name.rfind('_try')]
        sig = slab.Sound(file_name)
        res = sig.envelope()
        single_envs[file_name.name] = res
        results[int(distance)][str(file_name.stem)] = res
    envs = [None] * 5
    for i in range(5):
        envs[i] = numpy.mean([results[i + 1][key].data for id, key in enumerate(results[i + 1])], axis=0)
    total_average_energy = numpy.mean(numpy.mean(envs, axis=0), axis=1)

    fig, axes = plt.subplots(nrows=4, ncols=1)
    topo1 = plt.subplot2grid((5, 3), (0, 0), rowspan=2)
    topo2 = plt.subplot2grid((5, 3), (0, 1), rowspan=2)
    topo3 = plt.subplot2grid((5, 3), (0, 2), rowspan=2)
    evoked_fig = plt.subplot2grid((5, 3), (2, 0), rowspan=2, colspan=3)
    gfp_fig = plt.subplot2grid((5, 3), (4, 0), colspan=3)
    # ax5 = plt.subplot2grid((5, 3), (5, 0), colspan=3)
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    # mne.viz.topomap.plot_evoked_topomap(combined_evokeds_avrgd, times=0.226, show=False, axes=ax1, colorbar=False)
    combined_evokeds_avrgd.plot_topomap(FCz_peak_times[0], axes=topo1, show=False, colorbar=False)
    combined_evokeds_avrgd.plot_topomap(FCz_peak_times[1], axes=topo2, show=False, colorbar=False)
    combined_evokeds_avrgd.plot_topomap(FCz_peak_times[2], axes=topo3, show=False, colorbar=False)
    mne.viz.plot_evoked(combined_evokeds_avrgd, spatial_colors=True, axes=evoked_fig)
    evoked_fig.axvline(0, color='red')
    evoked_fig.axvline(FCz_peak_times[0], color='#111111', ls='--')
    evoked_fig.axvline(FCz_peak_times[1], color='#111111', ls='--')
    evoked_fig.axvline(FCz_peak_times[2], color='#111111', ls='--')
    evoked_fig.set_title("Evoked Response")
    evoked_fig.set_xlim([-0.1, 0.5])
    evoked_fig.set_ylim([-3.5, 3.5])
    mne.viz.plot_evoked(combined_evokeds_avrgd, spatial_colors=True, gfp="only", axes=gfp_fig)
    gfp_fig.axvline(0, color='red')
    gfp_fig.axvline(FCz_peak_times[0], color='#111111', ls='--')
    gfp_fig.axvline(FCz_peak_times[1], color='#111111', ls='--')
    gfp_fig.axvline(FCz_peak_times[2], color='#111111', ls='--')
    gfp_fig.set_title("GFP")
    gfp_fig.set_xlim([-0.1, 0.5])
    gfp_fig.set_ylim([-0.1, 1.6])
    # ax5.plot(total_average_energy, color='#D55E00')
    # ax5.fill_between(numpy.arange(0, len(envs[0])), total_average_energy, color='#D55E00', alpha=0.2)
    # ax5.set_xlim((-0.2 * 44100, 0.7 * 44100))
    # ax5.set_xticks([-0.1 * 44100, 0, 0.1 * 44100, 0.2 * 44100, 0.3 * 44100, 0.4 * 44100, 0.5 * 44100, 0.6 * 44100],
    #            [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # ax5.set_xlabel("Time (s)")
    # ax5.set_ylim((0, numpy.amax(total_average_energy)))
    # ax5.set_yticks(numpy.linspace(numpy.amin(total_average_energy), numpy.amax(total_average_energy), 5), numpy.linspace(0, 1, 5))
    # ax5.set_title("Average sound energy")
    # Do cluster-based permutation analysis over all
    # subjects to look for statistical significant
    # differences between conditions.
    # get difference response between two conditions.
    evoked_diff = mne.combine_evoked(
        [evokeds_avrgd["vocal_effort/4"], evokeds_avrgd["vocal_effort/5"]], weights=[-1, 1])
    # evoked_diff.plot_joint()  # plot difference
    # evoked_diff.plot_image()  # plot difference
    # get adjacency matrix.
    adjacency, _ = mne.channels.find_ch_adjacency(
        evokeds_avrgd["vocal_effort/1"].info, None)
    # Transpose data of every condition into recuired shape for cluster analysis.
    # recuired shape:  X = observations x time x space.
    # observation: evoked response of one subject. I.e.: X = [5, 1251, 64] if subjects == 5.
    for condition in evokeds:
        evokeds_data[condition] = np.array(
            [evokeds[condition][e].get_data() for e in range(len(ids))])
        evokeds_data[condition] = evokeds_data[condition].transpose(0, 2, 1)
    # Choose conditions of interest in evokeds_data.
    X = [evokeds_data["vocal_effort/4"], evokeds_data["vocal_effort/5"]]
    t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
        X, threshold=dict(start=.2, step=.2), adjacency=adjacency, n_permutations=1000, n_jobs=-1)
    significant_points = cluster_pv.reshape(t_obs.shape).T < .05
    evoked_diff.plot_image(mask=significant_points,
                           show_names="all", titles=None)

    # permutation test for one subject.
    epochs = mne.read_epochs(
        "D:/Projects/distance_perception/analysis/data/pilot/ze1mss/epochs/ze1mss-epo.fif")
    epochs = epochs.crop(tmin=-0.3, tmax=0.7)
    diff = mne.combine_evoked(
        [epochs["distance/20"].average(), epochs["distance/2000"].average()], weights=[1, -1])
    diff.plot_joint()
    diff.plot_image()
    adjacency, _ = mne.channels.find_ch_adjacency(
        epochs.info, None)
    X = [epochs["distance/20"].get_data().transpose(0, 2, 1),
         epochs["distance/2000"].get_data().transpose(0, 2, 1)]
    t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
        X, threshold=dict(start=.2, step=.2), adjacency=adjacency, n_permutations=1000, n_jobs=-1)
    significant_points = cluster_pv.reshape(t_obs.shape).T < .05
    diff.plot_image(mask=significant_points,
                    show_names="all", titles=None)
