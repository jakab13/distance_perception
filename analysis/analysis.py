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
# %matplotlib qt


plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])


def ignore_conds(d, *keys):
    return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))


if __name__ == "__main__":
    experiment = "vocal_effort"  # either "noise" or "laughter"
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
    ids = ids_300ms
    # make dictionaries with empty event keys.
    # first copy config file to prevent changes.
    evokeds, evokeds_avrgd, evokeds_data = cfg["epochs"][f"event_id_{experiment}"].copy(
    ), cfg["epochs"][f"event_id_{experiment}"].copy(), cfg["epochs"][f"event_id_{experiment}"].copy()
    for key in cfg["epochs"][f"event_id_{experiment}"]:
        evokeds[key], evokeds_avrgd[key], evokeds_data[key] = list(), list(), list()
        # get evokeds for every condition and subject.
    # peak_times = {}
    # peak_amplitudes = {}
    for id in ids:
        evokeds_folder = sub_DIR / id / "evokeds"
        evoked = mne.read_evokeds(
            evokeds_folder / pathlib.Path(id + '-ave.fif'))
        for condition in evoked:
            if condition.comment in evokeds:
                evokeds[condition.comment].append(condition.crop(-0.3, 0.7))
                if len(evokeds[condition.comment]) == len(ids):
                    evokeds_avrgd[condition.comment] = mne.grand_average(
                        evokeds[condition.comment])
                else:
                    continue
        # peak_times[id] = {}
        # peak_amplitudes[id] = {}
        # for condition in evoked:
        #     print(condition)
        #     evoked_picked = condition.copy()
        #     evoked_picked.pick_channels(["FCz"])
        #     peak_data = evoked_picked.get_peak(return_amplitude=True)
        #     peak_times[id][condition.comment] = peak_data[1]
        #     peak_amplitudes[id][condition.comment] = peak_data[2]
    # show some topographic and line plots to get an impression of evoked responses.
    # plot control condition to have a look at simple auditory evoked signals without reverb info.
    # channels = ["FCz"]  # select electrode of interest.
    # evokeds_avrgd["deviant"].plot_joint()
    # evokeds_avrgd["distance/20"].plot_topomap(times=np.linspace(
    #     0.05, 0.6, 10), title="Topograhical responses control")
    # mne.viz.plot_evoked_topo(evokeds_avrgd["distance/20"])
    # Central electrodes show auditory evoked activity and differences between conditions.
    # Select Cz and look at conditions around 350 ms after stimuluis onset.
    # mne.viz.plot_compare_evokeds(ignore_conds(
    #     evokeds_avrgd, "deviant", "button_press"), picks=channels, cmap=plt.get_cmap('plasma'))
    # evokeds_avrgd["vocal_effort/1"].plot_image()
    combined_evokeds_avrgd = mne.combine_evoked(
        [evokeds_avrgd["vocal_effort/1"], evokeds_avrgd["vocal_effort/2"],
         evokeds_avrgd["vocal_effort/3"], evokeds_avrgd["vocal_effort/4"],
         evokeds_avrgd["vocal_effort/5"]], weights=[0.2, 0.2, 0.2, 0.2, 0.2])
    # combined_evokeds_avrgd.plot_topomap(times=numpy.linspace(0.15, 0.5, 20),
    #                                     title="Combined Evoked (long) (150-500ms)",
    #                                     vmin=-5, vmax=5)
    time_ranges_of_interest = [
        (0.05, 0.14),
        (0.22, 0.27)
    ]
    combined_evokeds_avrgd.plot_joint(times=[0.108, 0.226, 0.341])
    combined_evokeds_avrgd.plot(gfp="only", spatial_colors=True)
    combined_evokeds_avrgd.plot(gfp=True, spatial_colors=True)

    plt.savefig(fig_path / "Combined Evoked (long) (150-500ms).png")

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
