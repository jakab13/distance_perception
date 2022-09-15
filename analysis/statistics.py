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
import scipy


def ignore_conds(d, *keys):
    return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))

# TODO: implement bootstrapping.
# TODO: do cluster permutation test with all distances together against control.

def jttrend(matrix):
    conditions = matrix.shape[0]
    u_total = 0
    for i in range(0, conditions - 1):
        for j in range(i + 1, conditions):
            for i_elem in matrix[i]:
                for j_elem in matrix[j]:
                    if i_elem is not None and j_elem is not None:
                        u = (i_elem < j_elem) + 0.5 * (i_elem == j_elem)
                        u_total += u
    nj = np.zeros(matrix.shape[0], dtype=int)
    for j, column in enumerate(matrix):
        for elem in column:
            if elem is not None:
                nj[j] += 1
    n = sum(nj)
    numerator = u_total - ((n*n - sum(nj*nj)) / 4)
    denominator = np.sqrt((n*n * (2*n + 3) - sum(abs(nj*nj * (2*nj + 3))))/72)
    z = numerator/denominator
    p = scipy.stats.norm.sf(abs(z)) #one-sided
    return z, p

def evoked_jttrend(matrix, window_length=20):
    total_length = matrix.shape[1]
    idx = 0
    jt_array = numpy.empty(total_length - window_length)
    jt_array_p_values = numpy.empty(total_length - window_length)
    while idx < total_length - window_length:
        matrix_window = matrix[:, idx: idx + window_length]
        jt_array[idx] = jttrend(matrix_window)[0]
        jt_array_p_values[idx] = jttrend(matrix_window)[1]
        idx += 1
    return jt_array, jt_array_p_values

matrix = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
matrix = np.array([
    [0, 0, 1, 1, 2, 2, 4, 9, None, None],
    [0, 0, 5, 7, 8, 11, 13, 23, 25, 97],
    [2, 3, 6, 9, 10, 11, 11, 12, 21, None],
    [0, 3, 5, 6, 10, 19, 56, 100, 132, None],
    [2, 4, 6, 6, 6, 7, 18, 39, 60, None]
])
matrix = np.array([
    [45, 35, 51, 31, 62, None],
    [59, 53, 31, 47, 42, 59],
    [49, 69, 52, 55, 63, None],
    [72, 55, 65, 58, 61, 51]
])

if __name__ == "__main__":
    experiment = "noise"  # either "noise" or "laughter"
    # get pilot folder directory.
    DIR = pathlib.Path(os.getcwd())
    fig_path = pathlib.Path(os.getcwd()) / "analysis" / "figures"
    sub_DIR = DIR / "analysis" / "data" / f"pilot_{experiment}"
    with open(DIR / "analysis" / "preproc_config.json") as file:
        cfg = json.load(file)
    # get subject ids
    ids = list(name for name in os.listdir(sub_DIR)
               if os.path.isdir(os.path.join(sub_DIR, name)))
    # make dictionaries with empty event keys.
    # first copy config file to prevent changes.
    evokeds, evokeds_avrgd, evokeds_data = cfg["epochs"][f"event_id_{experiment}"].copy(
    ), cfg["epochs"][f"event_id_{experiment}"].copy(), cfg["epochs"][f"event_id_{experiment}"].copy()
    for key in cfg["epochs"][f"event_id_{experiment}"]:
        evokeds[key], evokeds_avrgd[key], evokeds_data[key] = list(), list(), list()
        # get evokeds for every condition and subject.
    for id in ids:
        evokeds_folder = sub_DIR / id / "evokeds"
        evoked = mne.read_evokeds(
            evokeds_folder / pathlib.Path(id + '-ave.fif'))
        for condition in evoked:
            if condition.comment in evokeds:
                evokeds[condition.comment].append(condition)
                if len(evokeds[condition.comment]) == len(ids):
                    evokeds_avrgd[condition.comment] = mne.grand_average(
                        evokeds[condition.comment])
                else:
                    continue
    # show some topographic and line plots to get an impression of evoked responses.
    # plot control condition to have a look at simple auditory evoked signals without reverb info.
    channel = ["Cz"]  # select electrode of interest.
    evokeds_avrgd["distance/20"].plot_joint()
    evokeds_avrgd["button_press"].plot_topomap(times=np.linspace(
        -0.2, 0.6, 10), title=None)
    mne.viz.plot_evoked_topo(evokeds_avrgd["distance/20"])
    # Central electrodes show auditory evoked activity and differences between conditions.
    # Select Cz and look at conditions around 350 ms after stimulus onset.
    mne.viz.plot_compare_evokeds(ignore_conds(evokeds, "button_press", "deviant"), picks=channel, cmap=plt.get_cmap('plasma'))  # plot CI
    mne.viz.plot_compare_evokeds(ignore_conds(evokeds_avrgd, "button_press", "deviant"), picks=channel, cmap=plt.get_cmap('plasma'))
    evokeds_avrgd["distance/2000"].plot_image()

    # Do cluster-based permutation analysis over all
    # subjects to look for statistical significant
    # differences between conditions.
    # get difference response between two conditions.
    evoked_diff = mne.combine_evoked(
        [evokeds_avrgd["control"], evokeds_avrgd["distance/2000"]], weights=[1, -1])
    evoked_diff.plot_joint();  # plot difference
    evoked_diff.plot_image();  # plot difference
    mne.viz.plot_compare_evokeds(evoked_diff, picks=channel, cmap=plt.get_cmap('plasma'))
    # get adjacency matrix.
    adjacency, _ = mne.channels.find_ch_adjacency(
        evokeds_avrgd["control"].info, None)
    # Transpose data of every condition into recuired shape for cluster analysis.
    # recuired shape:  X = observations x time x space.
    # observation: evoked response of one subject. I.e.: X = [5, 1251, 64] if subjects == 5.
    for condition in evokeds:
        evokeds_data[condition] = np.array(
            [evokeds[condition][e].get_data() for e in range(len(ids))])
        evokeds_data[condition] = evokeds_data[condition].transpose(0, 2, 1)
    # Choose conditions of interest in evokeds_data.
    X = [evokeds_data["distance/20"], evokeds_data["distance/2000"]]
    t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
        X, threshold=dict(start=.2, step=.2), adjacency=adjacency, n_permutations=1000, n_jobs=-1)
    significant_points = cluster_pv.reshape(t_obs.shape).T < .05
    evoked_diff.plot_image(mask=significant_points,
                           show_names="all", titles=None)


    # permutation test for one subject.
    id = ids[0]
    epochs = mne.read_epochs(
        f"D:/Projects/distance_perception/analysis/data/pilot_{experiment}/{id}/epochs/{id}-epo.fif")
    mne.viz.plot_compare_evokeds(epochs["button_press"].average(), picks=["Cz"])
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
