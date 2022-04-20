import json
from mne.stats import spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency, make_1020_channel_selections
from scipy.stats import ttest_ind
import numpy as np
import pathlib
import mne
import os
import matplotlib.pyplot as plt
%matplotlib qt


plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])


def ignore_conds(d, *keys):
    return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))


if __name__ == "__main__":
    experiment = "laughter"  # either "noise" or "laughter"
    # get pilot folder directory.
    DIR = pathlib.Path(os.getcwd())
    fig_path = pathlib.Path(os.getcwd()) / "analysis" / "figures"
    sub_DIR = DIR / "analysis" / "data" / f"pilot_{experiment}"  # pilot_laughter or pilot_noise
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
                evokeds[condition.comment].append(condition.crop(-0.3, 0.7))
                if len(evokeds[condition.comment]) == len(ids):
                    evokeds_avrgd[condition.comment] = mne.grand_average(
                        evokeds[condition.comment])
                else:
                    continue
    # show some topographic and line plots to get an impression of evoked responses.
    # plot control condition to have a look at simple auditory evoked signals without reverb info.
    channel = ["Cz"]  # select electrode of interest.
    evokeds_avrgd["deviant"].plot_joint()
    evokeds_avrgd["distance/20"].plot_topomap(times=np.linspace(
        0.05, 0.6, 10), title="Topograhical responses control")
    mne.viz.plot_evoked_topo(evokeds_avrgd["distance/20"])
    # Central electrodes show auditory evoked activity and differences between conditions.
    # Select Cz and look at conditions around 350 ms after stimuluis onset.
    mne.viz.plot_compare_evokeds(ignore_conds(
        evokeds_avrgd, "deviant", "button_press"), picks=channel, cmap=plt.get_cmap('plasma'))
    evokeds_avrgd["distance/2000"].plot_image()

    # Do cluster-based permutation analysis over all
    # subjects to look for statistical significant
    # differences between conditions.
    # get difference response between two conditions.
    evoked_diff = mne.combine_evoked(
        [evokeds_avrgd["distance/20"], evokeds_avrgd["control"]], weights=[1, -1])
    evoked_diff.plot_joint()  # plot difference
    evoked_diff.plot_image()  # plot difference
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
