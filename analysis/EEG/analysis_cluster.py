import os
import pathlib
import mne
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rc('image', cmap='hot')

EXPERIMENT = "vocal_effort"

DIR = pathlib.Path(os.getcwd())
sub_DIR = DIR / "analysis" / "EEG" / "data" / EXPERIMENT
subject_ids = list(name for name in os.listdir(sub_DIR) if os.path.isdir(os.path.join(sub_DIR, name)))
with open(DIR / "analysis"/ "EEG" / "preproc_config.json") as file:
    cfg = json.load(file)

epochs_all = dict()
evokeds_all = dict()
evokeds_group = dict()
for subject_id in subject_ids:
    epochs_folder = sub_DIR / subject_id / "epochs"
    epochs = mne.read_epochs(epochs_folder / pathlib.Path(subject_id + '-epo.fif'))
    epochs_all[subject_id] = epochs

    evokeds = epochs.average(by_event_type=True)
    evokeds = [evoked for evoked in evokeds if evoked.comment != "deviant"]
    evokeds_all[subject_id] = evokeds

for subject_id in subject_ids:
    evokeds = evokeds_all[subject_id]
    for evoked in evokeds:
        if evoked.comment in evokeds_group:
            evokeds_group[evoked.comment].append(evoked)
        else:
            evokeds_group[evoked.comment] = [evoked]

subject_id = subject_ids[0]
for subject_id in subject_ids:
    evokeds = evokeds_all[subject_id]
    for idx, evoked in enumerate(evokeds):
        evoked.plot_joint()
        fig_title = f"Evoked average (sub={subject_id}) - Distance {idx + 1}"
        plt.title(fig_title)
        plt.savefig(DIR / "analysis" / "EEG" / "figures" / EXPERIMENT / "epochs" / fig_title)
        plt.close()
    mne.viz.plot_compare_evokeds(evokeds, cmap="copper")
    fig_title = f"Compare Evokeds (sub={subject_id})"
    plt.savefig(DIR / "analysis" / "EEG" / "figures" / EXPERIMENT / "evokeds" / fig_title)
    plt.close()

for i in range(4):
    evokeds_diff = [mne.combine_evoked([evokeds[0], evokeds[-i-1]], weights=[1, -1]) for evokeds in evokeds_all.values()]
    mne.viz.plot_compare_evokeds({"diff": evokeds_diff}, picks="FCz")

for subject_id in subject_ids:
    evokeds = evokeds_all[subject_id]
    evokeds_diff = [mne.combine_evoked([evokeds[0], evokeds[-1]], weights=[1, -1]) for evokeds in evokeds_all.values()]
    mne.viz.plot_compare_evokeds({"diff": evokeds_diff}, picks="FCz")

# Spatio temporal clustering ========================================================================

for subject_id in subject_ids:
    epochs = epochs_all[subject_id]
    event_id = {k: v for k, v in epochs.event_id.items() if k != "deviant"}
    epochs.equalize_event_counts(event_id)

    X = [epochs[event_name].get_data(copy=False) for event_name in event_id]
    X = list()
    for condition, evk in evokeds_group.items():
        a = [e.get_data() for e in evk]
        X.append(a)
    X = [np.transpose(x, (0, 2, 1)) for x in X]

    adjacency, ch_names = mne.channels.find_ch_adjacency(epochs.info, ch_type="eeg")
    # mne.viz.plot_ch_adjacency(epochs.info, adjacency, ch_names)

    tail = 1
    alpha_cluster_forming = 0.001
    n_conditions = len(event_id)
    n_observations = len(X[0])
    dfn = n_conditions - 1
    dfd = n_observations - n_conditions
    f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

    cluster_stats = mne.stats.spatio_temporal_cluster_test(
        X,
        n_permutations=1000,
        threshold=f_thresh,
        tail=tail,
        n_jobs=None,
        buffer_size=None,
        adjacency=adjacency,
    )
    F_obs, clusters, p_values, _ = cluster_stats

    p_accept = 0.05
    good_cluster_inds = np.where(p_values < p_accept)[0]
    evokeds = {cond: epochs[cond].average() for cond in event_id}

    # for i_clu, clu_idx in enumerate(good_cluster_inds):
    #     time_inds, space_inds = np.squeeze(clusters[clu_idx])
    #     ch_inds = np.unique(space_inds)
    #     time_inds = np.unique(time_inds)
    #     sig_times = epochs.times[time_inds]
    #     p_value = p_values[clu_idx]
    #     cluster_data = {
    #         "experiment": EXPERIMENT",
    #         "subject_id": subject_id,
    #         "p_value": p_value,
    #         "t_min": time_inds[0],
    #         "t_max": time_inds[-1],
    #         "t_mean": np.mean([time_inds[0], time_inds[-1]])
    #     }
    #     output_path = 'spatio_temporal_clusters.csv'
    #     row = pd.DataFrame.from_dict([cluster_data])
    #     row.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        f_map = F_obs[time_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = epochs.times[time_inds]

        # create spatial mask
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(16, 4))

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap="Reds",
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
        )
        image = ax_topo.images[0]

        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.5)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="600%", pad=2.2)
        title = "Cluster #{0}, {1} sensor".format(i_clu + 1, len(ch_inds))
        if len(ch_inds) > 1:
            title += "s (mean)"
        mne.viz.plot_compare_evokeds(
            evokeds,
            title=title,
            picks=ch_inds,
            axes=ax_signals,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
            cmap="copper"
        )

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="green", alpha=0.3
        )
        # fig_title = f"Spatio-Temporal cluster (sub={subject_id} at {sig_times[0]}-{sig_times[-1]}s).png"
        fig_title = f"Spatio-Temporal cluster (Grand Average at {sig_times[0]}-{sig_times[-1]}s).png"
        plt.savefig(DIR / "analysis" / "EEG" / "figures" / EXPERIMENT / "clusters" / str(fig_title), format="png")
        plt.close()