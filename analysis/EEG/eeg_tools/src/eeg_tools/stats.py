import mne
import scipy.stats
from mne.stats import spatio_temporal_cluster_test
import numpy as np
import matplotlib.pyplot as plt


def target_test(evokeds, time_windows, electrodes, conditions, ind=False, parametric=False):
    index = "time"
    report = "{electrode}, time: {tmin}-{tmax} s; stat={statistic:.3f}, p={p}"
    print("Targeted t-test results:")
    for tmin, tmax in time_windows:
        cond0 = mne.grand_average(evokeds[conditions[0]]).copy().crop(tmin, tmax).to_data_frame(index=index)
        cond1 = mne.grand_average(evokeds[conditions[1]]).copy().crop(tmin, tmax).to_data_frame(index=index)
        for electrode in electrodes:
            # extract data
            A = cond0[electrode]
            B = cond1[electrode]
            # conduct t test
            if ind and not parametric:
                s, p = scipy.stats.wilcoxon(A, B)
            elif ind and parametric:
                s, p = scipy.stats.ttest_ind(A, B)
            elif not ind and not parametric:
                s, p = scipy.stats.mannwhitneyu(A, B)
            elif not ind and parametric:
                s, p = scipy.stats.ttest_ind(A, B)
            else:
                print("Desired test not found. Aborting ... ")
            # display results
            format_dict = dict(electrode=electrode, tmin=tmin, tmax=tmax, statistic=s, p=p)
            print(report.format(**format_dict))
    return s, p


def spatem_test(evokeds, conditions, pval=.05, n_permutations=1000, n_jobs=-1, plot=True):
    evokeds_data = dict()
    for condition in evokeds:
        evokeds_data[condition] = np.array(
            [evokeds[condition][e].get_data() for e in range(len(evokeds[list(evokeds.keys())[0]]))])
        evokeds_data[condition] = evokeds_data[condition].transpose(0, 2, 1)
    adjacency, _ = mne.channels.find_ch_adjacency(
        evokeds[list(evokeds.keys())[0]][0].info, None)
    X = [evokeds_data[conditions[0]], evokeds_data[conditions[1]]]
    t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
        X, threshold=dict(start=.2, step=.2), adjacency=adjacency, n_permutations=n_permutations, n_jobs=n_jobs)
    significant_points = cluster_pv.reshape(t_obs.shape).T < pval
    if plot:
        cond0 = mne.grand_average(evokeds[conditions[0]])
        cond1 = mne.grand_average(evokeds[conditions[1]])
        evoked_diff = mne.combine_evoked(
            [cond0, cond1], weights=[1, -1])
        evoked_diff.plot_joint()
        plt.close()
        selections = mne.channels.make_1020_channel_selections(evoked_diff.info, midline="z")
        fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
        axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
        evoked_diff.plot_image(axes=axes, group_by=selections, colorbar=False, show=False,
                               mask=significant_points, show_names="all", titles=None)
        plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=.3,
                     label="µV")
        plt.show()


if "__name__" == "__main__":
    import sys
    import numpy as np
    sys.path.append("D:/Projects/eeg_tools/src/eeg_tools")
    import analysis
    import settings
    evokeds = analysis.get_evokeds(settings.ids, settings.root_dir, return_average=False)
    spatem_test(evokeds)
    s, p = target_test(evokeds, time_windows=[(-0.2, 0)], electrodes=["FCz"], conditions=["pinknoise/1", "pinknoise/5"])
