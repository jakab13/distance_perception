import mne
import numpy as np
import os
# import sys
# path = os.getcwd() + "\\src\\" + "eeg_tools"
# sys.path.append(path)
# import analysis.EEG.eeg_tools.src.eeg_tools.utils as utils
# import pathlib

# TODO: make PCA work for more than one component. Right now only works if n_components=1.

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def snr(epochs):
    epochs_tmp = epochs.copy()
    n_epochs = epochs_tmp.get_data().shape[0]
    if not n_epochs % 2 == 0:
        epochs_tmp = epochs_tmp[:-1]
    n_epochs = epochs_tmp.get_data().shape[0]
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp.get_data()[i, :, :] = -epochs_tmp.get_data()[i, :, :]
    noises = epochs_tmp.average().get_data()
    shuffled_noises = shuffle_along_axis(noises, axis=1)
    signals = shuffled_noises.copy()
    for idx, noise in enumerate(shuffled_noises):
        for epoch in epochs.average().get_data():
            signal = epoch-noise
        signals[idx] = signal
    snr = signals / noises
    rms = np.mean(np.sqrt(snr**2))
    return rms


def PCA(epochs, n_components=1):
    X = epochs.get_data()
    n_epochs, n_channels, n_times = epochs.get_data().shape
    X -= np.expand_dims(X.mean(axis=2), axis=2)  # center data on 0
    X = np.transpose(epochs._data,
                     (1, 0, 2)).reshape(n_channels,
                                        n_epochs * n_times).T  # concatenate
    C0 = X.T @ X  # Data covariance Matrix
    D, P = np.linalg.eig(C0)  # eigendecomposition of C0
    idx = np.argsort(D)[::-1][0:n_components]   # sort array
    # by descending magnitude
    D = D[idx]
    P = P[:, idx]  # rotation matrix
    pca_evokeds = dict()
    for cond in epochs.event_id.keys():
        # use rotation matrix on every single condition
        n_epochs, n_channels, n_times = epochs[cond]._data.shape
        X = epochs[cond]._data
        X -= np.expand_dims(X.mean(axis=2), axis=2)  # center data on 0
        X = np.transpose(epochs[cond]._data,
                         (1, 0, 2)).reshape(n_channels, n_epochs * n_times).T
        Y = X @ P  # get principle components
        pca = np.reshape(Y.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
        pca_evoked = mne.EvokedArray(np.mean(pca, axis=0),
                                     mne.create_info(
                                     n_components, epochs[cond].info["sfreq"],
                                     ch_types="eeg"),
                                     tmin=epochs[cond].tmin)
        for component in range(n_components):
            # pca_evokeds[cond] = pca_evoked.pick_channels(ch_names=list(str(component)) for component in range(n_components)))
             pca_evokeds[cond] = pca_evoked.pick_channels(ch_names=list(str(component)))
    return pca_evokeds


def quality_check(ids, out_folder, root_dir, n_figs=12, fig_size=(60,60)):
    axs_size = int(round(np.sqrt(len(ids)) + 0.5))  # round up
    _fig_paths = utils.find(path=root_dir, mode="pattern", pattern="*.jpg")
    n_sort = n_figs
    if not os.path.isdir(out_folder):
        os.makedirs(pathlib.Path(out_folder))
    for figure in range(n_figs):
        _fig_paths_sorted = _fig_paths[::n_sort]
        fig, axs = plt.subplots(axs_size, axs_size, figsize=fig_size)
        axs = axs.flatten()
        for i, fig_path in enumerate(_fig_paths_sorted):
            img = plt.imread(fig_path)
            axs[i].imshow(img)
            axs[i].set_axis_off()
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(pathlib.Path(out_folder / os.path.basename(os.path.normpath(fig_path))))
        plt.close()
        for fig_path in _fig_paths_sorted:
            if fig_path in _fig_paths:
                _fig_paths.remove(fig_path)
        n_sort -= 1
        fig.clf()


def get_evokeds(ids, root_dir, return_average=False):
    all_evokeds = dict()
    for id in ids:
        evokeds = utils.read_object("evokeds", root_dir, id)
        for condition in evokeds:
            if condition.comment not in all_evokeds.keys():
                all_evokeds[condition.comment] = [condition]
            else:
                all_evokeds[condition.comment].append(condition)
    if return_average == True:
        evokeds_avrgd = dict()
        for key in all_evokeds:
            evokeds_avrgd[key] = mne.grand_average(all_evokeds[key])
        return all_evokeds, evokeds_avrgd
    else:
        return all_evokeds


if __name__ == "__main__":
    import sys
    import os
    path = os.getcwd() + "\\src\\" + "eeg_tools"
    sys.path.append(path)
    from matplotlib import pyplot as plt
    import analysis.EEG.eeg_tools.src.eeg_tools.settings as settings
    from autoreject import Ransac
    import mne
    import numpy as np
    import analysis.EEG.eeg_tools.src.eeg_tools.utils as utils
    import pathlib
    import analysis.EEG.eeg_tools.src.eeg_tools.preprocessing as preprocessing
    raw = mne.io.read_raw_fif("D:\EEG\distance_perception\pinknoise\data\\2ojf8e\\raw\\2ojf8e_raw.fif", preload=True)
    raw.filter(1, 40)
    events = mne.pick_events(events=mne.events_from_annotations(raw)[0], exclude=1)
    epochs = mne.Epochs(raw, events=events, **settings.cfg["epochs"], preload=True)
    epochs = preprocessing.set_ref(epochs, **settings.cfg["rereference"], plot=False)
    epochs_tmp = epochs.copy()
    n_epochs = epochs_tmp.get_data().shape[0]
    if not n_epochs % 2 == 0:
        epochs_tmp = epochs_tmp[:-1]
    n_epochs = epochs_tmp.get_data().shape[0]
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp.get_data()[i, :, :] = -epochs_tmp.get_data()[i, :, :]
    noises = epochs_tmp.average().get_data()
    shuffled_noises = shuffle_along_axis(noises, axis=1)
    signals = shuffled_noises.copy()
    for idx, noise in enumerate(shuffled_noises):
        for epoch in epochs.average().get_data():
            signal = epoch-noise
        signals[idx] = signal
    snr = signals / noises
    rms = np.mean(np.sqrt(snr**2))
