import os
import pathlib
import numpy as np
import mne
from matplotlib import pyplot as plt, patches
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA

mapping = {"1": "Fp1", "2": "Fp2", "3": "F7", "4": "F3", "5": "Fz", "6": "F4", "7": "F8", "8": "FC5", "9": "FC1",
           "10": "FC2", "11": "FC6", "12": "T7", "13": "C3", "14": "Cz", "15": "C4", "16": "T8", "17": "TP9", "18": "CP5", "19": "CP1",
           "20": "CP2", "21": "CP6", "22": "TP10", "23": "P7", "24": "P3", "25": "Pz", "26": "P4", "27": "P8", "28": "PO9", "29": "O1",
           "30": "Oz", "31": "O2", "32": "PO10", "33": "AF7", "34": "AF3", "35": "AF4", "36": "AF8", "37": "F5", "38": "F1", "39": "F2",
           "40": "F6", "41": "FT9", "42": "FT7", "43": "FC3", "44": "FC4", "45": "FT8", "46": "FT10", "47": "C5", "48": "C1", "49": "C2",
           "50": "C6", "51": "TP7", "52": "CP3", "53": "CPz", "54": "CP4", "55": "TP8", "56": "P5", "57": "P1", "58": "P2", "59": "P6",
           "60": "PO7", "61": "PO3", "62": "POz", "63": "PO4", "64": "PO8"}
tmin = -0.5
tmax = 2.0
event_id = {
    'deviant': 1,
    'control': 2,
    'distance/20': 3,
    'distance/200': 4,
    'distance/1000': 5,
    'distance/2000': 6,
    'button_press': 7
}

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

"""
functions for filtering, autorejecting epochs(AutoReject),
interpolating bad channels and rereferencing.
"""


def filtering(raw, notch=None, highpass=None, lowpass=None):
    """
    Filter the data. Make a 2 by 2 plot with time
    series data and power spectral density before and after.
    """
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle("Power Spectral Density")
    ax[0].set_title("before removing power line noise")
    ax[1].set_title("after removing power line noise")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    raw.plot_psd(average=True, area_mode=None, ax=ax[0], show=False)
    if notch is not None:
        raw.notch_filter(freqs=notch)
    if lowpass is not None:
        raw.filter(h_freq=lowpass, l_freq=None)
    if highpass is not None:
        raw.filter(h_freq=None, l_freq=highpass)
    raw.plot_psd(average=True, area_mode=None, ax=ax[1], show=False)
    fig.tight_layout()
    fig.savefig(
        fig_folder / pathlib.Path("remove_power_line_noise.pdf"), dpi=800)
    plt.close()
    return raw


def autoreject_epochs(epochs, n_interpolate=[1, 4, 8, 16]):
    ar = AutoReject(n_interpolate=n_interpolate)
    epochs = ar.fit_transform(epochs)
    fig, ax = plt.subplots(2)
    # plotipyt histogram of rejection thresholds
    ax[0].set_title("Rejection Thresholds")
    ax[0].hist(1e6 * np.array(list(ar.threshes_.values())), 30,
               color='g', alpha=0.4)
    ax[0].set(xlabel='Threshold (μV)', ylabel='Number of sensors')
    # plot cross validation error:
    loss = ar.loss_['eeg'].mean(axis=-1)  # losses are stored by channel type.
    im = ax[1].matshow(loss.T * 1e6, cmap=plt.get_cmap('viridis'))
    ax[1].set_xticks(range(len(ar.consensus)))
    ax[1].set_xticklabels(['%.1f' % c for c in ar.consensus])
    ax[1].set_yticks(range(len(ar.n_interpolate)))
    ax[1].set_yticklabels(ar.n_interpolate)
    # Draw rectangle at location of best parameters
    idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
    rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax[1].add_patch(rect)
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].set(xlabel=r'Consensus percentage $\kappa$',
              ylabel=r'Max sensors interpolated $\rho$',
              title='Mean cross validation error (x 1e6)')
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(fig_folder / pathlib.Path("reject_epochs.pdf"), dpi=800)
    plt.close()
    return epochs


# only needed when no average reference is chosen.
def interpolate_bads(epochs):
    ransac = Ransac()
    evoked = epochs.average()  # for plotting
    epochs = ransac.fit_transform(epochs)
    evoked.info["bads"] = ransac.bad_chs_
    # plot evoked response with and without interpolated bads:
    fig, ax = plt.subplots(2)
    evoked.plot(exclude=[], axes=ax[0], show=False)
    ax[0].set_title('Before RANSAC')
    evoked = epochs.average()  # for plotting
    evoked.info["bads"] = ransac.bad_chs_
    evoked.plot(exclude=[], axes=ax[1], show=False)
    ax[1].set_title('After RANSAC')
    fig.tight_layout()
    fig.savefig(
        fig_folder / pathlib.Path("interpolate_bad_channels.pdf"), dpi=800)
    plt.close()
    return epochs


def set_ref(epochs, type="average", ref=None, apply=True):
    """
    Create a robust average reference by first interpolating the bad channels
    to exclude outliers. The reference is applied as a projection. Return
    epochs with reference projection applied if apply=True
    """
    if type == "average":
        ransac = Ransac()
        epochs_tmp = epochs.copy()
        epochs_tmp = ransac.fit_transform(epochs)
        mne.set_eeg_reference(
            epochs_tmp, ref_channels="average", projection=True)
        robust_avg_proj = epochs_tmp.info["projs"][0]
        del epochs_tmp
        epochs.info["projs"].append(robust_avg_proj)
    else:
        mne.set_eeg_reference(epochs, ref_channels=ref, projection=True)
    if apply:
        epochs.apply_proj()
    return epochs


if __name__ == "__main__":
    # get directory in which pilot folders are
    DIR = pathlib.Path(os.getcwd()) / 'analysis' / 'data' / 'pilot'
    fig_path = pathlib.Path(os.getcwd()) / 'analysis' / 'figures'
    # get subject ids
    ids = list(name for name in os.listdir(DIR)
               if os.path.isdir(os.path.join(DIR, name)))
    # make raws and save them into raw_folder
    for id in ids:
        folder_path = DIR / id
        header_files = folder_path.glob('*.vhdr')
        raw_files = []
        for header_file in header_files:
            raw_files.append(mne.io.read_raw_brainvision(
                header_file, preload=True))
        raw = mne.concatenate_raws(raw_files)
        epochs_folder = DIR / id / 'epochs'
        raw_folder = DIR / id / 'raw_data'
        fig_folder = fig_path / id
        for folder in epochs_folder, raw_folder, fig_folder:
            if not os.path.isdir(folder):
                os.makedirs(folder)
        raw.rename_channels(mapping)
        montage_path = pathlib.Path(os.getcwd()) / \
            'analysis' / 'AS-96_REF.bvef'
        montage = mne.channels.read_custom_montage(fname=montage_path)
        raw.set_montage(montage)
        raw.save(raw_folder / pathlib.Path(id + "_raw.fif"), overwrite=True)
        # bandpass filter between 1 and 40 Hz
        raw = filtering(raw, highpass=1, lowpass=40)
        events = mne.events_from_annotations(raw)[0]  # get events
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax,
                            event_id=event_id, preload=True)  # epoch raw data
        del raw
        epochs = set_ref(epochs)  # average reference
        reference = mne.preprocessing.read_ica(fname=pathlib.Path(
            os.getcwd()) / 'analysis' / 'ica_reference.fif')
        components = reference.labels_["blinks"]
        ica = ICA(n_components=5, random_state=97)
        ica.fit(epochs)
        for component in components:
            mne.preprocessing.corrmap([reference, ica], template=(0, components[component]),
                                      label="blinks", plot=False, threshold=0.75)
            ica.apply(epochs, exclude=ica.labels_["blinks"])
        epochs_clean = autoreject_epochs(epochs)
        epochs_clean.save(
            epochs_folder / pathlib.Path(id + '-epo.fif'), overwrite=True)

    # test
    id = 'ze1mss'
    raw = mne.io.read_raw_fif(
        str(raw_folder) + "/" + id + "_raw.fif", preload=True)
    ica = ICA(n_components=10, random_state=97)
    ica.fit(epochs)
    ica.plot_components()
    ica.plot_sources(epochs)
    ica.plot_properties(raw, picks=[0, 2])
    ica.exclude = [0, 1]
    ica.apply(epochs)
    raw.plot()
    epochs.average().plot()
    epochs_clean.average().plot()
    mne.preprocessing.ICA.save(reference, fname=pathlib.Path(
        os.getcwd()) / 'analysis' / 'ica_reference.fif', overwrite=True)
    component = reference.labels_["blinks"]
    reference.get_components()
    reference.plot_components()
    raw.plot()
