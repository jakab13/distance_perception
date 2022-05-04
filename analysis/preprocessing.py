import os
import pathlib
import numpy as np
import mne
from matplotlib import pyplot as plt, patches
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA
import json

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

# TODO: change and use filtering functionality to include quasi-perfect notch filter (zapline)
# add more function arguments in every function and put into config file.
"""
Preprocessing pipeline. Execute whole code to iterate through BrainVision data
of every subject and preprocess until evoked responses. Single steps are explained
below.
"""
def snr(epochs):
    """
    Compute signal-to-noise ratio. Take root mean square of noise
    (interval before stimulus onset) and signal (interval where evoked activity is expected)
    and return quotient.
    """
    signal = epochs.copy().crop(0, 0.6).average().get_data()
    noise = epochs.copy().crop(None, 0).average().get_data()
    signal_rms = np.sqrt(np.mean(signal**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    return signal_rms/noise_rms


def filtering(raw, notch=None, highpass=None, lowpass=None):
    """
    Apply FIR filter to the raw dataset. Make a 2 by 2 plot with time
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


def autoreject_epochs(epochs,
                      n_interpolate=[1, 4, 8, 16],
                      consensus=None,
                      cv=10,
                      thresh_method="bayesian optimization",
                      n_jobs=1,
                      random_state=None):
    """
    Automatically reject epochs via AutoReject algorithm:
    Computation of sensor-wise peak-to-peak-amplitude thresholds
    via cross-validation.
    """
    ar = AutoReject(n_interpolate=n_interpolate)
    epochs = ar.fit_transform(epochs)
    fig, ax = plt.subplots(2)
    # plotipyt histogram of rejection thresholds
    ax[0].set_title("Rejection Thresholds")
    ax[0].hist(1e6 * np.array(list(ar.threshes_.values())), 30,
               color="g", alpha=0.4)
    ax[0].set(xlabel="Threshold (μV)", ylabel="Number of sensors")
    # plot cross validation error:
    loss = ar.loss_["eeg"].mean(axis=-1)  # losses are stored by channel type.
    im = ax[1].matshow(loss.T * 1e6, cmap=plt.get_cmap("viridis"))
    ax[1].set_xticks(range(len(ar.consensus)))
    ax[1].set_xticklabels(["%.1f" % c for c in ar.consensus])
    ax[1].set_yticks(range(len(ar.n_interpolate)))
    ax[1].set_yticklabels(ar.n_interpolate)
    # Draw rectangle at location of best parameters
    idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
    rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor="r", facecolor="none")
    ax[1].add_patch(rect)
    ax[1].xaxis.set_ticks_position("bottom")
    ax[1].set(xlabel=r"Consensus percentage $\kappa$",
              ylabel=r"Max sensors interpolated $\rho$",
              title="Mean cross validation error (x 1e6)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(fig_folder / pathlib.Path("reject_epochs.pdf"), dpi=800)
    plt.close()
    return epochs


def set_ref(epochs, type="average", ref=None, apply=True, j_jobs=-1):
    """
    Create a robust average reference by first interpolating the bad channels
    to exclude outliers. The reference is applied as a projection. Return
    epochs with reference projection applied if apply=True.
    epochs: mne.Epoch object.
    type: string.
    ref: string/list of strings. List of reference electrode names. Example: ["P09", "P08"]
    """
    if type == "average":
        ransac = Ransac(n_jobs=j_jobs)  # optimize process speed
        epochs_tmp = epochs.copy()
        epochs_tmp = ransac.fit_transform(epochs)
        mne.set_eeg_reference(
            epochs_tmp, ref_channels=type, projection=True)
        robust_avg_proj = epochs_tmp.info["projs"][0]
        epochs.info["projs"].append(robust_avg_proj)
        evoked = epochs.average()  # for plotting
        evoked.info["bads"] = ransac.bad_chs_
        fig, ax = plt.subplots(2)
        evoked.plot(exclude=[], axes=ax[0], show=False)
        ax[0].set_title('Before RANSAC')
        evoked = epochs_tmp.average()
        evoked.info["bads"] = ransac.bad_chs_
        evoked.plot(exclude=[], axes=ax[1], show=False)
        ax[1].set_title("After RANSAC")
        fig.tight_layout()
        fig.savefig(
            fig_folder / pathlib.Path("interpolate_bad_channels.pdf"), dpi=800)
        plt.close()
        del epochs_tmp
    else:
        mne.set_eeg_reference(epochs, ref_channels=ref, projection=True)
    if apply:
        epochs.apply_proj()
    return epochs


if __name__ == "__main__":
    experiment = "noise"  # "laughter" or "noise"
    DIR = pathlib.Path(os.getcwd())
    with open(DIR / "analysis" / "preproc_config.json") as file:
        cfg = json.load(file)
    # get pilot folder directories.
    pilot_DIR = DIR / "analysis" / "data" / f"pilot_{experiment}"
    fig_path = DIR / "analysis" / "figures" / f"{experiment}"
    # get subject ids
    ids = list(name for name in os.listdir(pilot_DIR)
               if os.path.isdir(os.path.join(pilot_DIR, name)))
    # STEP 1: make raw.fif files and save them into raw_folder.
    for id in ids:  # Iterate through subjects.
        folder_path = pilot_DIR / id
        header_files = folder_path.glob("*.vhdr")
        raw_files = []
        for header_file in header_files:
            raw_files.append(mne.io.read_raw_brainvision(
                header_file, preload=True))  # read BrainVision files.
        raw = mne.concatenate_raws(raw_files)  # make raw files
        # make folders for different file types + preprocessing figures.
        epochs_folder = pilot_DIR / id / "epochs"
        raw_folder = pilot_DIR / id / "raw_data"
        fig_folder = fig_path / id
        evokeds_folder = pilot_DIR / id / "evokeds"
        for folder in epochs_folder, raw_folder, fig_folder, evokeds_folder:
            if not os.path.isdir(folder):
                os.makedirs(folder)
        # Use BrainVision mapping as channel names.
        raw.rename_channels(cfg["mapping"])
        montage_path = pathlib.Path(os.getcwd()) / \
            "analysis" / "AS-96_REF.bvef"  # Use BrainVision montage file to specify electrode positions.
        montage = mne.channels.read_custom_montage(fname=montage_path)
        raw.set_montage(montage)
        raw.save(raw_folder / pathlib.Path(id + "_raw.fif"), overwrite=True)
        # STEP 2: filter and epoch raw data.
        raw = filtering(raw,
                        highpass=cfg["filtering"]["highpass"],
                        lowpass=cfg["filtering"]["lowpass"],
                        notch=cfg["filtering"]["notch"])
        events = mne.events_from_annotations(raw)[0]  # get events
        epochs = mne.Epochs(raw, events, tmin=cfg["epochs"]["tmin"], tmax=cfg["epochs"]["tmax"],
                            event_id=cfg["epochs"][f"event_id_{experiment}"], preload=True)
        del raw  # del raw data to save working memory.
        # STEP 3: rereference epochs. Defaults to average.
        epochs = set_ref(
            epochs, type=cfg["reref"]["type"], ref=cfg["reref"]["ref"])
        # STEP 4: apply ICA for blink and saccade artifact rejection, save epochs.
        reference = mne.preprocessing.read_ica(fname=pathlib.Path(
            os.getcwd()) / "analysis" / "reference_ica.fif")  # reference ICA containing blink and saccade components.
        components = reference.labels_["blinks"]
        ica = ICA(n_components=cfg["ica"]["n_components"],
                  method=cfg["ica"]["method"])
        ica.fit(epochs)
        for component in components:
            mne.preprocessing.corrmap([reference, ica], template=(0, components[component]),
                                      label="blinks", plot=False, threshold=cfg["ica"]["threshold"])
            ica.apply(epochs, exclude=ica.labels_["blinks"])  # apply ICA
        # STEP 5: Apply AutoReject algorithm to reject bad epochs via channel-wise
        # peak to peak ampltiude threshold estimation (cross-validation)
        epochs_clean = autoreject_epochs(
            epochs, n_interpolate=cfg["autoreject"]["n_interpolate"],
            n_jobs=cfg["autoreject"]["n_jobs"],
            cv=cfg["autoreject"]["cv"],
            thresh_method=cfg["autoreject"]["thresh_method"])
        epochs_clean.save(
            epochs_folder / pathlib.Path(id + "-epo.fif"), overwrite=True)
        # STEP 6: average epochs and write evokeds to a file.
        evokeds = [epochs_clean[condition].average()
                   for condition in cfg["epochs"][f"event_id_{experiment}"].keys()]
        mne.write_evokeds(evokeds_folder / pathlib.Path(id +
                          "-ave.fif"), evokeds, overwrite=True)
        # delete data to save working memory.
        del epochs, epochs_clean, ica, reference, components, evokeds
