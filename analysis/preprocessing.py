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

def filtering(data, notch=None, highpass=None, lowpass=None):
    """
    Apply FIR filter to the raw dataset. Make a 2 by 2 plot with time
    series data and power spectral density before and after.
    """
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle("Power Spectral Density")
    ax[0].set_title("before filtering")
    ax[1].set_title("after filtering")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    data.plot_psd(ax=ax[0], show=False)
    if notch is not None:  # ZapLine Notch filter
        X = data.get_data().T
        # remove power line noise with the zapline algorithm
        X, _ = dss_line_iter(X, fline=cfg["filtering"]["notch"],
                             sfreq=data.info["sfreq"],
                             nfft=cfg["filtering"]["nfft"])
        data._data = X.T  # put the data back into variable
        del X
    if lowpass is not None:
        data.filter(h_freq=lowpass, l_freq=None)
    if highpass is not None:
        data.filter(h_freq=None, l_freq=highpass)
    data.plot_psd(ax=ax[1], show=False)
    fig.tight_layout()
    if lowpass is not None and highpass == None:
        fig.savefig(
            fig_folder / pathlib.Path("lowpassfilter.pdf"), dpi=800)
    if highpass is not None and lowpass == None:
        fig.savefig(
            fig_folder / pathlib.Path("highpassfilter.pdf"), dpi=800)
    if highpass and lowpass is not None:
        fig.savefig(
            fig_folder / pathlib.Path("bandpassfilter.pdf"), dpi=800)
    if notch is not None:
        fig.savefig(
            fig_folder / pathlib.Path("ZapLine_filter.pdf"), dpi=800)
    plt.close()
    return data


def autoreject_epochs(epochs,
                      n_interpolate=[1, 4, 8, 16],
                      consensus=None,
                      cv=10,
                      thresh_method="bayesian optimization",
                      n_jobs=-1,
                      random_state=None):
    """
    Automatically reject epochs via AutoReject algorithm:
    Computation of sensor-wise peak-to-peak-amplitude thresholds
    via cross-validation.
    """
    ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)
    ar.fit(epochs[:50])
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
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
    fig.savefig(fig_folder / pathlib.Path("autoreject_best_fit.pdf"), dpi=800)
    plt.close()
    evoked_bad = epochs[reject_log.bad_epochs].average()
    snr_ar = snr(epochs_ar)
    plt.plot(evoked_bad.times, evoked_bad.data.T * 1e06, 'r', zorder=-1)
    epochs_ar.average().plot(axes=plt.gca(), show=False, titles=f"SNR: {snr_ar:.2f}")
    plt.savefig(
        fig_folder / pathlib.Path("autoreject_results.pdf"), dpi=800)
    plt.close()
    epochs_ar.plot_drop_log(show=False)
    plt.savefig(
        fig_folder / pathlib.Path("epochs_drop_log.pdf"), dpi=800)
    plt.close()
    return epochs_ar

def reref(epochs, type="average", n_jobs=-1):
    """
    If type "average": Create a robust average reference by first interpolating the bad channels
    to exclude outliers.
    If type "rest": use reference electrode standardization technique(point at infinity)
    epochs: mne.Epoch object.
    type: string --> "average", "rest", "mastoids"
    ref: string/list of strings. List of reference electrode names. Example: ["P9", "P10"]
    """
    if type == "average":
        epochs_clean = epochs.copy()
        ransac = Ransac(n_jobs=n_jobs)  # optimize speed
        epochs_clean = ransac.fit_transform(epochs_clean)
        epochs_clean.info["bads"] = ransac.bad_chs_
        epochs_clean.set_eeg_reference(ref_channels="average", projection=True)
        evoked = epochs.average()  # for plotting
        evoked_clean = epochs_clean.average()
        evoked.info["bads"] = ransac.bad_chs_
        average_reference = epochs_clean.info["projs"]
        epochs_clean.add_proj(average_reference)
        epochs_clean.apply_proj()
        fig, ax = plt.subplots(2)
        evoked.plot(exclude=[], axes=ax[0], show=False)
        ax[0].set_title('Before RANSAC')
        evoked_clean.plot(exclude=[], axes=ax[1], show=False)
        ax[1].set_title("After RANSAC")
        fig.tight_layout()
        fig.savefig(
            fig_folder / pathlib.Path("RANSAC_results.pdf"), dpi=800)
        plt.close()
        snr_pre = snr(epochs)
        snr_post = snr(epochs_clean)
        epochs.average().plot(axes=ax[0], show=False)
        ax[0].set_title(f"Original, SNR={snr_pre:.2f}")
        epochs_clean.average().plot(axes=ax[1], show=False)
        ax[1].set_title(f"AVG, SNR={snr_post:.2f}")
        fig.tight_layout()
        fig.savefig(
            fig_folder / pathlib.Path("AVG_reference.pdf"), dpi=800)
        plt.close()
        return epochs_clean
    if type == "rest":
        sphere = mne.make_sphere_model("auto", "auto", epochs.info)
        src = mne.setup_volume_source_space(
            sphere=sphere, exclude=30., pos=5.)
        forward = mne.make_forward_solution(
            epochs.info, trans=None, src=src, bem=sphere)
        epochs_rest = epochs.copy().set_eeg_reference("REST", forward=forward)
        fig, ax = plt.subplots(2)
        snr_pre = snr(epochs)
        snr_post = snr(epochs_rest)
        epochs.average().plot(axes=ax[0], show=False)
        ax[0].set_title(f"Original, SNR={snr_pre:.2f}")
        epochs_rest.average().plot(axes=ax[1], show=False)
        ax[1].set_title(f"REST, SNR={snr_post:.2f}")
        fig.tight_layout()
        fig.savefig(
            fig_folder / pathlib.Path("REST_reference.pdf"), dpi=800)
        plt.close()
        return epochs_rest
    if type == "mastoids":
        epochs_ref = epochs.copy().set_eeg_reference(["TP9", "TP10"])
        fig, ax = plt.subplots(2)
        snr_pre = snr(epochs)
        snr_post = snr(epochs_ref)
        epochs.average().plot(axes=ax[0], show=False)
        ax[0].set_title(f"Original, SNR={snr_pre:.2f}")
        epochs_ref.average().plot(axes=ax[1], show=False)
        ax[1].set_title(f"Mastoids, SNR={snr_post:.2f}")
        fig.tight_layout()
        fig.savefig(
            fig_folder / pathlib.Path("mastoids_reference.pdf"), dpi=800)
        plt.close()
        return epochs

def apply_ICA(epochs, reference, n_components=None, method="fastica",
              threshold="auto", n_interpolate=None, n_jobs=-1):
    """
    Run AutoReject, fit ICA on bad epochs to only include eye movement contaminated
    epochs plus noisy brain data. ICA works best when only "critical" data is
    shown.
    """
    epochs_ica = epochs.copy()
    snr_pre_ica = snr(epochs_ica)
    ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)
    ar.fit(epochs_ica[:50])
    epochs_ar, reject_log = ar.transform(epochs_ica, return_log=True)
    ica = ICA(n_components=n_components, method=method)
    ica.fit(epochs_ica[~reject_log.bad_epochs])
    # reference ICA containing blink and saccade components.
    reference = mne.preprocessing.read_ica(fname=reference)
    # .labels_ dict must contain "blinks" key with int values.
    components = reference.labels_["blinks"]
    for component in components:
        mne.preprocessing.corrmap([reference, ica], template=(0, components[component]),
                                  label="blinks", plot=False, threshold=cfg["ica"]["threshold"])
        ica.apply(epochs_ica, exclude=ica.labels_["blinks"])  # apply ICA
        ica.plot_components(ica.labels_["blinks"], show=False)
        plt.savefig(fig_folder / pathlib.Path("ica_components.pdf"), dpi=800)
        plt.close()
    snr_post_ica = snr(epochs_ica)
    ica.plot_overlay(epochs.average(), exclude=ica.labels_["blinks"],
                     show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")
    plt.savefig(fig_folder / pathlib.Path("ICA_results.pdf"), dpi=800)
    plt.close()
    return epochs_ica


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
