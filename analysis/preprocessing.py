# from meegkit.dss import dss_line_iter
import glob
import json
from mne.preprocessing import ICA
from autoreject import AutoReject, Ransac
from matplotlib import pyplot as plt, patches
import mne
import numpy as np
import pathlib
import os

# TODO: fix notch filter in filtering function (zapline? Doesnt work ATM).
# TODO: implement function to search for files in project folder.
# TODO: implement other way to calculate SNR. Maybe ask Alessandro?

def snr(epochs):
    """
    Compute signal-to-noise ratio. Take root mean square of noise
    (interval before stimulus onset) and signal (interval where evoked activity is expected)
    and return quotient.
    """
    signal = epochs.copy().crop(0, 0.4).average().get_data()
    noise = epochs.copy().crop(None, 0).average().get_data()
    signal_rms = np.sqrt(np.mean(signal**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    return signal_rms / noise_rms


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
    ar.fit(epochs)
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
    epochs_ar.average().plot(axes=plt.gca(), show=False,
                             titles=f"SNR: {snr_ar:.2f}")
    plt.savefig(
        fig_folder / pathlib.Path("autoreject_results.pdf"), dpi=800)
    plt.close()
    epochs_ar.plot_drop_log(show=False)
    plt.savefig(
        fig_folder / pathlib.Path("epochs_drop_log.pdf"), dpi=800)
    plt.close()
    return epochs_ar


def reref(epochs, type="average", n_jobs=-1, n_resample=50, min_channels=0.25,
          min_corr=0.75, unbroken_time=0.4, plot=True):
    """
    If type "average": Create a robust average reference by first interpolating the bad channels
    to exclude outliers. Take mean voltage over all inlier channels as reference.
    If type "rest": use reference electrode standardization technique (point at infinity).
    epochs: mne.Epoch object.
    type: string --> "average", "rest", "lm" (linked mastoids)
    """
    if type == "average":
        epochs_clean = epochs.copy()
        ransac = Ransac(n_jobs=n_jobs, n_resample=n_resample, min_channels=min_channels,
                        min_corr=min_corr, unbroken_time=unbroken_time)  # optimize speed
        ransac.fit(epochs_clean)
        epochs_clean.average().plot(exclude=[])
        bads = input("Sanity check for obvious bad sensors: ").split()
        if len(bads) != 0:
            if bads not in ransac.bad_chs_:
                ransac.bad_chs_.extend(bads)
        epochs_clean = ransac.transform(epochs_clean)
        evoked = epochs.average()
        evoked_clean = epochs_clean.average()
        evoked.info['bads'] = ransac.bad_chs_
        evoked_clean.info['bads'] = ransac.bad_chs_
        fig, ax = plt.subplots(2)
        evoked.plot(exclude=[], axes=ax[0], show=False)
        evoked_clean.plot(exclude=[], axes=ax[1], show=False)
        ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
        ax[1].set_title("After RANSAC")
        fig.tight_layout()
        fig.savefig(
            fig_folder / pathlib.Path("RANSAC_results.pdf"), dpi=800)
        plt.close()
        epochs = epochs_clean.copy()
        epochs_clean.set_eeg_reference(ref_channels="average", projection=True)
        average_reference = epochs_clean.info["projs"]
        epochs_clean.add_proj(average_reference)
        epochs_clean.apply_proj()
        snr_pre = snr(epochs)
        snr_post = snr(epochs_clean)
        epochs_reref = epochs_clean.copy()
    if type == "rest":
        sphere = mne.make_sphere_model("auto", "auto", epochs.info)
        src = mne.setup_volume_source_space(
            sphere=sphere, exclude=30., pos=5.)
        forward = mne.make_forward_solution(
            epochs.info, trans=None, src=src, bem=sphere)
        epochs_reref = epochs.copy().set_eeg_reference("REST", forward=forward)
        snr_pre = snr(epochs)
        snr_post = snr(epochs_reref)
    if type == "lm":
        epochs_reref = epochs.copy().set_eeg_reference(["TP9", "TP10"])
        snr_pre = snr(epochs)
        snr_post = snr(epochs_reref)
    if plot == True:
        fig, ax = plt.subplots(2)
        epochs.average().plot(axes=ax[0], show=False)
        epochs_reref.average().plot(axes=ax[1], show=False)
        ax[0].set_title(f"FCz, SNR={snr_pre:.2f}")
        ax[1].set_title(f"{type}, SNR={snr_post:.2f}")
        fig.tight_layout()
        fig.savefig(
            fig_folder / pathlib.Path(f"{type}_reference.pdf"), dpi=800)
        plt.close()
    return epochs_reref


def apply_ICA(epochs, reference, n_components=None, method="fastica",
              threshold="auto", n_interpolate=None):
    """
    Run independent component analysis. Fit all epochs to the mne.ICA class, use
    reference_ica.fif to show the algorithm how blinks and saccades look like.
    Apply ica and save components to keep track of the excluded component topography.
    """
    epochs_ica = epochs.copy()
    snr_pre_ica = snr(epochs_ica)
    # ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)
    # ar.fit(epochs_ica)
    # epochs_ar, reject_log = ar.transform(epochs_ica, return_log=True)
    ica = ICA(n_components=n_components, method=method)
    # ica.fit(epochs_ica[~reject_log.bad_epochs])
    ica.fit(epochs_ica)
    # reference ICA containing blink and saccade components.
    reference = mne.preprocessing.read_ica(fname=reference)
    # .labels_ dict must contain "blinks" key with int values.
    components = reference.labels_["blinks"]
    for component in components:
        mne.preprocessing.corrmap([reference, ica], template=(0, components[component]),
                                  label="blinks", plot=False, threshold=cfg["ica"]["threshold"])
        ica.apply(epochs_ica, exclude=ica.labels_["blinks"])  # apply ICA
    ica.plot_components(ica.labels_["blinks"], show=False)
    plt.savefig(fig_folder / pathlib.Path("ICA_components.pdf"), dpi=800)
    plt.close()
    ica.plot_sources(inst=epochs, show=False, start=0, stop=10, show_scrollbars=False)
    plt.savefig(fig_folder / pathlib.Path(f"ICA_sources.pdf"), dpi=800)
    plt.close()
    snr_post_ica = snr(epochs_ica)
    ica.plot_overlay(epochs.average(), exclude=ica.labels_["blinks"],
                     show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")
    plt.savefig(fig_folder / pathlib.Path("ICA_results.pdf"), dpi=800)
    plt.close()
    return epochs_ica


if __name__ == "__main__":
    experiment = "vocal_effort"  # "vocal_effort" or "noise" data.
    DIR = pathlib.Path(os.getcwd())
    with open(DIR / "analysis" / "preproc_config.json") as file:
        cfg = json.load(file)
    with open(DIR / "analysis" / "mapping.json") as file:
        mapping = json.load(file)
    # get pilot folder directories.
    data_DIR = DIR / "analysis" / "data" / f"{experiment}"
    fig_path = DIR / "analysis" / "figures" / f"{experiment}"
    # get subject ids
    ids = list(name for name in os.listdir(data_DIR)
               if os.path.isdir(os.path.join(data_DIR, name)))
    # STEP 1: make raw.fif files and save them into raw_folder.
    for id in ids[0]:  # Iterate through subjects.
        folder_path = data_DIR / id
        header_files = folder_path.glob("*.vhdr")
        raw_files = []
        for header_file in header_files:
            raw_files.append(mne.io.read_raw_brainvision(
                header_file, preload=True))  # read BrainVision files.
        raw = mne.concatenate_raws(raw_files)  # make raw files
        # make folders for different file types + preprocessing figures.
        epochs_folder = data_DIR / id / "epochs"
        raw_folder = data_DIR / id / "raw_data"
        fig_folder = fig_path / id
        evokeds_folder = data_DIR / id / "evokeds"
        for folder in epochs_folder, raw_folder, fig_folder, evokeds_folder:
            if not os.path.isdir(folder):
                os.makedirs(folder)
        # Use BrainVision mapping as channel names.
        raw.rename_channels(mapping)
        # Use BrainVision montage file to specify electrode positions.
        montage_path = DIR / "analysis" / cfg["montage"]["name"]
        montage = mne.channels.read_custom_montage(fname=montage_path)
        raw.set_montage(montage)
        raw.save(raw_folder / pathlib.Path(id + "_raw.fif"), overwrite=True)
        # STEP 2: bandpass filter at 1 - 40 Hz and epoch raw data.
        raw = filtering(raw,
                        highpass=cfg["filtering"]["highpass"],
                        lowpass=cfg["filtering"]["lowpass"],
                        notch=cfg["filtering"]["notch"])  # bandpass filter
        events = mne.events_from_annotations(raw)[0]  # get events
        epochs = mne.Epochs(raw, events, tmin=cfg["epochs"]["tmin"], tmax=cfg["epochs"]["tmax"],
                            event_id=cfg["epochs"][f"event_id_{experiment}"], preload=True,
                            baseline=cfg["epochs"]["baseline"], detrend=cfg["epochs"]["detrend"])  # apply baseline
        del raw  # del raw data to save working memory.
        # STEP 3: rereference epochs.
        epochs_ref = reref(
            epochs, type=cfg["reref"]["type"], n_jobs=cfg["reref"]["ransac"]["n_jobs"],
            n_resample=cfg["reref"]["ransac"]["n_resample"],
            min_channels=cfg["reref"]["ransac"]["min_channels"],
            min_corr=cfg["reref"]["ransac"]["min_corr"],
            unbroken_time=cfg["reref"]["ransac"]["unbroken_time"],
            plot=cfg["reref"]["plot"])
        # STEP 4: apply ICA for blink and saccade artifact rejection.
        epochs_ica = apply_ICA(epochs_ref, reference=DIR / 'analysis' / cfg["ica"]["reference"],
                               n_components=cfg["ica"]["n_components"],
                               threshold=cfg["ica"]["threshold"],
                               method=cfg["ica"]["method"])
        # STEP 5: Apply AutoReject algorithm to reject bad epochs via channel-wise
        # peak to peak ampltiude threshold estimation (cross-validation).
        epochs_clean = autoreject_epochs(
            epochs_ica, n_interpolate=cfg["autoreject"]["n_interpolate"],
            n_jobs=cfg["autoreject"]["n_jobs"],
            cv=cfg["autoreject"]["cv"],
            thresh_method=cfg["autoreject"]["thresh_method"])
        epochs_clean.apply_baseline((None, 0))
        epochs_clean.average().plot_image(
            titles=f"SNR:{snr(epochs_clean):.2f}", show=False)
        plt.savefig(
            fig_folder / pathlib.Path("clean_evoked_image.pdf"), dpi=800)
        plt.close()
        epochs_clean.save(
            epochs_folder / pathlib.Path(id + "-epo.fif"), overwrite=True)
        # STEP 6: average epochs and write evokeds to a file.
        evokeds = [epochs_clean[condition].average()
                   for condition in cfg["epochs"][f"event_id_{experiment}"].keys()]
        mne.write_evokeds(evokeds_folder / pathlib.Path(id +
                          "-ave.fif"), evokeds, overwrite=True)
        # delete data to save working memory.
        del epochs, epochs_ref, epochs_ica, epochs_clean, evokeds
