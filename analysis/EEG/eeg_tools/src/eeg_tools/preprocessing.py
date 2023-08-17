import analysis
import pathlib
import numpy as np
import mne
from matplotlib import pyplot as plt, patches
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA
from meegkit.dss import dss_line_iter


# path = os.getcwd() + "\\src\\" + "eeg_tools"
# sys.path.append(path)

# TODO: describe workflow of processing data.
# TODO: go through all the functions and make them more elegant.
# TODO: implement notch filter and general filtering alternatives.
# TODO: set_ref() uses 65 channels before referencering to calculate snr, one containing only zeros. Exclude the last channel.


def run_pipeline(raw, fig_folder, config, ica_ref=None, exclude_event_id=None):
    """
    Automated preprocessing pipeline for raw EEG data.

    The pipeline takes an mne.io.Raw instance and preprocesses it according to
    the configuration parameters. Currently available preprocessing options are:
    filtering (highpass, lowpass, bandpass, notch);
    epoching;
    rereferencing (robust average, REST, single or linked electrodes);
    ocular artifact rejection (independent component analysis);
    automated threshold rejection (AutoReject).

    Args:
        raw (mne.io.Raw): raw instance containing continuous time series EEG data.
        fig_folder (string): folder path in which preprocessing steps are documented.
        config (dict): JSON file containing preprocessing parameters.
        ica_ref (mne.preprocessing.ica.ICA): ica reference template for artifact rejection via correlation mapping.
        exclude_event_id (int): exclude events by stimulus annotation. Defaults to None.

    Returns:
        epochs (mne.Epochs): preprocessed epoched EEG data.
    """
    global plot  # delete this soon
    global _fig_folder
    _fig_folder = fig_folder
    if not config:
        raise FileNotFoundError(
            "Need config file to preprocess data according to parameters!")
    elif not fig_folder:
        plot = False
    else:
        if "filtering" in config:
            raw = filtering(data=raw, **config["filtering"])
        if "epochs" in config:
            events = mne.pick_events(events=mne.events_from_annotations(raw)[0],
                                     exclude=exclude_event_id)
            epochs = mne.Epochs(raw, events=events,
                                **config["epochs"], preload=True)
            epochs.plot(show=False, show_scalebars=False,
                        show_scrollbars=False, n_channels=20)
            plt.savefig(_fig_folder / pathlib.Path("epochs.jpg"), dpi=800)
            plt.close()
        if "rereference" in config:
            epochs = set_ref(epochs=epochs, **config["rereference"])
        if "ica" in config:
            epochs = apply_ICA(
                epochs=epochs, **config["ica"], reference=ica_ref)
        if "autoreject" in config:
            epochs = autoreject_epochs(epochs=epochs, **config["autoreject"])
    return epochs


def make_raw(header_files, id, fig_folder, mapping, montage, ref_ch="FCz",
             preload=True, add_ref_ch=True, plot=True):
    """
    Merges EEG files into an mne.io.Raw instance.

    Takes the BrainVision header files (format: .vhdr) and merges them subject-wise.
    Applies information about electrode positions and renames them according to the 10-20 system.
    Optionally adds a reference channel to the data which has zero voltage.

    Args:
        header_files (list of strings): header files of one subject (format: .vhdr).
        id (str): subject id.
        ref_ch (str): reference channel to be added.
        preload (bool): preload the data. Defaults to True.
        add_ref_ch (bool): add the reference channel. Defaults to True.
        mapping (dict): JSON file containing the electrode names.
        monateg (dict): JSON file containing information about sensor coordinates.

    Returns:
        raw (mne.io.Raw): continuous time series data which makes up the starting point for the preprocessing pipeline.
    """

    global _fig_folder
    _fig_folder = fig_folder
    raw_files = []
    for header_file in header_files:
        if id in header_file.stem and "block" in header_file.stem:
            raw_files.append(mne.io.read_raw_brainvision(
                header_file, preload=preload))  # read BrainVision files.
    raw = mne.concatenate_raws(raw_files)  # make raw files
    if mapping:
        raw.rename_channels(mapping)
    if add_ref_ch:
        raw.add_reference_channels(ref_ch)
    if montage:
        raw.set_montage(montage)
    if plot is True:
        raw.plot(show=False, show_scrollbars=False,
                 show_scalebars=False, start=2000.0, n_channels=20)
        plt.savefig(_fig_folder / pathlib.Path("raw.jpg"), dpi=800)
        plt.close()
    return raw


def read_raw(raw_folder, id):
    raw_file_name = raw_folder / str(id + "_raw.fif")
    raw = mne.io.read_raw_fif(raw_file_name, preload=True)
    return raw


def filtering(data, notch=None, highpass=None, lowpass=None, nfft=None, plot=True):
    """
    Applies FIR (finite impulse response) filter to the data.

    Filter can be either highpass, lowpass, a combination of those (bandpass) or a notch filter.
    Optionally, save the data plots before and after filtering.

    Args:
        data (mne.io.Raw|mne.Epochs|mne.Evoked): data to be filtered. Usually,
        continuous time series data is preferred to avoid disturbances by the filter design.
        notch (int): frequency to be filtered. Example: 50 to filter out power line noise.
        highpass (int|float): applies a highpass filter at given frequency.
        lowpass (int|float): applies a lowpass filter at given frequency.
        plot (bool): if True, saves the data as a plot before and after filtering.

    Returns:
        mne.io.preprocessing.Raw object
    """

    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle("Power Spectral Density")
    ax[0].set_title("Before filtering")
    ax[1].set_title("After filtering")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    if plot:
        data.plot_psd(ax=ax[0], show=False, exclude=["FCz"])
    if notch is not None:  # ZapLine notch filter not working right now.
        X = data.get_data().T
        # remove power line noise with the zapline algorithm
        X, _ = dss_line_iter(X, fline=notch,
                             sfreq=data.info["sfreq"],
                             nfft=nfft)
        data._data = X.T  # put the data back into variable
        del X
    if lowpass is not None:
        data.filter(h_freq=lowpass, l_freq=None)
    if highpass is not None:
        data.filter(h_freq=None, l_freq=highpass)
    if plot:
        data.plot_psd(ax=ax[1], show=False, exclude=["FCz"])
    if lowpass is not None and highpass == None:
        fig.savefig(
            _fig_folder / pathlib.Path("lowpassfilter.jpg"), dpi=800)
    if highpass is not None and lowpass == None:
        fig.savefig(
            _fig_folder / pathlib.Path("highpassfilter.jpg"), dpi=800)
    if highpass and lowpass is not None:
        fig.savefig(
            _fig_folder / pathlib.Path("bandpassfilter.jpg"), dpi=800)
    if notch is not None:
        fig.savefig(
            _fig_folder / pathlib.Path("ZapLine_filter.jpg"), dpi=800)
    if plot:
        plt.close()
    return data


def set_ref(epochs, ransac_parameters=None, type="average", elecs=None, plot=True):
    """
    Rereferences the data.

    Applies a reference to the data according to the desired type.
    Reference options are:
    robust average over all electrodes, exclude outlier sensors beforehand (RANSAC),
    Reference Electrode Standardization Technique (REST),
    linked mastoids (TP9, TP10),
    or any other desired sensor.

    Args:
        epochs (mne.Epochs): mne.Epochs instance to be rereferenced.
        ransac_parameters (dict|None): parameters for the RANSAC algorithm. (https://autoreject.github.io/stable/generated/autoreject.Ransac.html#autoreject.Ransac)
                                         If None, use default parameters.
        type (str|None): reference type. Can be "average", "lm", "rest" or None.
                         If None, see elecs argument.
        elecs (str|list of str/None): If type == None, enter reference name as string.
                                      If reference should consist of more than one electrode, insert list of strings.
        plot (bool): if True, saves a figure of the data before and after re-referencing.

    Returns:
        mne.Epochs object
    """

    if type == "average":
        epochs_clean = epochs.copy()
        if ransac_parameters:
            ransac = Ransac(**ransac_parameters)
        else:
            ransac = Ransac()
        ransac.fit(epochs_clean)
        epochs_clean.average().plot(exclude=[])
        bads = input(
            "Enter bad sensors here (separate several bad sensors via spacebar): ").split()
        if len(bads) != 0 and bads not in ransac.bad_chs_:
            ransac.bad_chs_.extend(bads)
        epochs_clean = ransac.transform(epochs_clean)
        evoked = epochs.average()
        evoked_clean = epochs_clean.average()
        evoked.info['bads'] = ransac.bad_chs_
        evoked_clean.info['bads'] = ransac.bad_chs_
        if plot:
            fig, ax = plt.subplots(2)
            evoked.plot(exclude=[], axes=ax[0], show=False)
            evoked_clean.plot(exclude=[], axes=ax[1], show=False)
            ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
            ax[1].set_title("After RANSAC")
            fig.tight_layout()
            fig.savefig(_fig_folder / pathlib.Path("RANSAC_results.jpg"), dpi=800)
            plt.close()
        epochs_clean.set_eeg_reference(ref_channels="average", projection=True)
        epochs_clean.apply_proj()
        snr_pre = analysis.snr(epochs)
        snr_post = analysis.snr(epochs_clean)
        epochs_reref = epochs_clean.copy()
    elif type == "rest":
        sphere = mne.make_sphere_model("auto", "auto", epochs.info)
        src = mne.setup_volume_source_space(
            sphere=sphere, exclude=30., pos=5.)
        forward = mne.make_forward_solution(
            epochs.info, trans=None, src=src, bem=sphere)
        epochs_reref = epochs.copy().set_eeg_reference("REST", forward=forward)
        snr_pre = analysis.snr(epochs)
        snr_post = analysis.snr(epochs_reref)
    elif type == "lm":
        epochs_reref = epochs.copy().set_eeg_reference(["TP9", "TP10"])
        snr_pre = analysis.snr(epochs)
        snr_post = analysis.snr(epochs_reref)
    elif type == None:
        epochs_reref = epochs.copy().set_eeg_reference(elecs)
    if plot:
        fig, ax = plt.subplots(2)
        epochs.average().plot(axes=ax[0], show=False)
        epochs_reref.average().plot(axes=ax[1], show=False)
        ax[0].set_title(f"FCz, SNR={snr_pre:.2f}")
        if type is not None:
            ax[1].set_title(f"{type}, SNR={snr_post:.2f}")
            fig.tight_layout()
            fig.savefig(
                _fig_folder / pathlib.Path(f"{type}_reference.jpg"), dpi=800)
            plt.close()
        if type is None:
            ax[1].set_title(f"{elecs}, SNR={snr_post:.2f}")
            fig.tight_layout()
            fig.savefig(
                _fig_folder / pathlib.Path(f"{elecs}_reference.jpg"), dpi=800)
            plt.close()
    return epochs_reref


def apply_ICA(epochs, reference=None, n_components=None, method="fastica",
              threshold="auto", rejection="manual", plot=True):
    """
    Removes brain unrelated activity in the epoched data.

    Applies an independent component analysis by fitting the epoched data into an ICA model, selecting independent components and remove them from the dataset. For detailed documentation about artifact removal using ICA, see: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html.

    Args:
        epochs (mne.Epochs):
        reference (None|str): filepath to the ICA reference.
        n_components (Int|float|None: Number of principal components (from the pre-whitening PCA step) that are passed to the ICA algorithm during fitting.
        method (‘fastica’ | ‘infomax’ | ‘picard’): The ICA method to use in the fit method.
        Use the fit_params argument to set additional parameters. Defaults to 'fastica'
        threshold ("auto"|float between 0 and 1): Minimum correlation percentage when using ICA template as reference.
        When "auto", computes best correlation percentage.
        rejection ("automatic"|"manual"): The rejection method to use. If "manual", plots time course and topographies of the components for the user to decide which component to exclude. If "automatic", ICA reference template must be provided and rejection is done automatically. Defaults to "manual".
        plot (bool): if True, saves plots for documentation.

    Returns:
            mne.Epochs object.
    """

    epochs_ica = epochs.copy()
    snr_pre_ica = analysis.snr(epochs_ica)
    # ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)
    # ar.fit(epochs_ica)
    # epochs_ar, reject_log = ar.transform(epochs_ica, return_log=True)
    if rejection == "automatic":
        ica = ICA(n_components=n_components, method=method)
        # ica.fit(epochs_ica[~reject_log.bad_epochs])
        ica.fit(epochs_ica)
        # reference ICA containing blink and saccade components.
        ref = reference
        # .labels_ dict must contain "blinks" key with int values.
        labels = list(ref.labels_.keys())
        components = list(ref.labels_.values())
        for component, label in zip(components, labels):
            mne.preprocessing.corrmap([ref, ica], template=(0, component[0]),
                                      label=label, plot=False, threshold=threshold)
            ica.apply(epochs_ica, exclude=ica.labels_["blinks"])  # apply ICA
        if plot:
            ica.plot_components(ica.labels_["blinks"], show=False)
            plt.savefig(_fig_folder / pathlib.Path("ICA_components.jpg"), dpi=800)
            plt.close()
            ica.plot_sources(inst=epochs, show=False, start=0,
                             stop=15, show_scrollbars=False)
            plt.savefig(_fig_folder / pathlib.Path(f"ICA_sources.jpg"), dpi=800)
            plt.close()
            snr_post_ica = analysis.snr(epochs_ica)
            ica.plot_overlay(epochs.average(), exclude=ica.labels_["blinks"],
                             show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")
            plt.savefig(_fig_folder / pathlib.Path("ICA_results.jpg"), dpi=800)
            plt.close()
        return epochs_ica
    if rejection == "manual":
        ica = ICA(n_components=n_components, method=method)
        # ica.fit(epochs_ica[~reject_log.bad_epochs])
        ica.fit(epochs_ica)
        ica.plot_components(picks=[x for x in range(20)])
        ica.plot_sources(epochs_ica, start=0, stop=15, show_scrollbars=False, block=True)
        ica.exclude = list((input("Enter components to exclude here (separate several components via spacebar): ").split()))
        ica.exclude = [int(x) for x in ica.exclude]
        ica.apply(epochs_ica, exclude=ica.exclude)
        if plot:
            ica.plot_components(ica.exclude, show=False)
            plt.savefig(_fig_folder / pathlib.Path("ICA_components.jpg"), dpi=800)
            plt.close()
            ica.plot_sources(inst=epochs, show=False, start=0,
                             stop=15, show_scrollbars=False)
            plt.savefig(_fig_folder / pathlib.Path(f"ICA_sources.jpg"), dpi=800)
            plt.close()
            snr_post_ica = analysis.snr(epochs_ica)
            ica.plot_overlay(epochs.average(), exclude=ica.exclude,
                             show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")
            plt.savefig(_fig_folder / pathlib.Path("ICA_results.jpg"), dpi=800)
            plt.close()
        return epochs_ica


def autoreject_epochs(epochs,
                      n_interpolate=[1, 4, 32],
                      consensus=None,
                      cv=10,
                      thresh_method="bayesian optimization",
                      n_jobs=-1,
                      random_state=None,
                      plot=True):
    """
    Automatically rejects epochs in the data based on peak-to-peak threshold estimation specifically for each epoch channel-wise.

    Args:
        epochs (mne.Epochs): The data to process.
        n_interpolate (None|array): The values to try for the number of channels for which to interpolate. This is rho. If None, defaults to np.array([1, 4, 32])
        consensus (None|array): The values to try for percentage of channels that must agree as a fraction of the total number of channels. This sets kappa. If None, defaults to np.linspace(0, 1.0, 11).
        cv (int|sklearn.model_selection object): Defaults to cv=10.
        thresh_method (str): ‘bayesian_optimization’ or ‘random_search’.
        n_jobs (int): The number of jobs. n_jobs = -1 uses maximum amount of parallel jobs, resulting in fastest computation.
        random_state (int|np.random.RandomState|None): The seed of the pseudo random number generator to use. Defaults to None.
        plot (bool): if True, saves plots for documentation of the processing step.

    Returns:
            mne.Epochs oject.
    """

    ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    if plot:
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
        fig.savefig(_fig_folder / pathlib.Path("autoreject_best_fit.jpg"), dpi=800)
        plt.close()
    evoked_bad = epochs[reject_log.bad_epochs].average()
    snr_ar = analysis.snr(epochs_ar)
    if plot:
        plt.plot(evoked_bad.times, evoked_bad.data.T * 1e06, 'r', zorder=-1)
        epochs_ar.average().plot(axes=plt.gca(), show=False)
        plt.savefig(
            _fig_folder / pathlib.Path("autoreject_results.jpg"), dpi=800)
        plt.close()
        epochs_ar.plot_drop_log(show=False)
        plt.savefig(
            _fig_folder / pathlib.Path("epochs_drop_log.jpg"), dpi=800)
        plt.close()
    return epochs_ar


def make_evokeds(epochs, plot=True, baseline=None):
    """
    Generate evoked responses from epoched data by averaging. Optionally apply a baseline
    Args:
        epochs:
        plot:
        baseline:

    Returns:
            mne.Evoked object.
    """

    if baseline is not None:
        epochs.apply_baseline(baseline)
    # evokeds = [epochs[condition].average()
    #           for condition in epochs.event_id.keys()]
    evokeds = epochs.average(by_event_type=True)
    if plot is True and _fig_folder is not None:
        snr = analysis.snr(epochs)
        avrgd = mne.grand_average(evokeds)
        avrgd.plot_joint(show=False, title=f"SNR: {snr:.2f}")
        plt.savefig(_fig_folder / pathlib.Path("evokeds.jpg", dpi=800))
        plt.close()
    return evokeds
