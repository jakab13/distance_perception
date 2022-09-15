from pathlib import Path
import numpy as np
from scipy import stats
from mne import read_epochs, concatenate_epochs
root = Path(__file__).parent.parent.absolute()


def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=axis), stats.sem(a, axis=axis)
    # standard error * critical value from the t-distribution
    ci = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return ci


def smooth(y, pts=20, pad=True):
    box = np.ones(pts)/pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def apply_transform(data, transforms):
    if not isinstance(transforms, list):
        transforms = [transforms]
    n_epochs, n_channels, n_times = data.shape
    data = data.transpose(1, 0, 2)
    data = data.reshape(n_channels, n_epochs * n_times).T
    for i, transform in enumerate(transforms):
        if i == 0:
            transformed = data @ transform
        else:
            transformed = transformed @ transform
    transformed = np.reshape(transformed.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
    return transformed


def line(a, b, x):
    return a + b*x


def surrograte_data(subject, experiment="freefield"):
    epochs = get_epochs(subject)
    n_epochs, n_channels, n_samples = epochs.get_data().shape
    data = epochs.get_data().reshape(n_channels, n_epochs*n_samples)
    times = np.arange(0, n_epochs*n_samples-n_samples, 1)
    times = np.random.choice(times, n_epochs)
    data = [data[:, t:t+n_samples] for t in times]
    data = np.hstack(data)
    epochs._data = data.reshape(n_epochs, n_channels, n_samples)
    return epochs


def get_epochs(subject, experiment="freefield", pick=None, resample=False,
               blocks=True, targets=True, post_targets=False,
               dualadapt_conditions=["mid"]):
    data_folder = Path(root/experiment/"input"/"epochs"/subject)
    all_epochs = []
    file_list = []
    if experiment == "freefield":
        if blocks:
            file_list.append(list(data_folder.glob("block[0-9]-epo.fif")))
        if targets:
            file_list.append(list(data_folder.glob("targets[0-9]-epo.fif")))
        if post_targets:
            file_list.append(list(data_folder.glob("post_targets[0-9]-epo.fif")))
    elif experiment == "dualadapt":
        for condition in dualadapt_conditions:
            file_list.append(list(data_folder.glob(f"{condition}-epo.fif")))
    file_list = [item for sublist in file_list for item in sublist]
    for f in file_list:
        epochs = read_epochs(f, preload=True)
        if resample:
            epochs.resample(resample)
        if pick is not None:
            epochs.pick_channels(pick)
        all_epochs.append(epochs)
    return concatenate_epochs(all_epochs)


def get_transforms(subject, experiment="freefield", suffix=""):
    data_folder = Path(root / experiment / "input" / "transforms" / subject)
    to_jd1 = np.load(str(data_folder/f"to_jd1{suffix}.npy"),
                     allow_pickle=True)
    from_jd1 = np.load(str(data_folder/f"from_jd1{suffix}.npy"),
                       allow_pickle=True)
    to_jd2 = np.load(str(data_folder/f"to_jd2{suffix}.npy"),
                     allow_pickle=True)
    from_jd2 = np.load(str(data_folder/f"from_jd2{suffix}.npy"),
                       allow_pickle=True)
    return to_jd1, from_jd1, to_jd2, from_jd2


def get_clustertest(subject, experiment="freefield", condition="a37.5"):
    """
    Get the results from the permutation cluster test.
    Arguments:
        subject (str): Name of the subject.
        experiment (str): Name of the experiment ("freefield" or "dualadapt").
        condition (str): Only relevant of experiment=="dualadapt"
                         ("a37.5", "a-37.5", "s12.5", "s-12.5" or "gaze").
    Returns:
        (np.ndarray): F-statistic of shape samples x channels.
        (list): Supra-threshold clusters. Element are tuples of two arrays,
                containing the samples and channels belonging to the cluster.
        (list): P-value of each cluster.
    """
    data_folder = Path(root / experiment / "output" / subject)
    if experiment == "freefield":
        data = np.load(str(data_folder/"permutation_test_results.npy"),
                       allow_pickle=True).item()
        statistic, clusters, p_values = \
            data["statistic"], data["clusters"], data["clusters_p"]
    else:
        data = np.load(str(data_folder/f"permutation_test_{condition}.npy"),
                       allow_pickle=True).item()
        statistic, clusters, p_values = \
            data["stat"], data["clusters"], data["clusters_p"]
    return statistic, clusters, p_values


def pca(data):
    data -= data.mean(axis=0)
    matrix = np.cov(data.T)
    eigenvals, eigenvecs = np.linalg.eig(matrix)
    sort = eigenvals.argsort()[::-1]
    return eigenvals[sort], eigenvecs[sort]


def get_response(subject, experiment="freefield", dropnan=True):
    in_folder = Path(root/experiment/"input"/"raw"/subject)
    response = []
    for response_file in in_folder.glob("block[0-9]-response.npy"):
        response.append(np.load(str(response_file)))
    response = np.concatenate(response)
    if dropnan:
        response = response[~np.isnan(response[:, 1])]
    return response


def peak_latency(evoked, tmin, tmax, mode="median"):
    peaks = np.zeros(len(evoked.info["ch_names"]))
    for i, ch in enumerate(evoked.info["ch_names"]):
        channel_data = evoked.copy()
        channel_data.pick_channels([ch])
        peaks[i] = channel_data.get_peak(tmin=tmin, tmax=tmax)[1]
    if mode == "median":
        return np.median(peaks)
    elif mode == "mean":
        return np.mean(peaks)
